use std::collections::HashMap;

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;

use lsp_types::{
  CodeAction, DidChangeTextDocumentParams, DidOpenTextDocumentParams,
  FileRename, Range, RenameFilesParams, TextDocumentIdentifier,
  TextDocumentPositionParams, Uri, WorkspaceEdit,
};

use serde::Deserialize;
use serde_json::Value;
// For deserializing query results
use crate::utils::TOKIO_RUNTIME;
use surrealdb::engine::remote::ws::{Client, Ws}; // Or other client type based on endpoint
use surrealdb::opt::auth::Root; // Or other auth strategy
use surrealdb::sql;
use surrealdb::sql::{Idiom, Param};
use surrealdb::Surreal;
use tokio::sync::oneshot;
use tracing::{error, info, warn};

use splitter_tree_sitter::{Chunk as TreeSitterChunk, TreeSitterCodeSplitter};
use text_splitter::{ChunkConfig, TextSplitter};
use tree_sitter::{Parser, Tree};

use super::{
  ContextAndCodePrompt, FIMPrompt, MemoryBackend, Prompt, PromptType,
};

use crate::config::{
  Config, SurrealDbConfig, TextSplitter as TextSplitterConfig,
  TreeSitter as TreeSitterConfig, ValidSplitter,
};

fn get_language_for_uri(uri: &Uri) -> Option<tree_sitter::Language> {
  warn!("Placeholder get_language_for_uri called for {:#?}", uri);
  // Example:
  // let extension = uri.path().split('.').last()?;
  // match extension {
  //     "rs" => Some(tree_sitter_rust::language()),
  //     "py" => Some(tree_sitter_python::language()),
  //     _ => None,
  // }
  None // Default to None if language not found/supported
}

#[derive(Debug, Clone)]
pub(crate) struct SurrealDbStore {
  db: Surreal<Client>,
  config: SurrealDbConfig,
}

// this will be created from the lsp config in languages.toml in helix
impl SurrealDbStore {
  pub(crate) async fn new(
    config: SurrealDbConfig,
    _main_config: Config,
  ) -> Result<Self> {
    info!(
      "Attempting to connect to SurrealDB at endpoint: {}",
      config.endpoint
    );
    let db = Surreal::new::<Ws>(&config.endpoint)
      .await
      .context("Failed to init SurrealDB engine")?;
    db.use_ns(config.namespace.as_deref().unwrap_or("default"))
      .use_db(config.database.as_deref().unwrap_or("default"))
      .await
      .context("Failed to switch SurrealDB NS/DB")?;
    if let (Some(user), Some(pass)) =
      (config.username.as_ref(), config.password.as_ref())
    {
      info!("Attempting SurrealDB signin for user: {}", user);
      db.signin(surrealdb::opt::auth::Root {
        username: user,
        password: pass,
      })
      .await
      .context("SurrealDB signin failed")?;
      info!("SurrealDB signin successful.");
    } else {
      info!("Connecting to SurrealDB without authentication.");
    }

    info!(
      "SurrealDB connection successful. NS='{}', DB='{}'",
      config.namespace.as_deref().unwrap_or("default"),
      config.database.as_deref().unwrap_or("default")
    );

    let model_name_str = config.embedding_model_name.clone(); // Clone model name string
    info!(
      "Checking/Defining SurrealML embedding model: {}",
      model_name_str
    );

    // 1. Check if model already exists using INFO FOR query
    let info_query = "INFO FOR MODEL $model;";
    // Create HashMap for binding the $model variable
    let vars: HashMap<String, sql::Value> = HashMap::from([
      ("model".into(), sql::Value::from(model_name_str.clone())), // Bind key "model" to model name string
    ]);

    info!("Executing INFO FOR MODEL query...");
    // Execute the INFO query
    let mut info_response = match db.query(info_query).bind(vars).await {
      Ok(response) => {
        info!("INFO FOR MODEL query successful.");
        response
      }
      Err(e) => {
        error!("Failed to execute 'INFO FOR MODEL' query: {}", e);
        return Err(anyhow!(e).context("Failed to query model info"));
      }
    };

    // INFO FOR returns an array of results; we expect one result for the specified model name.
    // That result itself is an Option<Value>. If the model doesn't exist, the result is Ok(None).
    // If it exists, the result is Ok(Some(Value)), where Value contains model details.
    let model_info_option: Option<Value> = match info_response.take(0) {
      // take(0) gets the first resultset
      Ok(info_opt) => {
        info!("Deserialized INFO FOR MODEL response successfully.");
        info_opt // This is the Option<Value> we need
      }
      Err(e) => {
        error!("Failed to deserialize 'INFO FOR MODEL' response: {}", e);
        return Err(anyhow!(e).context("Deserializing model info failed"));
      }
    };

    // Check if the Option<Value> is Some (meaning the model exists)
    if model_info_option.is_none() {
      info!(
        "Model '{}' not found in SurrealDB. Attempting to define it...",
        model_name_str
      );

      // 2. Construct the DEFINE MODEL query dynamically (as before)
      let source_clause = if config.model_onnx_location.starts_with("http") {
        format!("URL '{}'", config.model_onnx_location)
      } else {
        let path = config
          .model_onnx_location
          .strip_prefix("file://")
          .unwrap_or(&config.model_onnx_location);
        format!("FILE '{}'", path)
      };
      let input_def = "(text string)";
      let output_def =
        format!("(embedding vector<float, {}>)", config.embedding_dimension);

      // Use the actual model name string in the DEFINE statement
      let define_query = format!(
        "DEFINE MODEL {} TYPE ONNX {} INPUT {} OUTPUT {};",
        model_name_str, // Use the variable holding the name
        source_clause,
        input_def,
        output_def
      );
      info!("Executing Define Model Query: {}", define_query);

      // 3. Execute the DEFINE MODEL query (no bindings needed here)
      match db.query(&define_query).await {
        Ok(_) => info!(
          "Successfully defined model '{}' in SurrealDB.",
          model_name_str
        ),
        Err(e) => {
          error!("Failed to define model '{}': {}", model_name_str, e);
          return Err(anyhow!(e).context(format!(
            "Failed to define SurrealDB model '{}'",
            model_name_str
          )));
        }
      }
    } else {
      info!("Model '{}' already exists in SurrealDB.", model_name_str);
      // Optionally log details from model_info_option if needed:
      // info!("Existing model info: {:?}", model_info_option.unwrap());
    }
    // --- >>> END Define Model Logic <<< ---

    Ok(Self { db, config })
  }

  async fn index_document(
    db: Surreal<Client>,
    config: SurrealDbConfig,
    uri: Uri,
    text: String,
  ) -> Result<()> {
    let uri_string = uri.to_string();
    info!("[INDEX_TASK] Starting indexing for {}", uri_string);

    // 1. Delete existing chunks
    // ... (delete query as before) ...
    let delete_query = "DELETE type::table($tb) WHERE uri = $uri;";
    let vars_del: HashMap<String, sql::Value> = HashMap::from([
      ("tb".into(), sql::Value::from(config.table_name.clone())),
      ("uri".into(), sql::Value::from(uri_string.clone())),
    ]);

    db.query(delete_query)
      .bind(vars_del)
      .await
      .context("Failed to delete old chunks")?;
    info!("[INDEX_TASK] Deleted old chunks for {}", uri_string);

    // 2. Chunk the new text based on configured splitter
    let chunks: Vec<String>; // Store results as owned Strings

    match &config.splitter {
      ValidSplitter::TextSplitter(ts_config) => {
        info!(
          "[INDEX_TASK] Using TextSplitter with chunk size: {}",
          ts_config.chunk_size
        );

        let splitter = TextSplitter::new(
          // Pass ChunkConfig to new()
          ChunkConfig::new(ts_config.chunk_size) // Create config with size
            .with_trim(true), // Configure trimming on the config
        );
        // --- End Correction ---
        chunks = splitter.chunks(&text).map(String::from).collect();
      }

      ValidSplitter::TreeSitter(ts_config) => {
        info!("[INDEX_TASK] Attempting TreeSitter chunking (size: {}, overlap: {})...", ts_config.chunk_size, ts_config.chunk_overlap);
        // --- TreeSitter Logic ---
        // a. Get Language
        if let Some(language) = get_language_for_uri(&uri) {
          // b. Create Parser
          let mut parser = Parser::new();
          parser.set_language(&language).with_context(|| {
            format!("Failed to set tree-sitter language for {}", uri_string)
          })?;
          // c. Parse into Tree
          if let Some(tree) = parser.parse(&text, None) {
            info!(
              "[INDEX_TASK] Successfully parsed {} with tree-sitter.",
              uri_string
            );
            // d. Use the TreeSitterCodeSplitter from the separate crate/module
            // Assuming the splitter code user provided is in e.g., crate::splitters::tree_sitter
            match TreeSitterCodeSplitter::new(
              ts_config.chunk_size,
              ts_config.chunk_overlap,
            ) {
              Ok(splitter) => {
                match splitter.split(&tree, text.as_bytes()) {
                  Ok(ts_chunks) => {
                    // Convert Vec<splitters::tree_sitter::Chunk> to Vec<String>
                    chunks = ts_chunks
                      .into_iter()
                      .map(|c| c.text.to_string())
                      .collect();
                    info!("[INDEX_TASK] Split into {} chunks using TreeSitterCodeSplitter.", chunks.len());
                  }
                  Err(e) => {
                    error!("[INDEX_TASK] TreeSitterCodeSplitter failed for {}: {:?}. Falling back to text splitter.", uri_string, e);
                    // Fallback

                    let splitter = TextSplitter::new(
                      // Pass ChunkConfig to new()
                      ChunkConfig::new(ts_config.chunk_size) // Create config with size
                        .with_trim(true), // Configure trimming on the config
                    );
                    // --- End Correction ---
                    chunks = splitter.chunks(&text).map(String::from).collect();
                  }
                }
              }
              Err(e) => {
                error!("[INDEX_TASK] Failed to create TreeSitterCodeSplitter: {:?}. Falling back to text splitter.", e);
                // Fallback

                let splitter = TextSplitter::new(
                  // Pass ChunkConfig to new()
                  ChunkConfig::new(ts_config.chunk_size) // Create config with size
                    .with_trim(true), // Configure trimming on the config
                );
                // --- End Correction ---
                chunks = splitter.chunks(&text).map(String::from).collect();
              }
            }
          } else {
            warn!("[INDEX_TASK] Tree-sitter parsing returned None for {}. Falling back to text splitter.", uri_string);
            // Fallback

            let splitter = TextSplitter::new(
              // Pass ChunkConfig to new()
              ChunkConfig::new(ts_config.chunk_size) // Create config with size
                .with_trim(true), // Configure trimming on the config
            );
            // --- End Correction ---
            chunks = splitter.chunks(&text).map(String::from).collect();
          }
        } else {
          warn!("[INDEX_TASK] Tree-sitter language not found for URI {}. Falling back to text splitter.", uri_string);
          // Fallback

          let splitter = TextSplitter::new(
            // Pass ChunkConfig to new()
            ChunkConfig::new(ts_config.chunk_size) // Create config with size
              .with_trim(true), // Configure trimming on the config
          );
          // --- End Correction ---
          chunks = splitter.chunks(&text).map(String::from).collect();
        }
        // --- End TreeSitter Logic ---
      }
    }

    info!(
      "[INDEX_TASK] Total chunks generated for {}: {}",
      uri_string,
      chunks.len()
    );

    // 3. Iterate and insert/embed each chunk
    for (i, chunk) in chunks.iter().enumerate() {
      // Iterate over Vec<String>
      if chunk.trim().is_empty() {
        // Double check chunk isn't just whitespace
        warn!(
          "[INDEX_TASK] Skipping empty/whitespace chunk {} for {}",
          i, uri_string
        );
        continue;
      }
      let insert_query = "CREATE type::table($tb) SET text = $text, uri = $uri, embedding = vector::embed($model, $text);";
      let vars: HashMap<String, sql::Value> = HashMap::from([
        ("tb".into(), sql::Value::from(config.table_name.clone())),
        ("text".into(), sql::Value::from(chunk.clone())), // Pass the String chunk
        ("uri".into(), sql::Value::from(uri_string.clone())),
        (
          "model".into(),
          sql::Value::from(config.embedding_model_name.clone()),
        ),
      ]);
      match db.query(insert_query).bind(vars).await {
        Ok(_) => info!("[INDEX_TASK] Inserted chunk {} for {}", i, uri_string),
        Err(e) => error!(
          "[INDEX_TASK] Failed to insert chunk {} for {}: {}",
          i, uri_string, e
        ),
      }
    }
    info!("[INDEX_TASK] Finished indexing for {}", uri_string);
    Ok(())
  }
}

// Simple struct to deserialize vector search results
#[derive(Deserialize, Debug)]
struct ChunkResult {
  text: String,
  // score: f32,
}

// --- Chunking Helper (Placeholder - Needs proper implementation) ---
// TODO: Replace with actual chunking logic, potentially using config.splitter
fn chunk_text(text: &str, _config: &SurrealDbConfig) -> Vec<String> {
  info!("[CHUNK] Chunking text (length {})...", text.len());
  // Very basic placeholder: split by double newline, keep non-empty parts
  let chunks: Vec<String> = text
    .split("\n\n")
    .map(|s| s.trim())
    .filter(|s| !s.is_empty())
    .map(String::from)
    .collect();
  info!("[CHUNK] Generated {} chunks.", chunks.len());
  if chunks.is_empty() && !text.trim().is_empty() {
    warn!("[CHUNK] No chunks generated, using full text as one chunk.");
    return vec![text.to_string()]; // Fallback for non-empty text with no double newlines
  }
  chunks
}
// --- End Chunking Helper ---

#[async_trait::async_trait]
impl MemoryBackend for SurrealDbStore {
  // --- Indexing Methods (Stubs for now) ---

  fn opened_text_document(
    &self,
    params: DidOpenTextDocumentParams,
  ) -> Result<()> {
    info!(
      "[SurrealDbStore] opened_text_document received for URI: {:#?}",
      params.text_document.uri
    );
    // Clone necessary data for the async task
    let db_clone = self.db.clone();
    let config_clone = self.config.clone();
    let uri_string = params.text_document.uri.to_string();
    let file_content = params.text_document.text;

    // Spawn background task to handle indexing
    info!("[SurrealDbStore] Spawning indexing task for opened document.");
    TOKIO_RUNTIME.spawn(async move {
      if let Err(e) = Self::index_document(
        db_clone,
        config_clone,
        uri_string.clone(),
        file_content,
      )
      .await
      {
        error!(
          "[SurrealDbStore Task] Error indexing opened document {}: {:?}",
          uri_string, e
        );
      }
    });

    Ok(()) // Return immediately
  }

  fn changed_text_document(
    &self,
    params: DidChangeTextDocumentParams,
  ) -> Result<()> {
    info!(
      "[SurrealDbStore] changed_text_document received for URI: {:#?}",
      params.text_document.uri
    );
    // PROBLEM: params.content_changes doesn't contain the full text.
    // For now, we cannot reliably re-index without the full text.
    warn!("[SurrealDbStore] changed_text_document: Full re-indexing on change requires getting the complete updated file content, which is not directly available here. Indexing skipped.");
    // TODO: Implement a mechanism to get the full updated text content
    //       Maybe the memory_worker needs to maintain the text buffer?
    //       Once full text is available ("new_full_text"):
    //       let db_clone = self.db.clone();
    //       let config_clone = self.config.clone();
    //       let uri_string = params.text_document.uri.to_string();
    //       TOKIO_RUNTIME.spawn(async move {
    //           if let Err(e) = Self::index_document(db_clone, config_clone, uri_string.clone(), new_full_text).await {
    //               error!("[SurrealDbStore Task] Error re-indexing changed document {}: {:?}", uri_string, e);
    //           }
    //       });
    Ok(())
  }

  fn renamed_files(&self, params: RenameFilesParams) -> Result<()> {
    info!("[SurrealDbStore] renamed_files received.");
    let db_clone = self.db.clone();
    let config_clone = self.config.clone();
    let files_to_rename: Vec<FileRename> = params.files; // Clone or take ownership

    // Spawn background task
    info!(
      "[SurrealDbStore] Spawning rename task for {:#?} files.",
      files_to_rename.len()
    );
    TOKIO_RUNTIME.spawn(async move {
      for rename in files_to_rename {
        info!(
          "[SurrealDbStore Task] Renaming URI from {} to {}",
          rename.old_uri, rename.new_uri
        );

        // Update the URI field for all chunks matching the old URI
        let update_query =
          "UPDATE type::table($tb) SET uri = $new WHERE uri = $old;";
        let vars: HashMap<String, sql::Value> = HashMap::from([
          (
            "tb".into(),
            sql::Value::from(config_clone.table_name.clone()),
          ),
          ("new".into(), sql::Value::from(rename.new_uri)), // Use new_uri directly if it's String/&str
          ("old".into(), sql::Value::from(rename.old_uri.clone())), // Use old_uri directly
        ]);

        match db_clone.query(update_query).bind(vars).await {
          Ok(_) => info!(
            "[SurrealDbStore Task] Successfully updated URI for {:#?}",
            rename.old_uri
          ),
          Err(e) => error!(
            "[SurrealDbStore Task] Failed to update URI for {}: {}",
            rename.old_uri, e
          ),
        }
      }
      info!("[SurrealDbStore Task] Finished rename operations.");
    });

    Ok(()) // Return immediately
  }

  // --- Other required methods (provide basic impl) ---

  fn code_action_request(
    &self,
    _text_document_identifier: &TextDocumentIdentifier,
    _range: &Range,
    _trigger: &str,
  ) -> Result<bool> {
    // For now, allow actions - refine later if needed
    Ok(true)
  }

  fn file_request(
    &self,
    _text_document_identifier: &TextDocumentIdentifier,
  ) -> Result<String> {
    // This is tricky - the memory backend is usually expected to hold file content.
    // Fetching from DB seems wrong. This might need a rethink or coordination
    // with how the memory_worker uses this function. For now, return empty or error.
    warn!(
      "[SurrealDbStore] file_request: Not implemented, returning empty string."
    );
    // Alternatives: Fetch from DB (inefficient?), error out, coordinate with worker.
    Ok(String::new()) // Placeholder
  }

  fn get_filter_text(
    &self,
    _position: &TextDocumentPositionParams,
  ) -> Result<String> {
    // Usually gets text around cursor for completion filtering. Not relevant for vector context.
    Ok(String::new()) // Return empty
  }

  // --- Context Retrieval Method ---

  async fn build_prompt(
    &self,
    position: &TextDocumentPositionParams,
    prompt_type: PromptType,
    params: &Value, // Config params passed down
  ) -> Result<Prompt> {
    info!(
      "[SurrealDbStore] build_prompt entered for URI: {:#?}",
      position.text_document.uri
    );

    // --- Step 1: Get the full file content (needed for the 'code' part of prompt) ---
    // We might need to request this from the memory worker again, or maybe store it?
    // Let's assume we fetch it again for simplicity, though inefficient.
    // This requires adding a blocking channel or making build_prompt sync...
    // Alternative: Maybe the 'code' field isn't essential if context is good?
    // Let's return empty code for now to avoid complexity.
    let code_content = String::new(); // Placeholder - needs actual file content
    warn!("[SurrealDbStore] build_prompt: Returning empty 'code' field.");

    // --- Step 2: Determine the query text for vector search ---
    // Simplification: Use a hardcoded query or extract from params?
    // Extracting from last user message requires access to history here, which we don't have easily.
    // Let's try extracting a potential query from the static params if available.
    let query_text = params
      .get("query")
      .and_then(Value::as_str)
      .unwrap_or("")
      .to_string(); // Example: Use a "query" field in params if provided
    if query_text.is_empty() {
      warn!("[SurrealDbStore] build_prompt: No query text found in params, using fixed query for vector search.");
      // query_text = "provide relevant context".to_string(); // Fixed fallback
    } else {
      info!(
        "[SurrealDbStore] build_prompt: Using query text from params: '{}'",
        query_text
      );
    }

    // --- Step 3: Perform Vector Search (only if query is not empty) ---
    let mut context_chunks: Vec<String> = Vec::new();
    if !query_text.is_empty() {
      let query = "SELECT text FROM type::table($tb) WHERE vector <|similar|> vector::embed($model, $query) ORDER BY score DESC LIMIT 5;";
      let vars: HashMap<String, sql::Value> = HashMap::from([
        ("tb".into(), self.config.table_name.clone().into()), // Use configured table name
        (
          "model".into(),
          self.config.embedding_model_name.clone().into(),
        ), // Use configured model name
        ("query".into(), query_text.into()),
      ]);

      info!("[SurrealDbStore] build_prompt: Executing vector search query.");
      let mut response = self
        .db
        .query(query)
        .bind(vars)
        .await
        .context("Vector search query failed")?;
      info!("[SurrealDbStore] build_prompt: Vector search query executed.");

      // Take the first result (index 0) which contains the array of chunks
      let chunks: Vec<ChunkResult> = response
        .take(0)
        .context("Deserializing vector search results failed")?;
      info!(
        "[SurrealDbStore] build_prompt: Found {} relevant chunks.",
        chunks.len()
      );

      context_chunks = chunks.into_iter().map(|c| c.text).collect();
    } else {
      warn!("[SurrealDbStore] build_prompt: Skipping vector search due to empty query text.");
    }

    // --- Step 4: Format Context and Return Prompt ---
    let context_string = context_chunks.join("\n\n---\n\n"); // Join chunks with separator
    info!(
      "[SurrealDbStore] build_prompt: Final context string length: {}",
      context_string.len()
    );

    // For now, assume chat always uses ContextAndCode type
    Ok(Prompt::ContextAndCode(ContextAndCodePrompt {
      context: context_string,
      code: code_content, // Use the (currently empty) file content
      selected_text: None, // TODO: Handle selected text if needed
    }))
  }
}
