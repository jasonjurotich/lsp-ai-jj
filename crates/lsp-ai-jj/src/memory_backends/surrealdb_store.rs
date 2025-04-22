use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;

use lsp_types::{
  CodeAction, DidChangeTextDocumentParams, DidOpenTextDocumentParams,
  FileRename, Position, Range, RenameFilesParams, TextDocumentIdentifier,
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
use utils_tree_sitter;

use ropey::Rope;

use super::{
  ContextAndCodePrompt, FIMPrompt, MemoryBackend, Prompt, PromptType,
};
use splitter_tree_sitter::{Chunk as TreeSitterChunk, TreeSitterCodeSplitter};
use std::path::Path;
use text_splitter::{ChunkConfig, TextSplitter};
use tree_sitter::{Parser, Tree};

use crate::config::{
  Config, SurrealDbConfig, TextSplitter as TextSplitterConfig,
  TreeSitter as TreeSitterConfig, ValidSplitter,
};

fn lsp_position_to_char_index(pos: &Position, rope: &Rope) -> Result<usize> {
  let line_idx = pos.line as usize;
  if line_idx > rope.len_lines().saturating_sub(1) {
    // Check line bounds
    return Err(anyhow!(
      "LSP position line {} out of bounds ({} lines)",
      line_idx,
      rope.len_lines()
    ));
  }
  let line = rope.line(line_idx);
  // LSP character is UTF-16 code units, Ropey uses char indices.
  // Convert UTF-16 offset to char offset for the specific line.
  let char_idx = line
    .try_utf16_cu_to_char(pos.character as usize)
    .with_context(|| {
      format!(
        "LSP position character {} out of bounds for line {}",
        pos.character, line_idx
      )
    })?;
  // Get char index of the start of the line and add the char offset within the line
  Ok(rope.line_to_char(line_idx) + char_idx)
}

#[derive(Debug, Clone)]
pub(crate) struct SurrealDbStore {
  db: Surreal<Client>,
  config: SurrealDbConfig,
  documents: Arc<Mutex<HashMap<Uri, Rope>>>,
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

    Ok(Self {
      db,
      config,
      documents: Arc::new(Mutex::new(HashMap::new())),
    })
  }

  async fn index_document(
    db: Surreal<Client>,
    config: SurrealDbConfig,
    uri: Uri, // Expects Uri
    text: String,
  ) -> Result<()> {
    let uri_string = uri.to_string(); // Convert for logging/DB where needed
    info!("[INDEX_TASK] Starting indexing for {}", uri_string);

    // 1. Delete existing chunks
    let delete_query = "DELETE type::table($tb) WHERE uri = $uri;";
    let vars_del: HashMap<String, sql::Value> = HashMap::from([
      ("tb".into(), sql::Value::from(config.table_name.clone())),
      ("uri".into(), sql::Value::from(uri_string.clone())), // Bind String
    ]);
    db.query(delete_query)
      .bind(vars_del)
      .await
      .with_context(|| {
        format!("Failed to delete old chunks for {}", uri_string)
      })?;
    info!("[INDEX_TASK] Deleted old chunks for {}", uri_string);

    // --- 2. Chunk the new text based on configured splitter ---
    let chunks: Vec<String>; // Declare chunks variable

    match &config.splitter {
      // --- TextSplitter Arm ---
      ValidSplitter::TextSplitter(text_splitter_config) => {
        info!(
          "[INDEX_TASK] Using TextSplitter for {} with chunk size: {}",
          uri_string, text_splitter_config.chunk_size
        );
        // Use the corrected TextSplitter initialization
        let splitter = TextSplitter::new(
          ChunkConfig::new(text_splitter_config.chunk_size).with_trim(true),
        );
        chunks = splitter.chunks(&text).map(String::from).collect();
      }
      // --- TreeSitter Arm (Corrected) ---
      ValidSplitter::TreeSitter(tree_sitter_config) => {
        info!( "[INDEX_TASK] Attempting TreeSitter chunking for {} (size: {}, overlap: {})...", uri_string, tree_sitter_config.chunk_size, tree_sitter_config.chunk_overlap);

        let mut tree_sitter_success = false;
        let mut resulting_chunks: Vec<String> = Vec::new();

        // --- CORRECTED: Get file extension from lsp_types::Uri ---

        let extension = if uri.scheme().map(|s| s.as_str()) == Some("file") {
          // 1. Get the path object (type fluent_uri::Path)
          let path_obj = uri.path();
          // 2. Convert the path object to a string slice (&str) using as_str()
          let path_str = path_obj.as_str();
          // 3. Create std::path::Path using the string slice
          std::path::Path::new(path_str)
            .extension() // Now called on a type that AsRef<OsStr>
            .and_then(|os_str| os_str.to_str()) // Convert OsStr to &str
        } else {
          // If not a file scheme, we can't get a filesystem extension
          warn!("[INDEX_TASK] URI scheme is not 'file', cannot determine language for tree-sitter: {}", uri_string);
          None
        };
        // --- End Correction ---

        if let Some(ext) = extension {
          // Proceed only if we got an extension string
          info!("[INDEX_TASK] Found extension '{}'. Getting parser...", ext);
          match utils_tree_sitter::get_parser_for_extension(ext) {
            // Use the extracted &str
            Ok(mut parser) => {
              info!(
                "[INDEX_TASK] Got parser for extension '{}'. Parsing text...",
                ext
              );
              if let Some(tree) = parser.parse(&text, None) {
                info!("[INDEX_TASK] Successfully parsed with tree-sitter.");
                match TreeSitterCodeSplitter::new(
                  tree_sitter_config.chunk_size,
                  tree_sitter_config.chunk_overlap,
                ) {
                  Ok(splitter) => {
                    match splitter.split(&tree, text.as_bytes()) {
                      Ok(ts_chunks) => {
                        resulting_chunks = ts_chunks
                          .into_iter()
                          .map(|c| c.text.to_string())
                          .collect();
                        info!("[INDEX_TASK] Split into {} chunks using TreeSitterCodeSplitter.", resulting_chunks.len());
                        tree_sitter_success = true;
                      }
                      Err(e) => {
                        error!("[INDEX_TASK] TreeSitter split failed: {:?}. Will fallback.", e);
                      }
                    }
                  }
                  Err(e) => {
                    error!("[INDEX_TASK] Failed creating TreeSitterCodeSplitter: {:?}. Will fallback.", e);
                  }
                }
              } else {
                warn!("[INDEX_TASK] Tree-sitter parsing returned None. Will fallback.");
              }
            }
            Err(e) => {
              warn!("[INDEX_TASK] Failed getting parser for extension '{}': {:?}. Will fallback.", ext, e);
            }
          }
        } else {
          // This case handles non-file URIs or paths without extensions
          warn!("[INDEX_TASK] Could not determine suitable extension for URI {}. Will fallback.", uri_string);
        }

        // Assign chunks based on success flag or fallback
        if tree_sitter_success {
          chunks = resulting_chunks;
        } else {
          warn!(
            "[INDEX_TASK] Falling back to text splitter for {}.",
            uri_string
          );
          let splitter = TextSplitter::new(
            ChunkConfig::new(tree_sitter_config.chunk_size).with_trim(true),
          );
          chunks = splitter.chunks(&text).map(String::from).collect();
        }
      } // End TreeSitter arm
    } // End match config.splitter

    info!(
      "[INDEX_TASK] Total chunks generated for {}: {}",
      uri_string,
      chunks.len()
    );

    // --- 3. Iterate and insert/embed each chunk ---
    for (i, chunk) in chunks.iter().enumerate() {
      if chunk.trim().is_empty() {
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
        ("uri".into(), sql::Value::from(uri_string.clone())), // Bind String URI
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
  } // End of index_document function
}

// Simple struct to deserialize vector search results
#[derive(Deserialize, Debug)]
struct ChunkResult {
  text: String,
  // score: f32,
}

#[async_trait::async_trait]
impl MemoryBackend for SurrealDbStore {
  // --- Indexing Methods (Stubs for now) ---

  fn opened_text_document(
    &self,
    params: DidOpenTextDocumentParams,
  ) -> Result<()> {
    let uri = params.text_document.uri;
    let file_content_string = params.text_document.text; // Initial content is String
    info!(
      "[SurrealDbStore] opened_text_document received for URI: {:#?}",
      uri
    );

    // --- Store initial content as Rope ---
    let rope = Rope::from_str(&file_content_string);
    match self.documents.lock() {
      Ok(mut docs) => {
        info!(
          "[SurrealDbStore] Storing initial Rope ({} chars) for {:#?}",
          rope.len_chars(),
          uri
        );
        docs.insert(uri.clone(), rope); // Store the Rope
      }
      Err(poison_error) => {
        error!(
          "Failed to lock documents map in opened_text_document: {}",
          poison_error
        );
        return Err(anyhow!("Mutex poisoned in opened_text_document"));
      }
    }
    // --- End store ---

    // --- Spawn Indexing Task (pass initial String) ---
    let db_clone = self.db.clone();
    let config_clone = self.config.clone();
    info!("[SurrealDbStore] Spawning indexing task for opened document.");
    TOKIO_RUNTIME.spawn(async move {
      // Pass the Uri and the initial full content String
      if let Err(e) = Self::index_document(
        db_clone,
        config_clone,
        uri.clone(),
        file_content_string,
      )
      .await
      {
        error!(
          "[SurrealDbStore Task] Error indexing opened document {:#?}: {:?}",
          uri, e
        );
      }
    });

    Ok(())
  }

  fn changed_text_document(
    &self,
    params: DidChangeTextDocumentParams,
  ) -> Result<()> {
    let uri = params.text_document.uri; // uri is type Uri
                                        // Use Debug format {:?} for logging Uri
    info!("[SurrealDbStore] changed_text_document received for URI: {:?} ({} changes)", uri, params.content_changes.len());

    // --- Step 1: Acquire the lock ---
    let mut docs_guard = match self.documents.lock() {
      Ok(guard) => guard,
      Err(poison_error) => {
        error!("Failed to lock documents map: {}", poison_error);
        return Err(anyhow!("Mutex poisoned in changed_text_document"));
      }
    };

    // --- Step 2: Get the mutable Rope ---
    let rope = match docs_guard.get_mut(&uri) {
      Some(r) => r,
      None => {
        // Use Debug format {:?} for logging Uri
        error!(
          "Received change for untracked document: {:?}. Skipping.",
          uri
        );
        return Ok(());
      }
    };

    // --- Step 3: Apply changes ---
    // Use Debug format {:?} for logging Uri
    info!(
      "[SurrealDbStore] Applying {} changes to stored Rope for {:?}",
      params.content_changes.len(),
      uri
    );
    let mut received_full_text = false;
    for change in params.content_changes {
      if let Some(range) = change.range {
        // Incremental Change
        // Match on the tuple of results from the conversion functions
        match (
          lsp_position_to_char_index(&range.start, rope),
          lsp_position_to_char_index(&range.end, rope),
        ) {
          // Case 1: Both start and end positions converted successfully
          (Ok(start_char), Ok(end_char)) => {
            info!(
              "[SurrealDbStore] Applying change: range {:?} -> chars {}..{}",
              range, start_char, end_char
            ); // Log successful conversion
               // Bounds checking and applying the edit (as before)
            if start_char > end_char || start_char > rope.len_chars() {
              error!("Invalid change range indices: start={}, end={}. Rope len={}. Skipping change for {:?}", start_char, end_char, rope.len_chars(), uri);
              continue;
            }
            let current_end_char = std::cmp::min(end_char, rope.len_chars()); // Use a different name to avoid shadowing
            if start_char > current_end_char {
              error!("Invalid change range after clamp: start {} > end {}. Skipping change for {:?}", start_char, current_end_char, uri);
              continue;
            }
            rope.remove(start_char..current_end_char);
            rope.insert(start_char, &change.text);
            info!("[SurrealDbStore] Change applied successfully.");
          }
          // Case 2: Start position failed, End position failed (or succeeded - doesn't matter)
          (Err(e), _) => {
            // Use wildcard _ for the second element
            error!("Failed to convert start position {:?} for {:?}: {:?}. Skipping change.", range.start, uri, e);
            continue; // Skip this change
          }
          // Case 3: Start position succeeded, End position failed
          (_, Err(e)) => {
            // Use wildcard _ for the first element
            error!("Failed to convert end position {:?} for {:?}: {:?}. Skipping change.", range.end, uri, e);
            continue; // Skip this change
          } // Note: Cases 2 and 3 cover all possibilities where at least one Err occurred.
            // The pattern `Err(e)` used before was trying to match the whole tuple as a single Err, which is wrong.
        }
      } else {
        // Full Text Change logic (remains the same)
        info!("[SurrealDbStore] Received full text change for {:?}", uri);
        *rope = Rope::from_str(&change.text);
        received_full_text = true;
        break;
      }
    }

    // --- Step 4: Get final text and release lock ---
    let new_full_text = rope.to_string();
    drop(docs_guard);
    // Use Debug format {:?} for logging Uri
    info!(
      "[SurrealDbStore] Rope updated for {:?}. New text length: {}",
      uri,
      new_full_text.len()
    );

    // --- Step 5: Spawn re-indexing task ---

    if params.content_changes.is_empty() && !received_full_text {
      // Use Debug format {:?} for logging Uri
      warn!("[SurrealDbStore] No content changes received for {:?}. Skipping indexing spawn.", uri);
      return Ok(());
    }

    let db_clone = self.db.clone();
    let config_clone = self.config.clone();
    // Use Debug format {:?} for logging Uri
    info!(
      "[SurrealDbStore] Spawning re-indexing task for changed document: {:?}",
      uri
    );
    TOKIO_RUNTIME.spawn(async move {
      if let Err(e) =
        Self::index_document(db_clone, config_clone, uri.clone(), new_full_text)
          .await
      {
        // Keep Debug {:?} here as previously specified
        error!(
          "[SurrealDbStore Task] Error re-indexing changed document {:?}: {:?}",
          uri, e
        );
      }
    });

    // --- Step 6: Return success ---
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
