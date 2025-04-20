use std::collections::HashMap;

use anyhow::{Context, Result};
use async_trait::async_trait;

use lsp_types::{
  DidChangeTextDocumentParams, DidOpenTextDocumentParams, Range,
  RenameFilesParams, TextDocumentIdentifier, TextDocumentPositionParams, Uri,
};

use serde::Deserialize;
use serde_json::Value;
// For deserializing query results
use surrealdb::engine::remote::ws::{Client, Ws}; // Or other client type based on endpoint
use surrealdb::opt::auth::Root; // Or other auth strategy
use surrealdb::sql;
use surrealdb::Surreal;
use tokio::sync::oneshot;
use tracing::{error, info, warn};

use super::{
  ContextAndCodePrompt, FIMPrompt, MemoryBackend, Prompt, PromptType,
};

use crate::config::{Config, SurrealDbConfig};

// Simple struct to deserialize vector search results
#[derive(Deserialize, Debug)]
struct ChunkResult {
  text: String,
  // score: f32,
}

#[derive(Debug, Clone)]
pub(crate) struct SurrealDbStore {
  db: Surreal<Client>,
  config: SurrealDbConfig,
}

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
      db.signin(Root {
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

    Ok(Self { db, config })
  }
}

#[async_trait::async_trait]
impl MemoryBackend for SurrealDbStore {
  // --- Indexing Methods (Stubs for now) ---

  fn opened_text_document(
    &self,
    params: DidOpenTextDocumentParams,
  ) -> Result<()> {
    info!("[SurrealDbStore] opened_text_document: Indexing not implemented yet. URI: {:#?}", params.text_document.uri);
    // TODO: Implement chunking, embedding via vector::embed, and INSERT/UPDATE
    Ok(())
  }

  fn changed_text_document(
    &self,
    params: DidChangeTextDocumentParams,
  ) -> Result<()> {
    info!("[SurrealDbStore] changed_text_document: Indexing not implemented yet. URI: {:#?}", params.text_document.uri);
    // TODO: Implement efficient update/re-indexing
    Ok(())
  }

  fn renamed_files(&self, params: RenameFilesParams) -> anyhow::Result<()> {
    info!(
      "[SurrealDbStore] renamed_files: Indexing update not implemented yet."
    );
    // TODO: Implement updating URIs in the database
    Ok(())
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
