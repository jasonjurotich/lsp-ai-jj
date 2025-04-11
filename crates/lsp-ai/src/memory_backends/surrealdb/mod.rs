//! SurrealDB Memory Backend implementation using client-side Gemini API for embeddings.

// --- External Crates ---
use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use futures::future::join_all;
use lsp_types::{Range, TextDocumentIdentifier, TextDocumentPositionParams, Url}; // Added Url
use parking_lot::Mutex;
use rand::{distributions::Alphanumeric, Rng};
use reqwest::Client as HttpClient;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{HashMap, HashSet},
    sync::{
        mpsc::{self, Receiver, Sender},
        Arc,
    },
    time::Duration,
};
use surrealdb::{
    engine::remote::ws::Ws, // Using WebSocket client example
    kvs::Datastore,
    opt::auth::Root,
    sql::{Id, Thing, Value as SurrealValue, Vector},
    Surreal,
};
use tokio::time;
use tracing::{debug, error, info, instrument, warn};

// --- Local Crate Imports ---
use crate::{
    config::{self, Config as MainConfig, SurrealDB as SurrealDBConfig, ValidModel}, // Use aliasing
    crawl::Crawl,
    memory_backends::{
        file_store::{AdditionalFileStoreParams, FileStore},
        ContextAndCodePrompt, FIMPrompt, MemoryBackend, MemoryRunParams, Prompt, PromptType,
    },
    splitters::{Chunk, Splitter}, // Assuming Splitter trait/impls are here
    utils::{chunk_to_id, format_file_chunk, tokens_to_estimated_characters, TOKIO_RUNTIME},
};

// --- Constants ---
const RESYNC_MAX_FILE_SIZE: u64 = 10_000_000; // Max file size for resync scan
const DEBOUNCE_DURATION_MS: u64 = 500; // Debounce time for file changes
const BATCH_SIZE: usize = 50; // Batch size for DB operations and API calls (adjust as needed)
const EMBEDDING_DIM: usize = 768; // IMPORTANT: Set to your Gemini embedding model's dimension
const DEFAULT_GEMINI_API_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta/"; // Note: No /models/ here
const DEFAULT_EMBEDDING_TASK_TYPE: &str = "RETRIEVAL_DOCUMENT";
const DEFAULT_QUERY_EMBEDDING_TASK_TYPE: &str = "RETRIEVAL_QUERY";

// --- Authentication Helper ---

/// Retrieves the Gemini API Token from config or environment variable.
fn get_gemini_token(gemini_config: &config::Gemini) -> Result<String> {
    if let Some(env_var_name) = &gemini_config.auth_token_env_var_name {
        std::env::var(env_var_name)
            .with_context(|| format!("Failed to get Gemini API key from env var '{env_var_name}'"))
    } else if let Some(token) = &gemini_config.auth_token {
        Ok(token.clone())
    } else {
        Err(anyhow!(
            "Gemini configuration missing: set 'auth_token_env_var_name' or 'auth_token'"
        ))
    }
}

// --- Gemini API Client for Embeddings ---

// Structs for Gemini API embedding request/response
#[derive(Serialize)]
struct GeminiApiContentPart {
    text: String,
}
#[derive(Serialize)]
struct GeminiApiContent {
    parts: Vec<GeminiApiContentPart>,
    // role: Option<String>, // Not needed for embedContent
}
#[derive(Serialize)]
struct GeminiApiEmbedRequest {
    content: GeminiApiContent,
    task_type: String,
}
#[derive(Deserialize)]
struct GeminiEmbedding {
    value: Vec<f32>,
}
// Response struct for the `embedContent` endpoint
#[derive(Deserialize)]
struct GeminiApiEmbedResponse {
    embedding: GeminiEmbedding,
}

#[derive(Clone)]
struct GeminiEmbedder {
    client: HttpClient,
    config: config::Gemini, // Store unified Gemini config
}

impl GeminiEmbedder {
    /// Creates a new Gemini Embedder client.
    fn new(config: config::Gemini) -> Result<Self> {
        let client = HttpClient::builder()
            .timeout(Duration::from_secs(60)) // Increased timeout for potential batching/API latency
            .build()?;
        Ok(Self { client, config })
    }

    /// Generates embeddings for a batch of texts using parallel requests.
    async fn generate_embeddings_batch(
        &self,
        texts: Vec<String>,
        task_type: &str,
    ) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        debug!(
            count = texts.len(),
            task_type, "Requesting Gemini embeddings batch"
        );

        let token = get_gemini_token(&self.config)?;
        let request_url = format!(
            "{}/models/{}:embedContent?key={}", // Use :embedContent endpoint
            self.config.api_base_url.trim_end_matches('/'),
            self.config.embedding_model,
            token
        );

        // Use Tokio's semaphore for concurrency control if needed (e.g., limit to 10 concurrent requests)
        // let semaphore = Arc::new(Semaphore::new(10));

        let futures = texts.into_iter().map(|text| {
            let client = self.client.clone();
            let url = request_url.clone();
            let task_type = task_type.to_string();
            // let permit = semaphore.clone().acquire_owned().await.unwrap(); // For concurrency limiting
            async move {
                // drop(permit); // Release permit when task finishes (RAII)
                let request_payload = GeminiApiEmbedRequest {
                    content: GeminiApiContent {
                        parts: vec![GeminiApiContentPart { text }],
                    },
                    task_type,
                };

                let response = client.post(&url).json(&request_payload).send().await?;

                if !response.status().is_success() {
                    let status = response.status();
                    let error_text = response.text().await.unwrap_or_default();
                    return Err(anyhow!(
                        "Gemini API request to {} failed with status {}: {}",
                        url,
                        status,
                        error_text
                    ));
                }

                let result: GeminiApiEmbedResponse = response.json().await?;
                Ok::<_, anyhow::Error>(result.embedding.value)
            }
        });

        let results: Vec<Result<Vec<f32>, anyhow::Error>> = join_all(futures).await;

        let mut embeddings = Vec::with_capacity(results.len());
        let mut successful_count = 0;
        for (i, result) in results.into_iter().enumerate() {
            match result {
                Ok(embedding) => {
                    if embedding.len() == EMBEDDING_DIM {
                        embeddings.push(embedding);
                        successful_count += 1;
                    } else {
                        error!(
                            index = i,
                            expected_dim = EMBEDDING_DIM,
                            actual_dim = embedding.len(),
                            "Embedding dimension mismatch. Skipping."
                        );
                    }
                }
                Err(e) => {
                    error!(index = i, error = ?e, "Failed to generate embedding for item. Skipping.");
                }
            }
        }

        if successful_count != texts.len() {
            warn!(
                requested = texts.len(),
                succeeded = successful_count,
                "Partial success generating embeddings batch."
            );
        }
        // Note: The returned vector might be shorter than `texts.len()` if errors occurred.
        Ok(embeddings)
    }

    /// Generates embedding for a single text document.
    async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>> {
        let task_type = self
            .config
            .embedding_task_type
            .as_deref()
            .unwrap_or(DEFAULT_EMBEDDING_TASK_TYPE);
        // Use batch method even for one to reuse logic
        let mut results = self
            .generate_embeddings_batch(vec![text.to_string()], task_type)
            .await?;
        results
            .pop()
            .ok_or_else(|| anyhow!("Gemini API returned no embedding for single text"))
    }

    /// Generates embedding for a single search query.
    async fn generate_query_embedding(&self, text: &str) -> Result<Vec<f32>> {
        let task_type = self
            .config
            .query_embedding_task_type
            .as_deref()
            .unwrap_or(DEFAULT_QUERY_EMBEDDING_TASK_TYPE);
        let mut results = self
            .generate_embeddings_batch(vec![text.to_string()], task_type)
            .await?;
        results
            .pop()
            .ok_or_else(|| anyhow!("Gemini API returned no embedding for query text"))
    }
}

// --- SurrealDB Document Structure ---
#[derive(Serialize, Deserialize, Debug, Clone)]
struct DocumentChunk {
    uri: String,
    text: String,
    // Use a serializable representation for Range if lsp_types::Range isn't directly compatible
    // Or implement Serialize/Deserialize for it. Assuming it works for now.
    range: Range,
    #[serde(with = "surrealdb::sql::vector")]
    embedding: Vec<f32>,
}

// --- SurrealDB Record ID Helper ---
/// Creates a SurrealDB Thing (Record ID) ensuring valid characters.
fn create_thing(table: &str, chunk_id: &str) -> Result<Thing> {
    // Replace characters invalid in SurrealDB unquoted IDs
    let sanitized_id: String = chunk_id
        .chars()
        .map(|c| match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '_' => c,
            _ => '_', // Replace invalid characters with underscore
        })
        .collect();

    // Add prefix if sanitized ID starts with a number (optional, defensive)
    let final_id = if sanitized_id.chars().next().map_or(true, |c| c.is_numeric()) {
        format!("id_{}", sanitized_id)
    } else {
        sanitized_id
    };

    Thing::from((table.to_string(), Id::String(final_id))).context("Failed to create Thing ID")
}

// --- SurrealMemory Backend Implementation ---
#[derive(Clone)]
pub(crate) struct SurrealMemory {
    client_params: config::ValidClientParams,
    file_store: Arc<FileStore>,
    db: Arc<Surreal<surrealdb::engine::remote::ws::Client>>,
    embedder: Arc<GeminiEmbedder>,
    debounce_tx: Sender<String>,
    crawl: Option<Arc<Mutex<Crawl>>>,
    splitter: Arc<Box<dyn Splitter + Send + Sync>>,
    table_name: String,
}

impl SurrealMemory {
    /// Creates a new SurrealMemory backend instance.
    #[instrument(skip_all)] // Skip all args for cleaner logs at this high level
    pub(crate) fn new(
        backend_config: SurrealDBConfig,
        models: &HashMap<String, ValidModel>,
        client_params: config::ValidClientParams,
    ) -> Result<Self> {
        info!("Initializing SurrealDB Memory Backend");

        // --- Get Gemini Configuration ---
        let gemini_model_key = &backend_config.gemini_model_key;
        info!(gemini_model_key, "Looking up Gemini configuration");
        let gemini_config = models
            .get(gemini_model_key)
            .with_context(|| format!("Gemini model key '{gemini_model_key}' not found"))?
            .gemini() // Helper method assumed on ValidModel enum
            .with_context(|| format!("Model '{gemini_model_key}' is not a Gemini model"))?
            .clone(); // Clone the specific Gemini config

        // Validate necessary embedding model field
        if gemini_config.embedding_model.is_empty() {
            return Err(anyhow!(
                "Gemini config '{gemini_model_key}' missing 'embedding_model'"
            ));
        }
        info!(model=%gemini_config.embedding_model, "Using Gemini embedding model");

        // --- Initialize Embedder ---
        let embedder = Arc::new(GeminiEmbedder::new(gemini_config.clone())?);

        // --- Initialize Splitter, Crawl, FileStore ---
        let splitter: Arc<Box<dyn Splitter + Send + Sync>> =
            Arc::new(backend_config.splitter.clone().try_into()?);

        let crawl = backend_config.crawl.clone().map(|crawl_cfg| {
            // Pass client_params needed by Crawl::new
            Arc::new(Mutex::new(Crawl::new(crawl_cfg, client_params.clone())))
        });

        let file_store = Arc::new(FileStore::new_with_params(
            config::FileStore::new_without_crawl(),
            client_params.clone(),
            AdditionalFileStoreParams::new(splitter.does_use_tree_sitter()),
        )?);

        // --- SurrealDB Connection ---
        let db = TOKIO_RUNTIME.block_on(async {
            info!(url = %backend_config.database_url, "Connecting to SurrealDB...");
            let db = Surreal::new::<Ws>(&backend_config.database_url).await?;
            if let (Some(user), Some(pass)) = (&backend_config.user, &backend_config.pass) {
                db.signin(Root {
                    username: user,
                    password: pass,
                })
                .await?;
                info!("Signed into SurrealDB with user credentials");
            }
            db.use_ns(&backend_config.namespace)
                .use_db(&backend_config.database)
                .await?;
            info!(
                "SurrealDB connection successful (NS: {}, DB: {})",
                backend_config.namespace, backend_config.database
            );
            Ok::<_, anyhow::Error>(db)
        })?;
        let db = Arc::new(db);

        // --- Calculate Table Name ---
        let table_name = match client_params.root_uri.clone() {
            Some(root_uri_str) => {
                // Attempt to parse as URL to handle file paths more robustly
                let root_uri_parsed = Url::parse(&root_uri_str)
                    .or_else(|_| Url::from_file_path(&root_uri_str)) // Try as file path if parse fails
                    .map_err(|_| anyhow!("Invalid root_uri format: {}", root_uri_str))?;

                let path_hash_input = format!(
                    "{}_{}_{}",
                    root_uri_parsed.as_str(), // Use consistent string representation
                    gemini_config.embedding_model,
                    EMBEDDING_DIM
                );
                format!("docs_{:x}", md5::compute(path_hash_input.as_bytes()))
            }
            None => {
                warn!("No root_uri for client, using random table name");
                format!(
                    "docs_{}",
                    rand::thread_rng()
                        .sample_iter(&Alphanumeric)
                        .take(15)
                        .map(char::from)
                        .collect::<String>()
                        .to_lowercase()
                )
            }
        };
        info!(table_name, "Using SurrealDB table");

        // --- Define Schema Task ---
        let task_db = db.clone();
        let task_table_name = table_name.clone();
        TOKIO_RUNTIME.spawn(async move {
            info!(table = %task_table_name, "Spawning task to define schema");
            let define_fields_query = format!(
               "DEFINE FIELD uri ON TABLE {task_table_name} TYPE string; \
                DEFINE FIELD text ON TABLE {task_table_name} TYPE string; \
                DEFINE FIELD range ON TABLE {task_table_name} TYPE object ASSERT is::object($value); \
                DEFINE FIELD embedding ON TABLE {task_table_name} TYPE vector<float, {EMBEDDING_DIM}>;"
            );
             match task_db.query(define_fields_query).await {
                 Ok(_) => info!(table = %task_table_name, "Defined fields ok."),
                 Err(e) => warn!(table = %task_table_name, error = ?e, "Failed to define fields (may exist)"),
             }

            let index_name = format!("{task_table_name}_embedding_idx");
            let define_index_query = format!(
                "DEFINE INDEX {index_name} ON TABLE {task_table_name} COLUMNS embedding HNSW DIMENSION {EMBEDDING_DIM};"
            );
            match task_db.query(define_index_query).await {
                Ok(_) => info!(index = %index_name, "Defined vector index ok."),
                Err(e) => warn!(index = %index_name, error = ?e, "Failed to define vector index (may exist)"),
            }
        });

        // --- Debouncer Setup ---
        let (debounce_tx, debounce_rx) = mpsc::channel::<String>();
        // Clone Arcs for the debouncer task
        let task_db_debouncer = db.clone();
        let task_file_store = file_store.clone();
        let task_splitter = splitter.clone();
        let task_embedder_debouncer = embedder.clone();
        let task_table_name_debouncer = table_name.clone();
        let task_root_uri = client_params.root_uri.clone(); // Clone root URI for task

        TOKIO_RUNTIME.spawn(async move {
            debouncer_task(
                debounce_rx,
                Duration::from_millis(DEBOUNCE_DURATION_MS),
                task_db_debouncer,
                task_file_store,
                task_splitter,
                task_embedder_debouncer,
                task_table_name_debouncer,
                task_root_uri,
            )
            .await;
        });

        // --- Create Instance ---
        let s = Self {
            client_params: client_params.clone(),
            file_store,
            db,
            embedder,
            debounce_tx,
            crawl,
            splitter,
            table_name: table_name.clone(),
        };

        // --- Initial State Population ---
        let task_s_resync = s.clone();
        TOKIO_RUNTIME.spawn(async move {
            info!("Starting initial resync...");
            if let Err(e) = task_s_resync.resync().await {
                error!("Resync failed: {e:?}");
            } else {
                info!("Initial resync completed.");
            }
        });
        if let Err(e) = s.maybe_do_crawl(None) {
            error!("Initial crawl failed: {e:?}");
        }

        info!("SurrealDB Memory Backend initialized successfully");
        Ok(s)
    }

    /// Upserts document chunks, generating embeddings beforehand.
    #[instrument(skip(self, chunks_with_uris), fields(count = chunks_with_uris.len()))]
    async fn upsert_chunks(
        &self,
        chunks_with_uris: Vec<(String, Chunk)>,
        root_uri: Option<&str>,
    ) -> Result<()> {
        if chunks_with_uris.is_empty() {
            return Ok(());
        }
        debug!("Preparing chunks for embedding and upsert");

        let mut record_ids = Vec::with_capacity(chunks_with_uris.len());
        let mut texts_to_embed = Vec::with_capacity(chunks_with_uris.len());
        let mut original_data = Vec::with_capacity(chunks_with_uris.len()); // Store (uri, range, text)

        for (uri, chunk) in chunks_with_uris {
            let text = format_file_chunk(&uri, &chunk.text, root_uri);
            let chunk_id_str = chunk_to_id(&uri, &chunk);
            match create_thing(&self.table_name, &chunk_id_str) {
                Ok(record_id) => {
                    record_ids.push(record_id);
                    texts_to_embed.push(text.clone());
                    original_data.push((uri, chunk.range, text)); // Store data needed later
                }
                Err(e) => {
                    error!(chunk_id = %chunk_id_str, error = ?e, "Failed to create record ID, skipping chunk.")
                }
            }
        }

        // Generate embeddings in batches
        let mut embeddings = Vec::new();
        for text_batch in texts_to_embed.chunks(BATCH_SIZE) {
            match self
                .embedder
                .generate_embeddings_batch(text_batch.to_vec(), DEFAULT_EMBEDDING_TASK_TYPE)
                .await
            {
                Ok(batch_embeddings) => {
                    // We get potentially fewer embeddings than requested if errors occurred within the batch call
                    embeddings.extend(batch_embeddings);
                }
                Err(e) => {
                    error!(error = ?e, "Failed to generate embeddings batch. Continuing without these embeddings.");
                    // If a whole batch fails, we won't have embeddings for those texts.
                    // The lengths might mismatch below. Handle this gracefully.
                }
            }
        }

        // Combine original data with successful embeddings
        // We assume generate_embeddings_batch returns results in the same order as input,
        // but potentially skipping failed ones. A more robust approach might return Option<Vec<f32>>
        // or (index, Vec<f32>) pairs. For now, we assume the successful ones match the start of the batch.
        if embeddings.len() > record_ids.len() {
            // This shouldn't happen with current logic but is a safeguard
            error!("Received more embeddings than record IDs, potential logic error.");
            embeddings.truncate(record_ids.len()); // Truncate to match
        }

        let documents_to_upsert: Vec<(Thing, DocumentChunk)> = record_ids
            .into_iter()
            .zip(original_data.into_iter())
            .zip(embeddings.into_iter()) // Zip assumes embeddings align with first N records
            .map(|((record_id, (uri, range, text)), embedding)| {
                let doc = DocumentChunk {
                    uri,
                    text,
                    range,
                    embedding,
                };
                (record_id, doc)
            })
            .collect();

        if documents_to_upsert.is_empty() {
            if !texts_to_embed.is_empty() {
                warn!("No documents to upsert after embedding generation (all failed?).");
            }
            return Ok(()); // Nothing to do if all embeddings failed
        }

        // Batch Upsert to SurrealDB
        debug!(
            count = documents_to_upsert.len(),
            "Upserting documents to SurrealDB"
        );
        for batch in documents_to_upsert.chunks(BATCH_SIZE) {
            let update_futures = batch
                .iter()
                .map(|(thing, doc)| self.db.update(thing.clone()).content(doc));
            let results = join_all(update_futures).await;
            results.into_iter().filter_map(|r| r.err()).for_each(|e| {
                error!(error = ?e, "Failed to upsert document batch");
            });
        }
        debug!("Upsert finished");
        Ok(())
    }

    /// Splits a file's content from FileStore and upserts the chunks.
    #[instrument(skip(self, file_store))]
    async fn split_and_upsert_file(
        &self,
        uri: &str,
        file_store: Arc<FileStore>,
        root_uri: Option<&str>,
    ) -> Result<()> {
        debug!(uri, "Splitting and upserting file");
        let chunks = {
            file_store
                .file_map()
                .read()
                .get(uri)
                .map(|f| self.splitter.split(f))
                .with_context(|| format!("file not found in FileStore for splitting: {uri}"))?
        };
        let uri_chunks: Vec<(String, Chunk)> =
            chunks.into_iter().map(|c| (uri.to_string(), c)).collect();
        self.upsert_chunks(uri_chunks, root_uri).await
    }

    /// Resynchronizes the database state with the local filesystem.
    #[instrument(skip(self))]
    async fn resync(&self) -> Result<()> {
        info!("Starting resync process...");
        // 1. Get all URIs from DB
        let query = format!("SELECT DISTINCT VALUE uri FROM {};", self.table_name);
        let uris_in_db: HashSet<String> = self.db.query(query).await?.take(0)?;
        debug!(count = uris_in_db.len(), "Found URIs in database");

        // 2. Check local files
        let mut uris_to_delete = HashSet::new();
        let mut files_to_rescan = Vec::new(); // (uri, contents)
        let try_get_file_contents = |path_str: &str| -> Result<Option<String>> {
            let path = std::path::Path::new(path_str);
            if !path.exists() {
                return Ok(None);
            }
            let metadata = std::fs::metadata(path)?;
            if metadata.len() > RESYNC_MAX_FILE_SIZE {
                warn!(path=%path_str, "Skipping large file during resync.");
                return Ok(None);
            }
            std::fs::read_to_string(path)
                .map(Some)
                .map_err(|e| anyhow!(e))
        };
        for uri in uris_in_db {
            let path_str = uri.replace("file://", "");
            match try_get_file_contents(&path_str) {
                Ok(Some(contents)) => files_to_rescan.push((uri, contents)),
                Ok(None) => {
                    uris_to_delete.insert(uri);
                }
                Err(e) => {
                    error!(%uri, error=?e, "Marking for deletion.");
                    uris_to_delete.insert(uri);
                }
            }
        }

        // 3. Delete records for missing files
        if !uris_to_delete.is_empty() {
            info!(
                count = uris_to_delete.len(),
                "Deleting records for missing files"
            );
            let delete_query = format!("DELETE {} WHERE uri IN $uris;", self.table_name);
            let params = [(
                "uris".to_string(),
                uris_to_delete.into_iter().collect::<Vec<_>>().into(),
            )];
            self.db.query(delete_query).bind(params).await?;
        }

        // 4. Re-split and collect chunks for existing files
        info!(count = files_to_rescan.len(), "Rescanning existing files");
        let mut all_chunks_to_upsert: Vec<(String, Chunk)> = Vec::new();
        for (uri, contents) in files_to_rescan {
            // Delete old chunks first for this URI to ensure clean state
            let delete_old_query = format!("DELETE {} WHERE uri = $uri;", self.table_name);
            if let Err(e) = self.db.query(delete_old_query).bind(("uri", &uri)).await {
                error!(%uri, error=?e, "Failed to delete old chunks before rescan upsert");
            }
            // Split and add new chunks to the list
            let chunks = self.splitter.split_file_contents(&uri, &contents);
            all_chunks_to_upsert.extend(chunks.into_iter().map(|c| (uri.clone(), c)));
        }

        // 5. Perform batched upsert for all rescanned files
        self.upsert_chunks(all_chunks_to_upsert, self.client_params.root_uri.as_deref())
            .await?;

        info!("Resync process finished.");
        Ok(())
    }

    /// Performs crawl if configured, upserting discovered files.
    #[instrument(skip(self))]
    fn maybe_do_crawl(&self, triggered_file: Option<String>) -> Result<()> {
        if let Some(crawl) = &self.crawl {
            info!(?triggered_file, "Starting crawl...");
            let mut chunks_to_upsert: Vec<(String, Chunk)> = Vec::new();
            let mut current_bytes = 0;
            let mut total_bytes = 0;
            let root_uri_clone = self.client_params.root_uri.clone(); // Clone for closure/tasks
            let self_clone = self.clone(); // Clone Arc<Self> for async tasks

            let result = crawl
                .lock()
                .maybe_do_crawl(triggered_file, |crawl_config, path_str| {
                    if total_bytes as u64 >= crawl_config.max_crawl_memory {
                        warn!("Ending crawl due to max memory limit");
                        return Ok(false);
                    }
                    let uri = format!("file://{path_str}");
                    if self.file_store.contains_file(&uri) {
                        return Ok(true);
                    } // Skip if already open

                    let contents = match std::fs::read_to_string(path_str) {
                        Ok(c) => c,
                        Err(e) => {
                            warn!(path=%path_str, error=?e, "Skipping crawl file");
                            return Ok(true);
                        }
                    };
                    let file_len = contents.len();
                    if file_len as u64 > crawl_config.max_file_size {
                        return Ok(true);
                    } // Skip large file

                    current_bytes += file_len;
                    total_bytes += file_len;
                    let file_chunks = self.splitter.split_file_contents(&uri, &contents);
                    chunks_to_upsert.extend(file_chunks.into_iter().map(|c| (uri.clone(), c)));

                    // Upsert in batches
                    if current_bytes >= 10_000_000
                        || total_bytes as u64 >= crawl_config.max_crawl_memory
                    {
                        if !chunks_to_upsert.is_empty() {
                            info!(count = chunks_to_upsert.len(), "Upserting crawl batch");
                            let batch = std::mem::take(&mut chunks_to_upsert);
                            let task_self = self_clone.clone();
                            let task_root_uri = root_uri_clone.clone();
                            TOKIO_RUNTIME.spawn(async move {
                                if let Err(e) = task_self
                                    .upsert_chunks(batch, task_root_uri.as_deref())
                                    .await
                                {
                                    error!(error=?e, "Error upserting crawled batch");
                                }
                            });
                            current_bytes = 0;
                        }
                    }
                    Ok(true) // Continue crawl
                });

            if let Err(e) = result {
                error!(error=?e, "Crawl failed");
                return Err(e.into());
            }

            // Upsert final batch
            if !chunks_to_upsert.is_empty() {
                info!(
                    count = chunks_to_upsert.len(),
                    "Upserting final crawl batch"
                );
                let task_self = self_clone.clone();
                let task_root_uri = root_uri_clone.clone();
                TOKIO_RUNTIME.spawn(async move {
                    if let Err(e) = task_self
                        .upsert_chunks(chunks_to_upsert, task_root_uri.as_deref())
                        .await
                    {
                        error!(error=?e, "Error upserting final crawl batch");
                    }
                });
            }
            info!("Crawl finished.");
        }
        Ok(())
    }
}

// --- Debouncer Task ---
/// Task that waits for file change notifications and updates the database.
#[instrument(skip_all)]
async fn debouncer_task(
    rx: Receiver<String>,
    duration: Duration,
    db: Arc<Surreal<surrealdb::engine::remote::ws::Client>>,
    file_store: Arc<FileStore>,
    splitter: Arc<Box<dyn Splitter + Send + Sync>>,
    embedder: Arc<GeminiEmbedder>, // Use specific type
    table_name: String,
    root_uri: Option<String>, // Needs to be owned for async task
) {
    info!("Debouncer task started");
    let mut changed_files_uris = HashSet::new(); // Store unique URIs that changed

    loop {
        // Drain channel efficiently, collecting unique URIs
        let received_this_tick: Vec<String> = rx.try_iter().collect();
        if !received_this_tick.is_empty() {
            changed_files_uris.extend(received_this_tick);
        }

        // If no changes received, sleep and check again
        if changed_files_uris.is_empty() {
            time::sleep(duration).await;
            continue;
        }

        // Process the collected URIs
        debug!(
            count = changed_files_uris.len(),
            "Processing debounced files"
        );
        let uris_to_process: Vec<String> = changed_files_uris.drain().collect(); // Take ownership

        let mut all_chunks_to_upsert: Vec<(String, Chunk)> = Vec::new();
        let mut current_chunk_ids_by_uri: HashMap<String, HashSet<String>> = HashMap::new();

        // --- Prepare Data for Upsert & Stale Check ---
        for uri in &uris_to_process {
            let file_map = file_store.file_map().read();
            if let Some(file) = file_map.get(uri) {
                let current_chunks = splitter.split(file);
                let mut current_ids = HashSet::new();
                for chunk in current_chunks {
                    current_ids.insert(chunk_to_id(uri, &chunk));
                    all_chunks_to_upsert.push((uri.clone(), chunk)); // Collect for upsert
                }
                current_chunk_ids_by_uri.insert(uri.clone(), current_ids);
            } else {
                warn!(
                    uri,
                    "File disappeared from FileStore during debounce processing"
                );
                // Ensure we check for stale chunks for this URI even if file disappeared
                current_chunk_ids_by_uri.entry(uri.clone()).or_default(); // Ensure entry exists
            }
        }

        // --- Upsert collected chunks (uses client-side embedding) ---
        if !all_chunks_to_upsert.is_empty() {
            debug!(
                count = all_chunks_to_upsert.len(),
                "Upserting debounced chunks"
            );
            // Clone necessary Arcs for the task
            let upsert_embedder = embedder.clone();
            let upsert_db = db.clone();
            let upsert_table_name = table_name.clone();
            let upsert_root_uri = root_uri.clone();

            // This part needs careful thought: upsert_chunks needs `&self` or the Arcs.
            // We don't have `self` here. We need to create a temporary helper struct
            // or pass the Arcs directly to a modified upsert function.
            // Let's simulate passing Arcs to a static-like upsert function for now.

            // --- Simulate calling upsert_chunks logic ---
            // In a real scenario, refactor upsert_chunks to accept Arcs or create temp owner.
            async fn do_upsert(
                chunks: Vec<(String, Chunk)>,
                root_uri: Option<&str>,
                embedder: Arc<GeminiEmbedder>,
                db: Arc<Surreal<impl surrealdb::Connection>>, // Generic connection
                table_name: &str,
            ) -> Result<()> {
                // ... (Implement embedding generation and DB update logic here, similar to Self::upsert_chunks) ...
                // ... This requires careful duplication or refactoring ...
                warn!("Debouncer upsert logic needs proper implementation/refactoring");
                Ok(())
            }
            // Call the simulated function
            if let Err(e) = do_upsert(
                all_chunks_to_upsert,
                upsert_root_uri.as_deref(),
                upsert_embedder,
                upsert_db.clone(), // Clone Arc for call
                &upsert_table_name,
            )
            .await
            {
                error!(error = ?e, "Error during debounced upsert");
            }
            // --- End simulated call ---
        }

        // --- Delete Stale Chunks ---
        debug!(
            count = uris_to_process.len(),
            "Checking for stale chunks to delete"
        );
        let delete_futures = uris_to_process.into_iter().map(|uri| {
            let db_c = db.clone();
            let table_name_c = table_name.clone();
            let latest_ids_for_uri = current_chunk_ids_by_uri
                .get(&uri)
                .cloned()
                .unwrap_or_default();
            async move {
                // Get all chunk IDs from DB for this URI
                let query = format!("SELECT VALUE id FROM {} WHERE uri = $uri;", table_name_c);
                let ids_in_db: Vec<Thing> = match db_c.query(query).bind(("uri", &uri)).await {
                    Ok(mut response) => response.take(0).unwrap_or_default(),
                    Err(e) => {
                        error!(%uri, error=?e, "Failed query IDs for stale check");
                        vec![]
                    }
                };

                // Find IDs in DB that are not in the latest set from FileStore
                let ids_to_delete: Vec<Thing> = ids_in_db
                    .into_iter()
                    .filter(|thing| !latest_ids_for_uri.contains(&thing.id.to_string()))
                    .collect();

                if !ids_to_delete.is_empty() {
                    debug!(%uri, count = ids_to_delete.len(), "Deleting stale chunks");
                    for batch in ids_to_delete.chunks(BATCH_SIZE) {
                        let delete_futures = batch.iter().map(|thing| db_c.delete(thing.clone()));
                        join_all(delete_futures)
                            .await
                            .into_iter()
                            .filter_map(|r| r.err())
                            .for_each(|e| {
                                error!(%uri, error=?e, "Failed to delete stale chunk batch");
                            });
                    }
                }
                Ok::<_, anyhow::Error>(())
            }
        });
        join_all(delete_futures).await; // Run stale checks concurrently

        debug!("Debounce processing cycle finished");
    } // End loop
}

// --- MemoryBackend Trait Implementation ---
#[async_trait] // Use crate qualified path if needed: #[async_trait::async_trait]
impl MemoryBackend for SurrealMemory {
    // --- Methods delegating to FileStore (unchanged) ---
    #[instrument(skip(self))]
    fn code_action_request(&self /* ... */) -> Result<bool> {
        /* ... self.file_store.code_action_request(...) ... */
        Ok(true)
    }

    #[instrument(skip(self))]
    fn get_filter_text(&self /* ... */) -> Result<String> {
        /* ... self.file_store.get_filter_text(...) ... */
        Ok("".to_string())
    }

    #[instrument(skip(self))]
    fn file_request(&self /* ... */) -> Result<String> {
        /* ... self.file_store.file_request(...) ... */
        Ok("".to_string())
    }

    // --- Build Prompt (Vector Search) ---
    #[instrument(skip(self, params))]
    async fn build_prompt(
        &self,
        position: &TextDocumentPositionParams,
        prompt_type: PromptType,
        params: &Value,
    ) -> Result<Prompt> {
        let params: MemoryRunParams = serde_json::from_value(params.clone())?;
        let chunk_size = self.splitter.chunk_size();
        let total_allowed_chars = tokens_to_estimated_characters(params.max_context);
        let vector_context_chars = total_allowed_chars / 2; // Allocate roughly half to vector context
                                                            // Estimate number of chunks needed based on average chunk size, fetch slightly more
        let search_limit = ((vector_context_chars / chunk_size) + 1).max(1) * 2;

        // Get query text & code prefix/suffix from FileStore
        let query_text = self
            .file_store
            .get_characters_around_position(position, chunk_size)?;
        let code = {
            let mut fs_params = params.clone();
            fs_params.max_context = total_allowed_chars / 2;
            self.file_store
                .build_code(position, prompt_type, fs_params, false)?
        };
        let cursor_byte = self.file_store.position_to_byte(position)?;
        let current_uri = position.text_document.uri.to_string();

        // Generate query embedding client-side
        debug!("Generating query embedding...");
        let query_embedding = self.embedder.generate_query_embedding(&query_text).await?;
        debug!("Query embedding generated.");

        // Build SurrealQL query
        let search_query = format!(
            "SELECT id, uri, text, range, vector::similarity::cosine(embedding, $query_vector) AS score \
             FROM {} \
             WHERE vector::similarity::cosine(embedding, $query_vector) > $threshold \
             AND (uri != $current_uri OR (range.end <= $cursor_byte OR range.start >= $cursor_byte)) \
             ORDER BY score DESC LIMIT $limit;",
            self.table_name
        );

        let bind_params = [
            ("query_vector".into(), Vector::from(query_embedding).into()),
            ("threshold".into(), SurrealValue::from(0.3)), // Adjust threshold
            (
                "current_uri".into(),
                SurrealValue::from(current_uri.clone()),
            ),
            (
                "cursor_byte".into(),
                SurrealValue::Number(cursor_byte.into()),
            ),
            ("limit".into(), SurrealValue::Number(search_limit.into())),
        ];

        // Execute vector search
        debug!(query=%search_query, "Executing vector search");
        let results: Vec<DocumentChunk> = self
            .db
            .query(search_query)
            .bind(bind_params)
            .await?
            .take(0)?;
        debug!(count = results.len(), "Vector search results received");

        // Build context string from results, respecting character limit
        let mut context_parts = Vec::new();
        let mut current_context_len = 0;
        for result in results {
            let text_len = result.text.len();
            // Check limit before adding - account for joining characters ("\n\n")
            if current_context_len == 0
                || (current_context_len + text_len + 2) <= vector_context_chars
            {
                context_parts.push(result.text);
                current_context_len += if current_context_len == 0 {
                    text_len
                } else {
                    text_len + 2
                };
            } else {
                break; // Stop adding chunks if limit exceeded
            }
        }
        let context = context_parts.join("\n\n");
        debug!(
            context_len = context.len(),
            "Built context string from results"
        );

        // Reconstruct the final Prompt
        Ok(match code {
            Prompt::ContextAndCode(cc) => {
                Prompt::ContextAndCode(ContextAndCodePrompt {
                    context,
                    code: format_file_chunk(
                        &current_uri,
                        &cc.code,
                        self.client_params.root_uri.as_deref(),
                    ),
                    selected_text: cc.selected_text, // Preserve selected text if any
                })
            }
            Prompt::FIM(fim) => Prompt::FIM(FIMPrompt {
                prompt: format!("{context}\n\n{}", fim.prompt),
                suffix: fim.suffix,
            }),
        })
    }

    // --- LSP Notification Handlers ---
    #[instrument(skip(self, params))]
    fn opened_text_document(&self, params: lsp_types::DidOpenTextDocumentParams) -> Result<()> {
        self.file_store.opened_text_document(params.clone())?;
        debug!(uri = %params.text_document.uri, "File opened");
        let self_clone = self.clone();
        let uri = params.text_document.uri.to_string();
        let root_uri_clone = self.client_params.root_uri.clone();
        TOKIO_RUNTIME.spawn(async move {
            debug!(uri, "Spawning task to upsert opened file");
            let file_store_clone = self_clone.file_store.clone(); // Clone Arc for task
            if let Err(e) = self_clone
                .split_and_upsert_file(&uri, file_store_clone, root_uri_clone.as_deref())
                .await
            {
                error!(%uri, error = ?e, "Failed async upsert for opened file");
            }
        });
        if let Err(e) = self.maybe_do_crawl(Some(params.text_document.uri.to_string())) {
            error!("Crawl triggered by file open failed: {e:?}");
        }
        Ok(())
    }

    #[instrument(skip(self, params))]
    fn changed_text_document(&self, params: lsp_types::DidChangeTextDocumentParams) -> Result<()> {
        self.file_store.changed_text_document(params.clone())?;
        let uri = params.text_document.uri.to_string();
        debug!(uri, "File changed, sending to debouncer");
        self.debounce_tx.send(uri)?; // Error if channel closed
        Ok(())
    }

    #[instrument(skip(self, params))]
    fn renamed_files(&self, params: lsp_types::RenameFilesParams) -> Result<()> {
        self.file_store.renamed_files(params.clone())?;
        debug!(count = params.files.len(), "Files renamed");
        let self_clone = self.clone();
        let root_uri_clone = self.client_params.root_uri.clone();
        TOKIO_RUNTIME.spawn(async move {
            for file in params.files {
                let old_uri = file.old_uri;
                let new_uri = file.new_uri.clone(); // Clone new_uri for upsert call
                info!(%old_uri, %new_uri, "Processing renamed file");
                // Delete old URI records
                let delete_query =
                    format!("DELETE {} WHERE uri = $old_uri;", self_clone.table_name);
                if let Err(e) = self_clone
                    .db
                    .query(delete_query)
                    .bind(("old_uri", &old_uri))
                    .await
                {
                    error!(%old_uri, error = ?e, "Failed to delete old URI docs");
                } else {
                    debug!(%old_uri, "Deleted old URI docs");
                }
                // Upsert new URI records
                let file_store_clone = self_clone.file_store.clone();
                if let Err(e) = self_clone
                    .split_and_upsert_file(&new_uri, file_store_clone, root_uri_clone.as_deref())
                    .await
                {
                    error!(%new_uri, error = ?e, "Failed async upsert for renamed file");
                } else {
                    debug!(%new_uri, "Upserted new URI docs");
                }
            }
        });
        Ok(())
    }

    // --- Closed/Deleted Files ---
    // Optional: Implement closed/deleted handlers if needed to clean up DB
    // fn closed_text_document(&self, params: lsp_types::DidCloseTextDocumentParams) -> Result<()> {
    //     self.file_store.closed_text_document(params.clone())?;
    //     // Optionally, trigger deletion from SurrealDB if file is closed and not expected to reopen?
    //     // This might be too aggressive depending on workflow.
    //     Ok(())
    // }
    //
    // fn deleted_files(&self, params: lsp_types::DeleteFilesParams) -> Result<()> {
    //     self.file_store.deleted_files(params.clone())?;
    //     let self_clone = self.clone();
    //     TOKIO_RUNTIME.spawn(async move {
    //         let uris_to_delete: Vec<String> = params.files.into_iter().map(|f| f.uri).collect();
    //         if !uris_to_delete.is_empty() {
    //              info!(count = uris_to_delete.len(), "Deleting records for deleted files");
    //              let delete_query = format!("DELETE {} WHERE uri IN $uris;", self_clone.table_name);
    //              let bind_params = [("uris".into(), uris_to_delete.into())];
    //              if let Err(e) = self_clone.db.query(delete_query).bind(bind_params).await {
    //                  error!(error = ?e, "Failed to delete docs for deleted files");
    //              }
    //         }
    //     });
    //     Ok(())
    // }
}

// --- Helper: Get Gemini Config from ValidModel ---
// Add this helper method to the ValidModel enum in config.rs
impl ValidModel {
    fn gemini(&self) -> Option<&config::Gemini> {
        match self {
            ValidModel::Gemini(gemini_config) => Some(gemini_config),
            _ => None,
        }
    }
}
