use anyhow::Context;
use fxhash::FxBuildHasher;
use lsp_types::{
    DidChangeTextDocumentParams, DidOpenTextDocumentParams, Range, RenameFilesParams,
    TextDocumentIdentifier, TextDocumentPositionParams,
};
use ordered_float::OrderedFloat;
use parking_lot::{Mutex, RwLock};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    io::Read,
    sync::{
        mpsc::{self, Sender},
        Arc,
    },
    time::Duration,
};
use tokio::time;
use tracing::{error, instrument, warn};

#[cfg(feature = "simsimd")]
use simsimd::{BinarySimilarity, SpatialSimilarity};

#[cfg(feature = "rayon")]
use rayon::iter::ParallelIterator;

use crate::{
    config::{self, Config, VectorDataType},
    crawl::Crawl,
    embedding_models::{EmbeddingModel, EmbeddingPurpose},
    memory_backends::MemoryRunParams,
    splitters::{ByteRange, Chunk, Splitter},
    utils::{format_file_chunk, tokens_to_estimated_characters, TOKIO_RUNTIME},
};

use super::{
    file_store::{AdditionalFileStoreParams, FileStore},
    ContextAndCodePrompt, FIMPrompt, MemoryBackend, Prompt, PromptType,
};

type IndexMap<K, V> = indexmap::IndexMap<K, V, FxBuildHasher>;

#[cfg(not(feature = "simsimd"))]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

#[cfg(not(feature = "simsimd"))]
fn hamming_distance(a: &[u8], b: &[u8]) -> usize {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x ^ y).count_ones() as usize)
        .sum()
}

struct StoredChunkUpsert {
    range: ByteRange,
    index: Option<usize>,
    vec: Option<Vec<f32>>,
    text: Option<String>,
}

impl StoredChunkUpsert {
    fn new(
        range: ByteRange,
        index: Option<usize>,
        vec: Option<Vec<f32>>,
        text: Option<String>,
    ) -> Self {
        Self {
            range,
            index,
            vec,
            text,
        }
    }
}

fn quantize(embedding: &[f32]) -> Vec<u8> {
    assert!(embedding.len() % 8 == 0);
    let bytes: Vec<u8> = embedding.iter().map(|x| x.clamp(0., 1.) as u8).collect();
    let mut quantised = Vec::with_capacity(embedding.len() / 8);
    for i in (0..bytes.len()).step_by(8) {
        let mut byte = 0u8;
        for j in 0..8 {
            byte |= bytes[i + j] << j;
        }
        quantised.push(byte);
    }

    quantised
}

enum StoredChunkVec {
    F32(Vec<f32>),
    Binary(Vec<u8>),
}

impl StoredChunkVec {
    fn new(data_type: VectorDataType, vec: Vec<f32>) -> Self {
        match data_type {
            VectorDataType::F32 => StoredChunkVec::F32(vec),
            VectorDataType::Binary => StoredChunkVec::Binary(quantize(&vec)),
        }
    }
}

struct StoredChunk {
    uri: String,
    vec: StoredChunkVec,
    text: String,
    range: ByteRange,
}

impl StoredChunk {
    fn new(uri: String, vec: StoredChunkVec, text: String, range: ByteRange) -> Self {
        Self {
            uri,
            vec,
            text,
            range,
        }
    }
}

struct VectorStoreInner {
    store: IndexMap<String, Vec<StoredChunk>>,
    data_type: VectorDataType,
}

impl VectorStoreInner {
    fn new(data_type: VectorDataType) -> Self {
        Self {
            data_type,
            store: IndexMap::default(),
        }
    }

    fn sync_file_chunks(
        &mut self,
        uri: &str,
        chunks_to_upsert: Vec<StoredChunkUpsert>,
        limit_chunks: Option<usize>,
    ) -> anyhow::Result<()> {
        match self.store.get_mut(uri) {
            Some(chunks) => {
                for chunk in chunks_to_upsert.into_iter() {
                    match (chunk.index, chunk.vec, chunk.text) {
                        // If we supply the index, we are editing the chunk
                        (Some(index), None, None) => chunks[index].range = chunk.range,
                        (Some(index), Some(vec), Some(text)) => {
                            chunks[index] = StoredChunk::new(
                                uri.to_string(),
                                StoredChunkVec::new(self.data_type, vec),
                                text,
                                chunk.range,
                            )
                        }
                        // If we don't supply the index, push the chunk on the end
                        (None, Some(vec), Some(text)) => chunks.push(StoredChunk::new(
                            uri.to_string(),
                            StoredChunkVec::new(self.data_type, vec),
                            text,
                            chunk.range,
                        )),
                        _ => {
                            anyhow::bail!("malformed StoredChunkUpsert - upsert must have index or vec and text")
                        }
                    }
                }
                if let Some(size) = limit_chunks {
                    chunks.truncate(size)
                }
            }
            None => {
                let chunks: anyhow::Result<Vec<StoredChunk>> = chunks_to_upsert
                    .into_iter()
                    .map(|c| {
                        Ok(StoredChunk::new(
                            uri.to_string(),
                            StoredChunkVec::new(
                                self.data_type,
                                c.vec
                                    .context("the vec for new StoredChunks cannot be empty")?,
                            ),
                            c.text
                                .context("the text for new StoredChunks cannot be empty")?,
                            c.range,
                        ))
                    })
                    .collect();
                self.store.insert(uri.to_string(), chunks?);
            }
        }
        Ok(())
    }

    fn rename_file(&mut self, old_uri: &str, new_uri: &str) -> anyhow::Result<()> {
        let old_chunks = self
            .store
            .swap_remove(old_uri)
            .with_context(|| format!("cannot rename non-existing file: {old_uri}"))?;
        self.store.insert(new_uri.to_string(), old_chunks);
        Ok(())
    }

    fn search(
        &self,
        limit: usize,
        rerank_top_k: Option<usize>,
        embedding: Vec<f32>,
        current_uri: &str,
        current_byte: usize,
    ) -> anyhow::Result<Vec<String>> {
        let scv_embedding = StoredChunkVec::new(self.data_type, embedding.clone());
        let find_limit = match rerank_top_k {
            Some(rerank) => rerank,
            None => limit,
        };
        let results: anyhow::Result<Vec<BTreeMap<_, _>>> =
            self.store
                .par_values()
                .try_fold_with(BTreeMap::new(), |mut acc, chunks| {
                    for chunk in chunks {
                        let score = match (&chunk.vec, &scv_embedding) {
                            (StoredChunkVec::F32(vec1), StoredChunkVec::F32(vec2)) => {
                                #[cfg(feature = "simsimd")]
                                {
                                    OrderedFloat(
                                        SpatialSimilarity::dot(vec1, vec2).context("vector length mismatch when taking the dot product")? as f32
                                    )
                                }
                                #[cfg(not(feature = "simsimd"))]
                                {
                                    OrderedFloat(dot_product(&vec1, &vec2))
                                }
                            }
                            (StoredChunkVec::Binary(vec1), StoredChunkVec::Binary(vec2)) => {
                                #[cfg(feature = "simsimd")]
                                {
                                    OrderedFloat(
                                        embedding.len() as f32
                                            - BinarySimilarity::hamming(vec1, vec2).context("vector length mismatch when taking the hamming distance")?
                                                as f32,
                                    )
                                }
                                #[cfg(not(feature = "simsimd"))]
                                {
                                    OrderedFloat(
                                        (embedding.len() - hamming_distance(&vec1, &vec2)) as f32,
                                    )
                                }
                            }
                            _ => anyhow::bail!("mismatch between vector data types in search"),
                        };
                        if acc.is_empty() {
                            acc.insert(score, chunk);
                        } else if acc.first_key_value().unwrap().0 < &score {
                            // We want to get limit + 1 here in case the limit is 1 and then we filter the chunk out later
                            if acc.len() == find_limit + 1 {
                                acc.pop_first();
                            }
                            acc.insert(score, chunk);
                        }
                    }
                    Ok(acc)
                })
                .collect();
        let mut top_results = BTreeMap::new();
        for result in results? {
            for (sub_result_score, sub_result_chunk) in result {
                let sub_result_score = if rerank_top_k.is_some() {
                    match &sub_result_chunk.vec {
                        StoredChunkVec::Binary(b) => {
                            // Convert binary vector to f32 vec
                            let mut b_f32 = vec![];
                            for byte in b {
                                for i in 0..8 {
                                    let x = byte >> (8 - i) & 1;
                                    b_f32.push(x as f32);
                                }
                            }
                            b_f32.truncate(embedding.len());
                            #[cfg(feature = "simsimd")]
                            {
                                OrderedFloat(
                                    SpatialSimilarity::dot(&b_f32, &embedding)
                                        .context("mismatch in vector length when taking the dot product when re-ranking")?
                                        as f32,
                                )
                            }
                            #[cfg(not(feature = "simsimd"))]
                            {
                                OrderedFloat(dot_product(&b_f32, &embedding) as f32)
                            }
                        }
                        StoredChunkVec::F32(_) => {
                            warn!("Not reranking in vector_store because vectors are not binary");
                            sub_result_score
                        }
                    }
                } else {
                    sub_result_score
                };

                // Filter out chunks that are in the current chunk
                if sub_result_chunk.uri == current_uri
                    && sub_result_chunk.range.start_byte <= current_byte
                    && sub_result_chunk.range.end_byte >= current_byte
                {
                    continue;
                }
                if top_results.is_empty() {
                    top_results.insert(sub_result_score, sub_result_chunk);
                } else if top_results.first_key_value().unwrap().0 < &sub_result_score {
                    if top_results.len() == limit {
                        top_results.pop_first();
                    }
                    top_results.insert(sub_result_score, sub_result_chunk);
                }
            }
        }
        Ok(top_results
            .into_iter()
            .rev()
            .map(|(_, chunk)| chunk.text.to_string())
            .collect())
    }
}

pub(crate) struct VectorStore {
    file_store: Arc<FileStore>,
    crawl: Option<Arc<Mutex<Crawl>>>,
    splitter: Arc<Box<dyn Splitter + Send + Sync>>,
    embedding_model: Arc<Box<dyn EmbeddingModel + Send + Sync>>,
    vector_store: Arc<RwLock<VectorStoreInner>>,
    config: Config,
    debounce_tx: Sender<String>,
}

impl VectorStore {
    pub(crate) fn new(
        mut vector_store_config: config::VectorStore,
        config: Config,
    ) -> anyhow::Result<Self> {
        let crawl = vector_store_config
            .crawl
            .take()
            .map(|x| Arc::new(Mutex::new(Crawl::new(x, config.clone()))));
        let splitter: Arc<Box<dyn Splitter + Send + Sync>> =
            Arc::new(vector_store_config.splitter.clone().try_into()?);
        let embedding_model: Arc<Box<dyn EmbeddingModel + Send + Sync>> =
            Arc::new(vector_store_config.embedding_model.try_into()?);
        let file_store = Arc::new(FileStore::new_with_params(
            config::FileStore::new_without_crawl(),
            config.clone(),
            AdditionalFileStoreParams::new(splitter.does_use_tree_sitter()),
        )?);
        let vector_store = Arc::new(RwLock::new(VectorStoreInner::new(
            vector_store_config.data_type,
        )));

        // Debounce document changes to reduce the number of embeddings we perform
        let (debounce_tx, debounce_rx) = mpsc::channel::<String>();
        let task_embedding_model = embedding_model.clone();
        let task_vector_store = vector_store.clone();
        let task_file_store = file_store.clone();
        let task_splitter = splitter.clone();
        let task_root_uri = config.client_params.root_uri.clone();
        TOKIO_RUNTIME.spawn(async move {
            let duration = Duration::from_millis(500);
            let mut file_uris = Vec::new();
            loop {
                time::sleep(duration).await;
                let new_uris: Vec<String> = debounce_rx.try_iter().collect();
                if !new_uris.is_empty() {
                    for uri in new_uris {
                        if !file_uris.iter().any(|p| *p == uri) {
                            file_uris.push(uri);
                        }
                    }
                } else {
                    if file_uris.is_empty() {
                        continue;
                    }

                    for uri in file_uris {
                        let chunks = {
                            let file_map = task_file_store.file_map().read();
                            let file = match file_map
                                .get(&uri)
                                .context("file not found for debounced embedding")
                            {
                                Ok(file) => file,
                                Err(e) => {
                                    error!("{e:?}");
                                    continue;
                                }
                            };
                            task_splitter.split(file)
                        };
                        let chunks_size = chunks.len();

                        // This is not as efficient as it could be, but it is ok for now
                        // We may want a better system than string comparing constantly
                        let chunks_to_upsert = match task_vector_store.read().store.get(&uri) {
                            Some(existing_chunks) => {
                                let mut chunks_to_upsert = vec![];
                                for (i, chunk) in chunks.into_iter().enumerate() {
                                    if let Some(existing_chunk) = existing_chunks.get(i) {
                                        // Edit chunk start and end byte
                                        let has_chunk_changed = chunk.text != existing_chunk.text;
                                        if !has_chunk_changed {
                                            if chunk.range.start_byte
                                                != existing_chunk.range.start_byte
                                                || chunk.range.end_byte
                                                    != existing_chunk.range.end_byte
                                            {
                                                chunks_to_upsert.push(StoredChunkUpsert::new(
                                                    chunk.range,
                                                    Some(i),
                                                    None,
                                                    None,
                                                ));
                                            }
                                        } else {
                                            chunks_to_upsert.push(StoredChunkUpsert::new(
                                                chunk.range,
                                                Some(i),
                                                None,
                                                Some(format_file_chunk(
                                                    &uri,
                                                    &chunk.text,
                                                    task_root_uri.as_deref(),
                                                )),
                                            ));
                                        }
                                    } else {
                                        chunks_to_upsert.push(StoredChunkUpsert::new(
                                            chunk.range,
                                            None,
                                            None,
                                            Some(format_file_chunk(
                                                &uri,
                                                &chunk.text,
                                                task_root_uri.as_deref(),
                                            )),
                                        ));
                                    }
                                }
                                chunks_to_upsert
                            }
                            None => chunks
                                .into_iter()
                                .map(|chunk| {
                                    StoredChunkUpsert::new(
                                        chunk.range,
                                        None,
                                        None,
                                        Some(format_file_chunk(
                                            &uri,
                                            &chunk.text,
                                            task_root_uri.as_deref(),
                                        )),
                                    )
                                })
                                .collect(),
                        };
                        // Embed all chunks with text
                        match task_embedding_model
                            .embed(
                                chunks_to_upsert
                                    .iter()
                                    .filter(|c| c.text.is_some())
                                    .map(|c| c.text.as_ref().unwrap().as_str())
                                    .collect(),
                                EmbeddingPurpose::Storage,
                            )
                            .await
                        {
                            Ok(mut embeddings) => {
                                let chunks_to_upsert: Vec<StoredChunkUpsert> = chunks_to_upsert
                                    .into_iter()
                                    .map(|mut c| {
                                        if c.text.is_some() {
                                            c.vec = Some(embeddings.remove(0))
                                        }
                                        c
                                    })
                                    .collect();
                                if let Err(e) = task_vector_store.write().sync_file_chunks(
                                    &uri,
                                    chunks_to_upsert,
                                    Some(chunks_size),
                                ) {
                                    error!("{e:?}");
                                }
                            }
                            Err(e) => {
                                error!("{e:?}");
                            }
                        }
                    }

                    file_uris = vec![];
                }
            }
        });

        let s = Self {
            file_store,
            crawl,
            splitter,
            embedding_model,
            vector_store,
            config,
            debounce_tx,
        };
        if let Err(e) = s.maybe_do_crawl(None) {
            error!("{e:?}")
        }
        Ok(s)
    }

    fn upsert_chunks(&self, uri: &str, chunks: Vec<Chunk>) {
        let task_uri = uri.to_string();
        let task_embedding_model = self.embedding_model.clone();
        let task_vector_store = self.vector_store.clone();
        let root_uri = self.config.client_params.root_uri.clone();
        TOKIO_RUNTIME.spawn(async move {
            match task_embedding_model
                .embed(
                    chunks.iter().map(|c| c.text.as_str()).collect(),
                    EmbeddingPurpose::Storage,
                )
                .await
            {
                Ok(embeddings) => {
                    let embedded_chunks: Vec<StoredChunkUpsert> = chunks
                        .into_iter()
                        .zip(embeddings)
                        .map(|(chunk, embedding)| {
                            StoredChunkUpsert::new(
                                chunk.range,
                                None,
                                Some(embedding),
                                Some(format_file_chunk(
                                    &task_uri,
                                    &chunk.text,
                                    root_uri.as_deref(),
                                )),
                            )
                        })
                        .collect();
                    if let Err(e) =
                        task_vector_store
                            .write()
                            .sync_file_chunks(&task_uri, embedded_chunks, None)
                    {
                        error!("{e:?}");
                    }
                }
                Err(e) => {
                    error!("{e:?}");
                }
            }
        });
    }

    fn maybe_do_crawl(&self, triggered_file: Option<String>) -> anyhow::Result<()> {
        if let Some(crawl) = &self.crawl {
            let mut total_bytes = 0;
            crawl
                .lock()
                .maybe_do_crawl(triggered_file, |config, path| {
                    // Break if total bytes is over the max crawl memory
                    if total_bytes as u64 >= config.max_crawl_memory {
                        warn!("Ending crawl early due to `max_crawl_memory` restraint");
                        return Ok(false);
                    }

                    // This means it has been opened before
                    let uri = format!("file://{path}");
                    if self.file_store.contains_file(&uri) {
                        return Ok(true);
                    }

                    // Open the file and see if it is small enough to read
                    let mut f = std::fs::File::open(path)?;
                    let metadata = f.metadata()?;
                    if metadata.len() > config.max_file_size {
                        warn!("Skipping file: {path} because it is too large");
                        return Ok(true);
                    }

                    // Read the file contents
                    let mut contents = vec![];
                    f.read_to_end(&mut contents)?;
                    let contents = String::from_utf8(contents)?;
                    total_bytes += contents.len();

                    // Store the file
                    let chunks = self.splitter.split_file_contents(&uri, &contents);
                    self.upsert_chunks(&uri, chunks);
                    Ok(true)
                })?;
        }
        Ok(())
    }
}

#[async_trait::async_trait]
impl MemoryBackend for VectorStore {
    #[instrument(skip(self))]
    fn code_action_request(
        &self,
        text_document_identifier: &TextDocumentIdentifier,
        range: &Range,
        trigger: &str,
    ) -> anyhow::Result<bool> {
        self.file_store
            .code_action_request(text_document_identifier, range, trigger)
    }

    #[instrument(skip(self))]
    fn file_request(
        &self,
        text_document_identifier: &TextDocumentIdentifier,
    ) -> anyhow::Result<String> {
        self.file_store.file_request(text_document_identifier)
    }

    #[instrument(skip(self))]
    fn opened_text_document(&self, params: DidOpenTextDocumentParams) -> anyhow::Result<()> {
        let uri = params.text_document.uri.to_string();
        self.file_store.opened_text_document(params)?;

        let file_map = self.file_store.file_map().read();
        let file = file_map.get(&uri).context("file not found")?;
        let chunks = self.splitter.split(file);
        self.upsert_chunks(&uri, chunks);

        if let Err(e) = self.maybe_do_crawl(Some(uri)) {
            error!("{e:?}")
        }
        Ok(())
    }

    #[instrument(skip(self))]
    fn changed_text_document(&self, params: DidChangeTextDocumentParams) -> anyhow::Result<()> {
        let uri = params.text_document.uri.to_string();
        self.file_store.changed_text_document(params.clone())?;
        self.debounce_tx.send(uri)?;
        Ok(())
    }

    #[instrument(skip(self))]
    fn renamed_files(&self, params: RenameFilesParams) -> anyhow::Result<()> {
        // TODO: Finish this
        self.file_store.renamed_files(params.clone())?;
        for file in params.files {
            let uri = file.new_uri;
            let old_uri = file.old_uri;
            if let Err(e) = self.vector_store.write().rename_file(&old_uri, &uri) {
                error!("{e:?}");
            }
        }
        Ok(())
    }

    #[instrument(skip(self))]
    fn get_filter_text(&self, position: &TextDocumentPositionParams) -> anyhow::Result<String> {
        self.file_store.get_filter_text(position)
    }

    #[instrument(skip(self))]
    async fn build_prompt(
        &self,
        position: &TextDocumentPositionParams,
        prompt_type: PromptType,
        params: &Value,
    ) -> anyhow::Result<Prompt> {
        let params: MemoryRunParams = params.try_into()?;
        let chunk_size = self.splitter.chunk_size();
        let total_allowed_characters = tokens_to_estimated_characters(params.max_context);

        // Build the query
        let query = self
            .file_store
            .get_characters_around_position(position, chunk_size)?;

        // Build the prompt
        let mut file_store_params = params.clone();
        file_store_params.max_context = chunk_size;
        let code = self
            .file_store
            .build_code(position, prompt_type, file_store_params, false)?;

        // Get the byte of the cursor
        let cursor_byte = self.file_store.position_to_byte(position)?;

        // Get the embedding
        let embedding = self
            .embedding_model
            .embed(vec![&query], EmbeddingPurpose::Retrieval)
            .await?
            .into_iter()
            .nth(0)
            .context("no embeddings returned")?;

        // Get the context
        let limit = (total_allowed_characters / chunk_size).saturating_sub(1);
        let context = self
            .vector_store
            .read()
            .search(
                limit,
                None,
                embedding,
                position.text_document.uri.as_ref(),
                cursor_byte,
            )?
            .join("\n\n");

        // Reconstruct the prompts
        Ok(match code {
            Prompt::ContextAndCode(context_and_code) => {
                Prompt::ContextAndCode(ContextAndCodePrompt {
                    context: context.to_owned(),
                    code: format_file_chunk(
                        position.text_document.uri.as_ref(),
                        &context_and_code.code,
                        self.config.client_params.root_uri.as_deref(),
                    ),
                    selected_text: None,
                })
            }
            Prompt::FIM(fim) => Prompt::FIM(FIMPrompt {
                prompt: format!("{context}\n\n{}", fim.prompt),
                suffix: fim.suffix,
            }),
        })
    }
}
