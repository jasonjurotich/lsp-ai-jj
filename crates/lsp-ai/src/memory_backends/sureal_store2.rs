
// #[cfg(not(feature = "simsimd"))]
// fn dot_product(a: &[f32], b: &[f32]) -> f32 {
//     a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
// }

// #[cfg(not(feature = "simsimd"))]
// fn hamming_distance(a: &[u8], b: &[u8]) -> usize {
//     a.iter()
//         .zip(b.iter())
//         .map(|(&x, &y)| (x ^ y).count_ones() as usize)
//         .sum()
// }


// dot_product and hamming_distance functions
// Original Purpose: Calculate vector similarity directly in Rust code for the in-memory search. dot_product for f32 vectors, hamming_distance for the quantized u8 vectors.
// SurrealDB Replacement: These functions are generally no longer needed for the primary search. SurrealDB calculates similarity inside the database using its built-in functions within your SELECT query.
// dot_product function: Replaced by SurrealDB's vector::similarity::dot() or vector::similarity::cosine() in queries.
// How: You'll use functions like vector::similarity::cosine(embedding, $query_embedding) or vector::similarity::dot(...) directly in your SurrealQL WHERE or SELECT clause, as shown in the search example previously. The database engine handles the calculation efficiently, often using the vector index.

//------------------------------------------


// struct StoredChunkUpsert {
//     range: ByteRange,
//     index: Option<usize>,
//     vec: Option<Vec<f32>>,
//     text: Option<String>,
// }

// impl StoredChunkUpsert {
//     fn new(
//         range: ByteRange,
//         index: Option<usize>,
//         vec: Option<Vec<f32>>,
//         text: Option<String>,
//     ) -> Self {
//         Self {
//             range,
//             index,
//             vec,
//             text,
//         }
//     }
// }

// StoredChunkUpsert struct:
// Original Purpose: A temporary data holder to represent a change (add or update) to be applied to the in-memory IndexMap. It bundles the range, optional index (for updates), optional new vector, and optional new text.
// SurrealDB Adaptation: You might still use a similar concept or struct temporarily within your Rust code (like in the debounce logic) to gather the necessary information for a database operation. However, its ultimate purpose changes:
// Instead of directly updating an in-memory Vec<StoredChunk>, the data from this struct (range, vec, text) will be used to:
// Construct a ChunkRecord object (the serde-compatible struct for SurrealDB).
// Formulate a CREATE or UPDATE SurrealQL query to send to the database.
// The index: Option<usize> field might be re-purposed or replaced. Instead of an index into a Vec, you'd need a way to identify the specific record to update in SurrealDB (e.g., using the uri and start_byte as a unique key, or retrieving the SurrealDB Thing ID first).
// Replacement Summary: The concept of bundling update info might remain, but it feeds database queries/records, not direct memory manipulation.


// First, define how we will identify a specific chunk record in the DB
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RecordIdentifier {
    UriAndStartByte(String, usize),
}

// Define what fields might be updated in an existing record
#[derive(Debug, Clone, Default)]
pub struct ChunkUpdateData {
    pub text: Option<String>,        // New formatted text, if changed
    pub embedding: Option<Vec<f32>>, // New embedding, if text changed
    pub range: Option<ByteRange>,    // New range, if position changed
}

// The main enum replacing the role of StoredChunkUpsert
#[derive(Debug, Clone)]
pub enum ChunkDbOperation {
    /// Instruction to create a new chunk record in the database.
    Create {
        uri: String,
        text: String,        // The full formatted text
        range: ByteRange,
        embedding: Vec<f32>, // The full embedding vector
    },
    /// Instruction to update an existing chunk record.
    Update {
        identifier: RecordIdentifier, // How to find the record to update
        updates: ChunkUpdateData,   // What fields to change
    },
    /// Instruction to delete an existing chunk record.
    Delete {
        identifier: RecordIdentifier, // How to find the record to delete
    },
}


//------------------------------------------




// fn quantize(embedding: &[f32]) -> Vec<u8> {
//     assert!(embedding.len() % 8 == 0);
//     let bytes: Vec<u8> = embedding.iter().map(|x| x.clamp(0., 1.) as u8).collect();
//     let mut quantised = Vec::with_capacity(embedding.len() / 8);
//     for i in (0..bytes.len()).step_by(8) {
//         let mut byte = 0u8;
//         for j in 0..8 {
//             byte |= bytes[i + j] << j;
//         }
//         quantised.push(byte);
//     }

//     quantised
// }

// enum StoredChunkVec {
//     F32(Vec<f32>),
//     Binary(Vec<u8>),
// }

// impl StoredChunkVec {
//     fn new(data_type: VectorDataType, vec: Vec<f32>) -> Self {
//         match data_type {
//             VectorDataType::F32 => StoredChunkVec::F32(vec),
//             VectorDataType::Binary => StoredChunkVec::Binary(quantize(&vec)),
//         }
//     }
// }

// quantize function and StoredChunkVec enum
// Original Purpose: To optionally convert f32 embedding vectors into a more compact binary u8 format (quantize) and allow storing either type (StoredChunkVec). This was useful for saving memory and using the fast hamming_distance.
// SurrealDB Replacement: These are likely removed entirely. SurrealDB's vector search is designed to work most effectively with standard floating-point vectors (vector<float, N>). Storing f32 vectors directly allows you to use cosine, dot product, or euclidean similarity accurately with optimized indexing (like HNSW). Storing binary vectors would prevent you from easily using these standard similarity metrics within SurrealDB's vector functions.
// How: You will store your embeddings as Vec<f32> in your Rust struct (ChunkRecord) and map that directly to the vector<float, N> type in your SurrealDB table schema.















// struct StoredChunk {
//     uri: String,
//     vec: StoredChunkVec,
//     text: String,
//     range: ByteRange,
// }


//------------------------------------------
// Original Purpose: The primary struct holding the chunk data in memory, including the StoredChunkVec which could be f32 or binary.
// SurrealDB Replacement: This is replaced by the serde-compatible struct designed to map to your SurrealDB table (we called it ChunkRecord previously.


#[derive(Debug, Serialize, Deserialize, Clone)]
struct ChunkRecord {
    // #[serde(skip_serializing_if = "Option::is_none")] // Optionally skip ID on create
    // id: Option<Thing>, // SurrealDB Record ID (optional for retrieval)
    uri: String,
    text: String,        // The formatted chunk text
    start_byte: usize,   // Using usize for consistency, check DB type compatibility (might need i64)
    end_byte: usize,
    embedding: Vec<f32>, // Store f32 vectors directly
}


// Requires #[derive(Serialize, Deserialize)].
// The vec: StoredChunkVec field is replaced by embedding: Vec<f32>.
// The range: ByteRange is typically flattened into start_byte and end_byte fields for easier database handling.




// struct VectorStoreInner {
//     store: IndexMap<String, Vec<StoredChunk>>,
//     data_type: VectorDataType,
// }

// Remember, the VectorStoreInner struct itself is removed. This functionality will be implemented as methods on your main VectorStore struct, using the surreal_client: Arc<Surreal<Any>>.


//------------------------------------------







// impl VectorStoreInner {
//     fn new(data_type: VectorDataType) -> Self {
//         Self {
//             data_type,
//             store: IndexMap::default(),
//         }
//     }

//     fn sync_file_chunks(
//         &mut self,
//         uri: &str,
//         chunks_to_upsert: Vec<StoredChunkUpsert>,
//         limit_chunks: Option<usize>,
//     ) -> anyhow::Result<()> {
//         match self.store.get_mut(uri) {
//             Some(chunks) => {
//                 for chunk in chunks_to_upsert.into_iter() {
//                     match (chunk.index, chunk.vec, chunk.text) {
//                         // If we supply the index, we are editing the chunk
//                         (Some(index), None, None) => chunks[index].range = chunk.range,
//                         (Some(index), Some(vec), Some(text)) => {
//                             chunks[index] = StoredChunk::new(
//                                 uri.to_string(),
//                                 StoredChunkVec::new(self.data_type, vec),
//                                 text,
//                                 chunk.range,
//                             )
//                         }
//                         // If we don't supply the index, push the chunk on the end
//                         (None, Some(vec), Some(text)) => chunks.push(StoredChunk::new(
//                             uri.to_string(),
//                             StoredChunkVec::new(self.data_type, vec),
//                             text,
//                             chunk.range,
//                         )),
//                         _ => {
//                             anyhow::bail!("malformed StoredChunkUpsert - upsert must have index or vec and text")
//                         }
//                     }
//                 }
//                 if let Some(size) = limit_chunks {
//                     chunks.truncate(size)
//                 }
//             }
//             None => {
//                 let chunks: anyhow::Result<Vec<StoredChunk>> = chunks_to_upsert
//                     .into_iter()
//                     .map(|c| {
//                         Ok(StoredChunk::new(
//                             uri.to_string(),
//                             StoredChunkVec::new(
//                                 self.data_type,
//                                 c.vec
//                                     .context("the vec for new StoredChunks cannot be empty")?,
//                             ),
//                             c.text
//                                 .context("the text for new StoredChunks cannot be empty")?,
//                             c.range,
//                         ))
//                     })
//                     .collect();
//                 self.store.insert(uri.to_string(), chunks?);
//             }
//         }
//         Ok(())
//     }



// sync_file_chunks Equivalent Functionality:

// Original: Updated an in-memory IndexMap based on StoredChunkUpsert.

// SurrealDB Equivalent: This isn't a single function but rather logic that processes a Vec<ChunkDbOperation> (the enum we defined) and executes different SurrealDB queries based on the operation type.


// Assuming 'op' is a ChunkDbOperation::Create variant
let record = ChunkRecord { // Build the record from op data
    uri: op.uri,
    text: op.text,
    start_byte: op.range.start_byte,
    end_byte: op.range.end_byte,
    embedding: op.embedding,
};

let created: Option<ChunkRecord> = self.surreal_client
    .create(&self.surreal_table_name) // self.surreal_table_name holds "chunks" etc.
    .content(record)
    .await?;


// Assuming 'op' is a ChunkDbOperation::Update variant
if let RecordIdentifier::UriAndStartByte(uri, start_byte) = &op.identifier {
    // Build the update map or struct dynamically based on op.updates
    let mut update_query = format!("UPDATE type::table($table) WHERE uri = $uri AND start_byte = $start_byte MERGE {{");
    let mut bindings = surrealdb::sql::Vars::new();
    bindings.insert("table".into(), self.surreal_table_name.clone().into());
    bindings.insert("uri".into(), uri.clone().into());
    bindings.insert("start_byte".into(), (*start_byte as i64).into()); // Use i64 for DB int usually

    let mut updates_added = false;
    if let Some(text) = &op.updates.text {
        update_query.push_str(" text: $text,");
        bindings.insert("text".into(), text.clone().into());
        updates_added = true;
    }
    if let Some(embedding) = &op.updates.embedding {
         update_query.push_str(" embedding: $embedding,");
         bindings.insert("embedding".into(), embedding.clone().into()); // Ensure embedding is compatible type
         updates_added = true;
    }
    if let Some(range) = &op.updates.range {
         update_query.push_str(" start_byte: $new_start, end_byte: $new_end,");
         bindings.insert("new_start".into(), (range.start_byte as i64).into());
         bindings.insert("new_end".into(), (range.end_byte as i64).into());
         updates_added = true;
    }

    if updates_added {
        // Remove trailing comma and close merge object/query
        update_query.pop(); // Remove last comma
        update_query.push_str(" };");

        let mut response = self.surreal_client.query(update_query).bind(bindings).await?;
        // Process response if needed, e.g., check updated record count
        let _updated: Vec<ChunkRecord> = response.take(0)?; // Example: deserialize updated records
    }
}


// Assuming 'op' is a ChunkDbOperation::Delete variant
if let RecordIdentifier::UriAndStartByte(uri, start_byte) = &op.identifier {
    let sql = "DELETE type::table($table) WHERE uri = $uri AND start_byte = $start_byte;";
    let mut response = self.surreal_client
        .query(sql)
        .bind(("table", &self.surreal_table_name))
        .bind(("uri", uri))
        .bind(("start_byte", *start_byte as i64)) // Use i64 for DB int
        .await?;
    // Process response if needed
    let _deleted: Vec<ChunkRecord> = response.take(0)?; // Example: deserialize deleted records
}



//------------------------------------------








//     fn rename_file(&mut self, old_uri: &str, new_uri: &str) -> anyhow::Result<()> {
//         let old_chunks = self
//             .store
//             .swap_remove(old_uri)
//             .with_context(|| format!("cannot rename non-existing file: {old_uri}"))?;
//         self.store.insert(new_uri.to_string(), old_chunks);
//         Ok(())
//     }

//     fn search(
//         &self,
//         limit: usize,
//         rerank_top_k: Option<usize>,
//         embedding: Vec<f32>,
//         current_uri: &str,
//         current_byte: usize,
//     ) -> anyhow::Result<Vec<String>> {
//         let scv_embedding = StoredChunkVec::new(self.data_type, embedding.clone());
//         let find_limit = match rerank_top_k {
//             Some(rerank) => rerank,
//             None => limit,
//         };
//         let results: anyhow::Result<Vec<BTreeMap<_, _>>> =
//             self.store
//                 .par_values()
//                 .try_fold_with(BTreeMap::new(), |mut acc, chunks| {
//                     for chunk in chunks {
//                         let score = match (&chunk.vec, &scv_embedding) {
//                             (StoredChunkVec::F32(vec1), StoredChunkVec::F32(vec2)) => {
//                                 #[cfg(feature = "simsimd")]
//                                 {
//                                     OrderedFloat(
//                                         SpatialSimilarity::dot(vec1, vec2).context("vector length mismatch when taking the dot product")? as f32
//                                     )
//                                 }
//                                 #[cfg(not(feature = "simsimd"))]
//                                 {
//                                     OrderedFloat(dot_product(&vec1, &vec2))
//                                 }
//                             }
//                             (StoredChunkVec::Binary(vec1), StoredChunkVec::Binary(vec2)) => {
//                                 #[cfg(feature = "simsimd")]
//                                 {
//                                     OrderedFloat(
//                                         embedding.len() as f32
//                                             - BinarySimilarity::hamming(vec1, vec2).context("vector length mismatch when taking the hamming distance")?
//                                                 as f32,
//                                     )
//                                 }
//                                 #[cfg(not(feature = "simsimd"))]
//                                 {
//                                     OrderedFloat(
//                                         (embedding.len() - hamming_distance(&vec1, &vec2)) as f32,
//                                     )
//                                 }
//                             }
//                             _ => anyhow::bail!("mismatch between vector data types in search"),
//                         };
//                         if acc.is_empty() {
//                             acc.insert(score, chunk);
//                         } else if acc.first_key_value().unwrap().0 < &score {
//                             // We want to get limit + 1 here in case the limit is 1 and then we filter the chunk out later
//                             if acc.len() == find_limit + 1 {
//                                 acc.pop_first();
//                             }
//                             acc.insert(score, chunk);
//                         }
//                     }
//                     Ok(acc)
//                 })
//                 .collect();
//         let mut top_results = BTreeMap::new();
//         for result in results? {
//             for (sub_result_score, sub_result_chunk) in result {
//                 let sub_result_score = if rerank_top_k.is_some() {
//                     match &sub_result_chunk.vec {
//                         StoredChunkVec::Binary(b) => {
//                             // Convert binary vector to f32 vec
//                             let mut b_f32 = vec![];
//                             for byte in b {
//                                 for i in 0..8 {
//                                     let x = byte >> (8 - i) & 1;
//                                     b_f32.push(x as f32);
//                                 }
//                             }
//                             b_f32.truncate(embedding.len());
//                             #[cfg(feature = "simsimd")]
//                             {
//                                 OrderedFloat(
//                                     SpatialSimilarity::dot(&b_f32, &embedding)
//                                         .context("mismatch in vector length when taking the dot product when re-ranking")?
//                                         as f32,
//                                 )
//                             }
//                             #[cfg(not(feature = "simsimd"))]
//                             {
//                                 OrderedFloat(dot_product(&b_f32, &embedding) as f32)
//                             }
//                         }
//                         StoredChunkVec::F32(_) => {
//                             warn!("Not reranking in vector_store because vectors are not binary");
//                             sub_result_score
//                         }
//                     }
//                 } else {
//                     sub_result_score
//                 };

//                 // Filter out chunks that are in the current chunk
//                 if sub_result_chunk.uri == current_uri
//                     && sub_result_chunk.range.start_byte <= current_byte
//                     && sub_result_chunk.range.end_byte >= current_byte
//                 {
//                     continue;
//                 }
//                 if top_results.is_empty() {
//                     top_results.insert(sub_result_score, sub_result_chunk);
//                 } else if top_results.first_key_value().unwrap().0 < &sub_result_score {
//                     if top_results.len() == limit {
//                         top_results.pop_first();
//                     }
//                     top_results.insert(sub_result_score, sub_result_chunk);
//                 }
//             }
//         }
//         Ok(top_results
//             .into_iter()
//             .rev()
//             .map(|(_, chunk)| chunk.text.to_string())
//             .collect())
//     }
// }















































// In your config structure
#[derive(Deserialize, Clone)] // If using serde for config loading
pub struct SurrealConfig {
    pub address: String,       // e.g., "ws://localhost:8000" or "rocksdb://path/to/db"
    pub username: Option<String>,
    pub password: Option<String>,
    pub namespace: String,     // e.g., "code_vectors"
    pub database: String,      // e.g., "main_db"
    pub table: String,         // e.g., "chunks"
}

// Add to your main Config

// Remember, the VectorStoreInner struct itself is removed. This functionality will be implemented as methods on your main VectorStore struct, using the surreal_client: Arc<Surreal<Any>>.


pub struct Config {
    // ... other fields
    pub surreal_config: SurrealConfig,
    #[serde(skip)] // Avoid trying to serialize/deserialize the client itself
    pub surreal_client: Option<Arc<Surreal<Any>>>, // Store the client handle here
}

use surrealdb::engine::any::Any; // Or specific engine
use surrealdb::Surreal;
use std::sync::Arc;

pub(crate) struct VectorStore {
    file_store: Arc<FileStore>,
    crawl: Option<Arc<Mutex<Crawl>>>,
    splitter: Arc<Box<dyn Splitter + Send + Sync>>,
    embedding_model: Arc<Box<dyn EmbeddingModel + Send + Sync>>,
    // vector_store: Arc<RwLock<VectorStoreInner>>, // REMOVE THIS
    config: Config, // Keep config
    surreal_client: Arc<Surreal<Any>>, // ADD THIS (or specific client type)
    debounce_tx: Sender<String>,
    surreal_table_name: String, // Store table name for convenience
}

// Inside VectorStore::new
let cfg = &config.surreal_config;
let client = Arc::new(Surreal::new::<Any>(&cfg.address).await?); // Use correct engine type
if let (Some(user), Some(pass)) = (&cfg.username, &cfg.password) {
    client.signin(surrealdb::opt::auth::Root {
        username: user.as_str(),
        password: pass.as_str(),
    }).await?;
}
client.use_ns(&cfg.namespace).use_db(&cfg.database).await?;

let surreal_table_name = cfg.table.clone();

// ... rest of initialization (FileStore, Splitter, EmbeddingModel, Debounce Task)

let s = Self {
    // ... other fields
    config,
    surreal_client: client,
    debounce_tx,
    surreal_table_name,
};
// ... maybe_do_crawl
Ok(s)
