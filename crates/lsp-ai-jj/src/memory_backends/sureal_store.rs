use crate::memory::{MemoryBackend, MemoryFilter, MemoryItem};
use async_trait::async_trait;
use std::collections::HashMap;
use surrealdb::{
    engine::local::{Db, Mem}, // Use `File` instead of `Mem` later if you want persistent storage
    sql::Thing,
    Surreal,
};

// #[derive(Debug, Clone)]
// pub struct SurrealStore {
//     client: Surreal<Db>,
// }

// #[derive(Debug, Clone)]
// pub struct SurrealConfig {
//     pub engine: String,       // "mem" or "file"
//     pub path: Option<String>, // Optional path if using file engine
// }

// impl SurrealStore {
//     pub async fn from_config(config: SurrealConfig) -> Self {
//         let db = match config.engine.as_str() {
//             "file" => {
//                 let path = config
//                     .path
//                     .unwrap_or_else(|| "./surreal_mem.db".to_string());
//                 surrealdb::engine::local::Db::new::<surrealdb::engine::local::File>(&path).unwrap()
//             }
//             _ => surrealdb::engine::local::Db::new::<Mem>().unwrap(),
//         };

//         let client = Surreal::new(db).unwrap();
//         client.use_ns("lsp_ai").use_db("memory").await.unwrap();

//         SurrealStore { client }
//     }
// }

// #[async_trait]
// impl MemoryBackend for SurrealStore {
//     async fn add_memory(&self, item: MemoryItem) {
//         let _: Result<Thing, _> = self.client.create("memory").content(item).await;
//     }

//     async fn get_relevant_memories(
//         &self,
//         query: &str,
//         _max_results: usize,
//         _filters: &MemoryFilter,
//     ) -> Vec<MemoryItem> {
//         let query = format!(
//             "SELECT * FROM memory WHERE content CONTAINS '{}'",
//             query.replace('\'', "''") // escape single quotes
//         );

//         match self.client.query(query).await {
//             Ok(mut res) => res.take::<Vec<MemoryItem>>(0).unwrap_or_default(),
//             Err(_) => vec![],
//         }
//     }
// }

#[derive(Debug, Clone)]
pub struct SurrealMemoryBackend {
    client: Arc<Surreal<Client>>,
}

impl SurrealMemoryBackend {
    pub async fn new() -> Self {
        let client = Surreal::new::<Client>("127.0.0.1:8000").await.unwrap();

        client
            .signin(Root {
                username: "root",
                password: "root",
            })
            .await
            .unwrap();

        client.use_ns("lsp").use_db("memory").await.unwrap();

        SurrealMemoryBackend {
            client: Arc::new(client),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurrealMemoryItem {
    pub id: Option<String>,
    pub role: String,
    pub content: String,
    pub timestamp: Option<String>,
    pub embedding: Option<Vec<f32>>,
}

impl From<SurrealMemoryItem> for MemoryItem {
    fn from(item: SurrealMemoryItem) -> Self {
        MemoryItem {
            id: item.id,
            role: item.role,
            content: item.content,
            timestamp: item.timestamp,
            embedding: item.embedding,
        }
    }
}

#[async_trait]
impl MemoryBackend for SurrealMemoryBackend {
    async fn add_memory(&self, mut item: MemoryItem) {
        if item.embedding.is_none() {
            item.embedding = embed_with_gemini(&item.content).await;
        }

        let record = SurrealMemoryItem {
            id: item.id,
            role: item.role,
            content: item.content,
            timestamp: item.timestamp,
            embedding: item.embedding,
        };

        let _: Result<surrealdb::sql::Thing, _> =
            self.client.create("memory").content(record).await;
    }

    async fn get_relevant_memories(
        &self,
        query: &str,
        max_results: usize,
        _filters: &MemoryFilter,
    ) -> Vec<MemoryItem> {
        let query_embedding = match embed_with_gemini(query).await {
            Some(e) => e,
            None => return vec![],
        };

        let array_str = format!(
            "[{}]",
            query_embedding
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );

        let query = format!(
            r#"
            SELECT *, vector::similarity::cosine(embedding, {embedding}) AS score 
            FROM memory 
            WHERE embedding IS NOT NONE 
            ORDER BY score DESC 
            LIMIT {limit}
            "#,
            embedding = array_str,
            limit = max_results,
        );

        match self.client.query(query).await {
            Ok(mut res) => res
                .take::<Vec<SurrealMemoryItem>>(0)
                .unwrap_or_default()
                .into_iter()
                .map(Into::into)
                .collect(),
            Err(_) => vec![],
        }
    }

    async fn clear(&self) {
        let _: Result<(), _> = self.client.query("DELETE memory;").await;
    }
}

async fn embed_with_gemini(text: &str) -> Option<Vec<f32>> {
    // Replace with your actual Gemini API key
    let api_key = std::env::var("GEMINI_API_KEY").ok()?;

    let client = Client::new();

    let res = client
        .post(&format!(
            "https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={}",
            api_key
        ))
        .json(&json!({
            "content": {
                "parts": [
                    { "text": text }
                ]
            }
        }))
        .send()
        .await
        .ok()?;

    let json: serde_json::Value = res.json().await.ok()?;
    let values = json["embedding"]["values"].as_array()?;

    Some(
        values
            .iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect(),
    )
}
