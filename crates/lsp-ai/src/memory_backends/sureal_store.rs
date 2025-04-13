use crate::memory::{MemoryBackend, MemoryFilter, MemoryItem};
use async_trait::async_trait;
use std::collections::HashMap;
use surrealdb::{
    engine::local::{Db, Mem}, // Use `File` instead of `Mem` later if you want persistent storage
    sql::Thing,
    Surreal,
};

#[derive(Debug, Clone)]
pub struct SurrealStore {
    client: Surreal<Db>,
}

#[derive(Debug, Clone)]
pub struct SurrealConfig {
    pub engine: String,       // "mem" or "file"
    pub path: Option<String>, // Optional path if using file engine
}

impl SurrealStore {
    pub async fn from_config(config: SurrealConfig) -> Self {
        let db = match config.engine.as_str() {
            "file" => {
                let path = config
                    .path
                    .unwrap_or_else(|| "./surreal_mem.db".to_string());
                surrealdb::engine::local::Db::new::<surrealdb::engine::local::File>(&path).unwrap()
            }
            _ => surrealdb::engine::local::Db::new::<Mem>().unwrap(),
        };

        let client = Surreal::new(db).unwrap();
        client.use_ns("lsp_ai").use_db("memory").await.unwrap();

        SurrealStore { client }
    }
}

#[async_trait]
impl MemoryBackend for SurrealStore {
    async fn add_memory(&self, item: MemoryItem) {
        let _: Result<Thing, _> = self.client.create("memory").content(item).await;
    }

    async fn get_relevant_memories(
        &self,
        query: &str,
        _max_results: usize,
        _filters: &MemoryFilter,
    ) -> Vec<MemoryItem> {
        let query = format!(
            "SELECT * FROM memory WHERE content CONTAINS '{}'",
            query.replace('\'', "''") // escape single quotes
        );

        match self.client.query(query).await {
            Ok(mut res) => res.take::<Vec<MemoryItem>>(0).unwrap_or_default(),
            Err(_) => vec![],
        }
    }
}
