// crates/lsp-ai/src/chat_history.rs
#![allow(dead_code)] // Remove later if all functions are used

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use surrealdb::{
    engine::remote::ws::Ws, // Or your chosen engine client type
    opt::RecordId,          // Import RecordId trait for `.id()` method
    sql::{Datetime, Thing}, // Import Datetime for SurrealDB timestamps
    Surreal,
};
use tracing::{debug, error, info, instrument};

// --- Constants ---
const DEFAULT_HISTORY_TABLE: &str = "conversations";
const DEFAULT_HISTORY_LIMIT: usize = 50; // Default number of messages to retrieve

// --- Data Structures ---

/// Represents a single message in the chat history retrieved from the DB.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    // pub timestamp: Option<DateTime<Utc>>, // Optional: If needed outside the store
}

/// Structure stored in the SurrealDB 'conversations' table.
#[derive(Debug, Deserialize, Serialize)]
struct ConversationRecord {
    // We'll let SurrealDB manage the 'id' (Thing)
    session_id: String,  // URI of the chat file (e.g., .md file)
    timestamp: Datetime, // SurrealDB Datetime for ordering
    role: String,        // "user" or "model"
    content: String,
}

// --- ChatHistoryStore ---

/// Manages storing and retrieving chat history in SurrealDB.
#[derive(Clone, Debug)]
pub struct ChatHistoryStore {
    db: Arc<Surreal<Ws>>, // Using Ws client example
    table_name: String,
}

impl ChatHistoryStore {
    /// Creates a new ChatHistoryStore instance and ensures the necessary schema exists.
    pub async fn new(db: Arc<Surreal<Ws>>) -> Result<Self> {
        let store = Self {
            db,
            table_name: DEFAULT_HISTORY_TABLE.to_string(),
        };
        store
            .ensure_schema()
            .await
            .context("Failed to ensure chat history schema")?;
        Ok(store)
    }

    /// Defines the necessary table, fields, and indexes for chat history storage.
    #[instrument(skip(self))]
    async fn ensure_schema(&self) -> Result<()> {
        info!(table = %self.table_name, "Ensuring chat history schema exists...");

        // Define Table (optional, implicit creation works)
        // let q1 = format!("DEFINE TABLE {};", self.table_name);

        // Define Fields with types and assertions
        let q2 = format!(
            "DEFINE FIELD session_id ON TABLE {} TYPE string ASSERT $value != NONE; \
             DEFINE FIELD timestamp ON TABLE {} TYPE datetime ASSERT $value != NONE; \
             DEFINE FIELD role ON TABLE {} TYPE string ASSERT $value IN ['user', 'model']; \
             DEFINE FIELD content ON TABLE {} TYPE string;",
            self.table_name, self.table_name, self.table_name, self.table_name
        );

        // Define Indexes for efficient querying
        let index_name = format!("{}_session_ts_idx", self.table_name);
        let q3 = format!(
            "DEFINE INDEX {} ON TABLE {} COLUMNS session_id, timestamp;",
            index_name, self.table_name
        );

        // Execute schema definition queries (combine for efficiency if possible)
        // Note: Errors here might just mean schema already exists, which is fine.
        if let Err(e) = self.db.query(&q2).await {
            warn!(error = ?e, table = %self.table_name, "Failed to define fields (may already exist)");
        } else {
            debug!(table = %self.table_name, "Fields defined ok");
        }
        if let Err(e) = self.db.query(&q3).await {
            warn!(error = ?e, index=%index_name, "Failed to define index (may already exist)");
        } else {
            debug!(index=%index_name, "Index defined ok");
        }

        info!(table = %self.table_name, "Chat history schema check complete.");
        Ok(())
    }

    /// Saves a chat message to the history.
    #[instrument(skip(self, content))]
    pub async fn save_message(&self, session_id: &str, role: &str, content: &str) -> Result<()> {
        if content.trim().is_empty() {
            debug!("Skipping empty message save.");
            return Ok(());
        }
        let record = ConversationRecord {
            session_id: session_id.to_string(),
            timestamp: Datetime::from(Utc::now()), // Use current time
            role: role.to_string(),
            content: content.to_string(),
        };

        debug!(%session_id, %role, "Saving chat message");
        // Create record in the table, letting SurrealDB generate the ID
        let created: Option<ConversationRecord> = self
            .db
            .create(&self.table_name) // Use table name directly for creation
            .content(record)
            .await
            .context("Failed to save chat message to SurrealDB")?;

        if created.is_none() {
            // This case might indicate an issue, though create usually returns the created record.
            error!("SurrealDB create operation did not return the saved record.");
            // Depending on strictness, you might return an error here.
        } else {
            debug!(%session_id, "Chat message saved successfully.");
        }

        Ok(())
    }

    /// Retrieves the most recent messages for a given session.
    #[instrument(skip(self))]
    pub async fn get_history(
        &self,
        session_id: &str,
        limit: Option<usize>,
    ) -> Result<Vec<ChatMessage>> {
        let history_limit = limit.unwrap_or(DEFAULT_HISTORY_LIMIT);
        debug!(%session_id, limit = history_limit, "Retrieving chat history");

        // Select necessary fields, filter by session_id, order by timestamp descending, limit
        let query = format!(
            "SELECT role, content FROM {} WHERE session_id = $session_id ORDER BY timestamp DESC LIMIT $limit;",
            self.table_name
        );

        let mut result = self
            .db
            .query(query)
            .bind(("session_id", session_id))
            .bind(("limit", history_limit))
            .await
            .context("Failed to query chat history from SurrealDB")?;

        // Take the results from the first (and only) statement response
        let mut messages: Vec<ChatMessage> = result
            .take(0) // Takes the Vec<T> result from the 0th statement
            .context("Failed to decode chat history query results")?;

        // Messages are currently newest-first, reverse them for chronological order
        messages.reverse();

        debug!(%session_id, count = messages.len(), "Chat history retrieved");
        Ok(messages)
    }

    /// Deletes all messages for a given session.
    #[instrument(skip(self))]
    pub async fn delete_history(&self, session_id: &str) -> Result<()> {
        info!(%session_id, "Deleting chat history");

        let query = format!("DELETE {} WHERE session_id = $session_id;", self.table_name);

        // Execute the delete query
        // DELETE doesn't typically return detailed records unless specified with RETURN
        self.db
            .query(query)
            .bind(("session_id", session_id))
            .await
            .context("Failed to delete chat history from SurrealDB")?;

        info!(%session_id, "Chat history deleted successfully.");
        Ok(())
    }
}

// Optional: Define a trait if you want to abstract the history store later
// #[async_trait]
// pub trait ChatHistoryBackend: Send + Sync {
//     async fn save_message(&self, session_id: &str, role: &str, content: &str) -> Result<()>;
//     async fn get_history(&self, session_id: &str, limit: Option<usize>) -> Result<Vec<ChatMessage>>;
//     async fn delete_history(&self, session_id: &str) -> Result<()>;
// }

// #[async_trait]
// impl ChatHistoryBackend for ChatHistoryStore {
//     async fn save_message(&self, session_id: &str, role: &str, content: &str) -> Result<()> {
//         self.save_message(session_id, role, content).await
//     }
//     async fn get_history(&self, session_id: &str, limit: Option<usize>) -> Result<Vec<ChatMessage>> {
//         self.get_history(session_id, limit).await
//     }
//     async fn delete_history(&self, session_id: &str) -> Result<()> {
//         self.delete_history(session_id).await
//     }
// }




How to Use:

Initialization: Somewhere in your main server setup (e.g., where you initialize SurrealMemory), create the ChatHistoryStore:
Rust

// Assuming `db: Arc<Surreal<Ws>>` is your connected SurrealDB client
let chat_history_store = Arc::new(ChatHistoryStore::new(db.clone()).await?);
In LSP Chat Handler:
Get the URI of the .md file (session_id).
Call chat_history_store.get_history(&session_id, Some(20)).await to get previous messages.
After getting the user prompt and the model response:
chat_history_store.save_message(&session_id, "user", &user_prompt).await?
chat_history_store.save_message(&session_id, "model", &gemini_response).await?
In LSP Delete Command Handler:
Get the URI of the .md file (session_id).
Call chat_history_store.delete_history(&session_id).await?.
This module provides the necessary database interactions for managing persistent chat history within your specified .md file workflow. Remember to integrate the calls to these methods into your LSP action and command handlers.
