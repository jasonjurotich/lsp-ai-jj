use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use tracing::{info, instrument};

use crate::{
    config::{self, ChatMessage, FIM},
    memory_backends::Prompt,
    transformer_worker::{
        DoGenerationResponse, DoGenerationStreamResponse, GenerationStreamRequest,
    },
    utils::{format_chat_messages, format_prompt},
};

use super::TransformerBackend;

// NOTE: We cannot deny unknown fields as the provided parameters may contain other fields relevant to other processes
#[derive(Debug, Deserialize)]
pub(crate) struct OllamaRunParams {
    pub(crate) fim: Option<FIM>,
    messages: Option<Vec<ChatMessage>>,
    #[serde(default)]
    options: HashMap<String, Value>,
    system: Option<String>,
    template: Option<String>,
    keep_alive: Option<String>,
}

pub(crate) struct Ollama {
    configuration: config::Ollama,
}

#[derive(Deserialize, Serialize)]
struct OllamaValidCompletionsResponse {
    response: String,
}

#[derive(Deserialize, Serialize)]
#[serde(untagged)]
enum OllamaCompletionsResponse {
    Success(OllamaValidCompletionsResponse),
    Error(OllamaError),
    Other(HashMap<String, Value>),
}

#[derive(Debug, Deserialize, Serialize)]
struct OllamaChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize, Serialize)]
struct OllamaError {
    error: Value,
}

#[derive(Deserialize, Serialize)]
struct OllamaValidChatResponse {
    message: OllamaChatMessage,
}

#[derive(Deserialize, Serialize)]
#[serde(untagged)]
enum OllamaChatResponse {
    Success(OllamaValidChatResponse),
    Error(OllamaError),
    Other(HashMap<String, Value>),
}

impl Ollama {
    #[instrument]
    pub(crate) fn new(configuration: config::Ollama) -> Self {
        Self { configuration }
    }

    async fn get_completion(
        &self,
        prompt: &str,
        params: OllamaRunParams,
    ) -> anyhow::Result<String> {
        let client = reqwest::Client::new();
        let params = json!({
            "model": self.configuration.model,
            "prompt": prompt,
            "options": params.options,
            "keep_alive": params.keep_alive,
            "raw": true,
            "stream": false
        });
        info!(
            "Calling Ollama compatible completions API with parameters:\n{}",
            serde_json::to_string_pretty(&params).unwrap()
        );
        let res: OllamaCompletionsResponse = client
            .post(
                self.configuration
                    .generate_endpoint
                    .as_deref()
                    .unwrap_or("http://localhost:11434/api/generate"),
            )
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .json(&params)
            .send()
            .await?
            .json()
            .await?;
        info!(
            "Response from Ollama compatible completions API:\n{}",
            serde_json::to_string_pretty(&res).unwrap()
        );
        match res {
            OllamaCompletionsResponse::Success(mut resp) => Ok(std::mem::take(&mut resp.response)),
            OllamaCompletionsResponse::Error(error) => {
                anyhow::bail!(
                    "making Ollama completions request: {:?}",
                    error.error.to_string()
                )
            }
            OllamaCompletionsResponse::Other(other) => {
                anyhow::bail!(
                    "unknown error while making Ollama completions request: {:?}",
                    other
                )
            }
        }
    }

    async fn get_chat(
        &self,
        messages: Vec<ChatMessage>,
        params: OllamaRunParams,
    ) -> anyhow::Result<String> {
        let client = reqwest::Client::new();
        let params = json!({
            "model": self.configuration.model,
            "system": params.system,
            "template": params.template,
            "messages": messages,
            "options": params.options,
            "keep_alive": params.keep_alive,
            "stream": false
        });
        info!(
            "Calling Ollama compatible chat API with parameters:\n{}",
            serde_json::to_string_pretty(&params).unwrap()
        );
        let res: OllamaChatResponse = client
            .post(
                self.configuration
                    .chat_endpoint
                    .as_deref()
                    .unwrap_or("http://localhost:11434/api/chat"),
            )
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .json(&params)
            .send()
            .await?
            .json()
            .await?;
        info!(
            "Response from Ollama compatible chat API:\n{}",
            serde_json::to_string_pretty(&res).unwrap()
        );
        match res {
            OllamaChatResponse::Success(mut resp) => Ok(std::mem::take(&mut resp.message.content)),
            OllamaChatResponse::Error(error) => {
                anyhow::bail!("making Ollama chat request: {:?}", error.error.to_string())
            }
            OllamaChatResponse::Other(other) => {
                anyhow::bail!(
                    "unknown error while making Ollama chat request: {:?}",
                    other
                )
            }
        }
    }

    async fn do_chat_completion(
        &self,
        prompt: &Prompt,
        params: OllamaRunParams,
    ) -> anyhow::Result<String> {
        match prompt {
            Prompt::ContextAndCode(code_and_context) => match &params.messages {
                Some(completion_messages) => {
                    let messages = format_chat_messages(completion_messages, code_and_context);
                    self.get_chat(messages, params).await
                }
                None => {
                    self.get_completion(&format_prompt(&code_and_context), params)
                        .await
                }
            },
            Prompt::FIM(fim) => match &params.fim {
                Some(fim_params) => {
                    self.get_completion(
                        &format!(
                            "{}{}{}{}{}",
                            fim_params.start,
                            fim.prompt,
                            fim_params.middle,
                            fim.suffix,
                            fim_params.end
                        ),
                        params,
                    )
                    .await
                }
                None => anyhow::bail!("Prompt type is FIM but no FIM parameters provided"),
            },
        }
    }
}

#[async_trait::async_trait]
impl TransformerBackend for Ollama {
    #[instrument(skip(self))]
    async fn do_generate(
        &self,
        prompt: &Prompt,

        params: Value,
    ) -> anyhow::Result<DoGenerationResponse> {
        let params: OllamaRunParams = serde_json::from_value(params)?;
        let generated_text = self.do_chat_completion(prompt, params).await?;
        Ok(DoGenerationResponse { generated_text })
    }

    #[instrument(skip(self))]
    async fn do_generate_stream(
        &self,
        request: &GenerationStreamRequest,
        _params: Value,
    ) -> anyhow::Result<DoGenerationStreamResponse> {
        anyhow::bail!("GenerationStream is not yet implemented")
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use serde_json::{from_value, json};

    #[tokio::test]
    async fn ollama_completion_do_generate() -> anyhow::Result<()> {
        let configuration: config::Ollama = from_value(json!({
            "model": "llama3",
        }))?;
        let ollama = Ollama::new(configuration);
        let prompt = Prompt::default_without_cursor();
        let run_params = json!({
            "options": {
                "num_predict": 4
            }
        });
        let response = ollama.do_generate(&prompt, run_params).await?;
        assert!(!response.generated_text.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn ollama_chat_do_generate() -> anyhow::Result<()> {
        let configuration: config::Ollama = from_value(json!({
            "model": "llama3",
        }))?;
        let ollama = Ollama::new(configuration);
        let prompt = Prompt::default_with_cursor();
        let run_params = json!({
            "messages": [
                {
                    "role": "system",
                    "content": "Test"
                },
                {
                    "role": "user",
                    "content": "Test {CONTEXT} - {CODE}"
                }
            ],
            "options": {
                "num_predict": 4
            }
        });
        let response = ollama.do_generate(&prompt, run_params).await?;
        assert!(!response.generated_text.is_empty());
        Ok(())
    }
}
