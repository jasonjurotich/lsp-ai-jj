use anyhow::Context;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tracing::{info, instrument};

use super::TransformerBackend;
use crate::{
  config,
  memory_backends::{ContextAndCodePrompt, Prompt},
  transformer_worker::{
    DoGenerationResponse, DoGenerationStreamResponse, GenerationStreamRequest,
  },
  utils::format_prompt_in_str,
};

fn format_gemini_contents(
  messages: &[GeminiContent],
  prompt: &ContextAndCodePrompt,
) -> Vec<GeminiContent> {
  messages
    .iter()
    .map(|m| {
      GeminiContent::new(
        m.role.to_owned(),
        m.parts
          .iter()
          .map(|p| Part {
            text: format_prompt_in_str(&p.text, prompt),
          })
          .collect(),
      )
    })
    .collect()
}

const fn max_tokens_default() -> usize {
  64
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Part {
  pub(crate) text: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GeminiContent {
  pub role: String,
  pub parts: Vec<Part>,
}

impl GeminiContent {
  pub fn new(role: String, parts: Vec<Part>) -> Self {
    Self { role, parts }
  }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename = "camelCase")]
#[serde(deny_unknown_fields)]
pub(crate) struct GeminiGenerationConfig {
  #[serde(default)]
  #[serde(rename = "stopSequences")]
  pub(crate) stop_sequences: Vec<String>,
  #[serde(rename = "maxOutputTokens")]
  #[serde(default = "max_tokens_default")]
  pub(crate) max_output_tokens: usize,
  pub(crate) temperature: Option<f32>,
  #[serde(rename = "topP")]
  pub(crate) top_p: Option<f32>,
  #[serde(rename = "topK")]
  pub(crate) top_k: Option<f32>,
}

// NOTE: We cannot deny unknown fields as the provided parameters may contain other fields relevant to other processes
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename = "camelCase")]
pub(crate) struct GeminiRunParams {
  contents: Vec<GeminiContent>,
  #[serde(rename = "systemInstruction")]
  system_instruction: GeminiContent,
  #[serde(rename = "generationConfig")]
  generation_config: Option<GeminiGenerationConfig>,
}

pub(crate) struct Gemini {
  configuration: config::Gemini,
}

impl Gemini {
  pub(crate) fn new(configuration: config::Gemini) -> Self {
    Self { configuration }
  }

  fn get_token(&self) -> anyhow::Result<String> {
    if let Some(env_var_name) = &self.configuration.auth_token_env_var_name {
      Ok(std::env::var(env_var_name)?)
    } else if let Some(token) = &self.configuration.auth_token {
      Ok(token.to_string())
    } else {
      anyhow::bail!(
                "set `auth_token_env_var_name` or `auth_token` to use an Gemini compatible API"
            )
    }
  }

  async fn get_chat(
    &self,
    messages: Vec<GeminiContent>,
    params: GeminiRunParams,
  ) -> anyhow::Result<String> {
    let client = reqwest::Client::new();
    let token = self.get_token()?;
    let params = json!({
         "contents": messages,
         "systemInstruction": params.system_instruction,
         "generationConfig": params.generation_config,
    });

    info!(
      "Calling Gemini compatible chat API with parameters:\n{}",
      serde_json::to_string_pretty(&params).unwrap()
    );

    let res: serde_json::Value = client
      .post(
        self
          .configuration
          .chat_endpoint
          .as_ref()
          .context("must specify `chat_endpoint` to use gemini")?
          .to_owned()
          + self.configuration.model.as_ref()
          + ":generateContent?key="
          + token.as_ref(),
      )
      .header("Content-Type", "application/json")
      .json(&params)
      .send()
      .await?
      .json()
      .await?;

    info!(
      "Raw response from Gemini API: {}",
      serde_json::to_string_pretty(&res).unwrap_or_default()
    );

    if let Some(error) = res.get("error") {
      anyhow::bail!("{:?}", error.to_string())
    } else if let Some(candidates) = res.get("candidates") {
      Ok(
        candidates
          .get(0)
          .context("API response missing candidate at index 0")?
          .get("content")
          .context("Candidate missing 'content' field")?
          .get("parts")
          .context("Content missing 'parts' field")?
          .get(0)
          .context("Parts array is empty")?
          .get("text")
          .context("Part missing 'text' field")?
          .as_str() // Convert to &str before converting to String if possible
          .context("Part 'text' field is not a string")?
          .to_string(),
      ) // Convert valid &str to String

    // .get(0)
    // .unwrap()
    // .get("content")
    // .unwrap()
    // .get("parts")
    // .unwrap()
    // .get(0)
    // .unwrap()
    // .get("text")
    // .unwrap()
    // .clone()
    // .to_string())
    } else {
      anyhow::bail!("Unknown error while making request to Gemini: {:?}", res);
    }
  }
  async fn do_chat_completion(
    &self,
    prompt: &Prompt,
    params: GeminiRunParams,
  ) -> anyhow::Result<String> {
    match prompt {
      Prompt::ContextAndCode(code_and_context) => {
        let messages =
          format_gemini_contents(&params.contents, code_and_context);
        self.get_chat(messages, params).await
      }
      _ => anyhow::bail!("Google Gemini backend does not yet support FIM"),
    }
  }
}

#[async_trait::async_trait]
impl TransformerBackend for Gemini {
  #[instrument(skip(self))]
  async fn do_generate(
    &self,
    prompt: &Prompt,
    params: Value,
  ) -> anyhow::Result<DoGenerationResponse> {
    info!("we are right before creating a GeminiRunParams struct");
    let params: GeminiRunParams = serde_json::from_value(params)?;
    info!(
      "we are right after creating a GeminiRunParams struct, which is {:?}",
      params
    );
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
  use serde_json::json;

  #[tokio::test]
  async fn gemini_chat_do_generate() -> anyhow::Result<()> {
    let configuration: config::Gemini = serde_json::from_value(json!({
        "chat_endpoint": "https://generativelanguage.googleapis.com/v1beta/models/",
        "model": "gemini-1.5-flash",
        "auth_token_env_var_name": "GEMINI_API_KEY",
    }))?;
    let gemini = Gemini::new(configuration);
    let prompt = Prompt::default_with_cursor();
    let run_params = json!({
        "systemInstruction": {
            "role": "system",
            "parts": [{
                "text": "You are a helpful and willing chatbot that will do whatever the user asks"
            }]
        },
        "generationConfig": {
            "maxOutputTokens": 10
        },
        "contents": [
          {
            "role": "user",
            "parts":[{
             "text": "Pretend you're a snowman and stay in character for each response."}]
            },
          {
            "role": "model",
            "parts":[{
             "text": "Hello! It's so cold! Isn't that great?"}]
            },
          {
            "role": "user",
            "parts":[{
             "text": "What's your favorite season of the year?"}]
            }
         ]
    });
    let response = gemini.do_generate(&prompt, run_params).await?;
    dbg!(&response.generated_text);
    assert!(!response.generated_text.is_empty());
    Ok(())
  }
}
