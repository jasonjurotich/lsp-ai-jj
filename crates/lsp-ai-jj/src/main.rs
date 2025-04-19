use anyhow::Result;
use clap::Parser;
use directories::BaseDirs;
use directories::ProjectDirs;
use lsp_server::{
  Connection, ExtractError, Message, Notification, Request, RequestId,
};
use lsp_types::{
  request::{
    CodeActionRequest, CodeActionResolveRequest, Completion, Shutdown,
  },
  CodeActionOptions, CompletionOptions, DidChangeTextDocumentParams,
  DidOpenTextDocumentParams, RenameFilesParams, ServerCapabilities,
  TextDocumentSyncKind,
};
use std::fs::File;
use std::sync::Mutex;
use std::{
  collections::HashMap,
  fs,
  path::{Path, PathBuf},
  sync::{mpsc, Arc},
  thread,
};

use once_cell::sync::Lazy;

use tracing::{debug, error, info, warn};
use tracing_appender::rolling;
use tracing_appender::{self, non_blocking, non_blocking::WorkerGuard};
use tracing_error::ErrorLayer;
use tracing_subscriber::{
  layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, FmtSubscriber,
};

mod config;
mod crawl;
mod custom_requests;
mod embedding_models;
mod memory_backends;
mod memory_worker;
mod splitters;
#[cfg(feature = "llama_cpp")]
mod template;
mod transformer_backends;
mod transformer_worker;
mod utils;

use config::Config;
use custom_requests::generation::Generation;
use memory_backends::MemoryBackend;
use transformer_backends::TransformerBackend;
use transformer_worker::{CompletionRequest, GenerationRequest, WorkerRequest};

use crate::{
  custom_requests::generation_stream::GenerationStream,
  transformer_worker::GenerationStreamRequest,
};

fn notification_is<N: lsp_types::notification::Notification>(
  notification: &Notification,
) -> bool {
  notification.method == N::METHOD
}

fn request_is<R: lsp_types::request::Request>(request: &Request) -> bool {
  request.method == R::METHOD
}

fn cast<R>(
  req: Request,
) -> Result<(RequestId, R::Params), ExtractError<Request>>
where
  R: lsp_types::request::Request,
  R::Params: serde::de::DeserializeOwned,
{
  req.extract(R::METHOD)
}

// LSP-AI parameters
#[derive(Parser)]
#[command(version)]
struct Args {
  // Whether to use a custom log file
  #[arg(long, default_value_t = false)]
  use_seperate_log_file: bool,
  // A dummy argument for now
  #[arg(long, default_value_t = true)]
  stdio: bool,
  // JSON configuration file location
  #[arg(long, value_parser = utils::validate_file_exists, required = false)]
  config: Option<PathBuf>,
}

fn initialize_logging(args: &Args) -> Option<WorkerGuard> {
  // --- 1. Create the Filter ---
  let env_filter = EnvFilter::new(
    std::env::var("LSP_AI_LOG").unwrap_or_else(|_| "lsp_ai_jj=debug".into()),
  );

  // --- 2. Create the Stderr Layer (Always) ---
  // This layer formats messages and writes them to stderr.
  // NOTE you must use stderr and not stdout, let everything be an err
  let stderr_layer = tracing_subscriber::fmt::layer()
    .with_writer(std::io::stderr)
    .with_ansi(true)
    .with_target(true)
    .with_level(true);

  // --- 3. Conditionally Create File Layer Components ---
  let mut file_layer_option = None; // Will hold the configured file layer if successful
  let mut log_guard: Option<WorkerGuard> = None; // Holds the guard needed for the file writer

  if args.use_seperate_log_file {
    // NOTE this is the path!! ~/Library/Caches/com.jasonjurotich.LspAiJj/
    // NOTE you must run cargo run -- --use-seperate-log-file for this to work!
    if let Some(proj_dirs) =
      ProjectDirs::from("com", "jasonjurotich", "LspAiJj")
    {
      let log_dir = proj_dirs.cache_dir();
      if fs::create_dir_all(log_dir).is_ok() {
        let file_appender = rolling::daily(log_dir, "lsp_ai_jj.log");
        let (non_blocking_writer, guard) = non_blocking(file_appender);

        let file_layer = tracing_subscriber::fmt::layer()
          .with_writer(non_blocking_writer)
          .with_ansi(true)
          .with_target(true)
          .with_level(true);

        file_layer_option = Some(file_layer);
        log_guard = Some(guard);
      } else {
        // Use eprintln for errors occurring before logger is fully initialized
        eprintln!("lsp-ai-jj: Failed to create log directory {:?}. File logging disabled.", log_dir);
      }
    } else {
      eprintln!("lsp-ai-jj: Could not determine cache directory. File logging disabled.");
    }
  }

  // --- 4. Build the Subscriber ---
  // Start with the registry and add layers common to both scenarios
  let registry = tracing_subscriber::registry()
    .with(env_filter)
    .with(ErrorLayer::default())
    .with(stderr_layer); // Add stderr layer - it's always active

  // --- 5. Initialize Globally (Single Call) ---
  // Now, add the file layer *if it exists*, then initialize.
  let init_result = match file_layer_option {
    Some(file_layer) => {
      // Add the file layer we created earlier and initialize
      registry.with(file_layer).try_init()
    }
    None => {
      // No file layer was created, initialize with just stderr etc.
      registry.try_init()
    }
  };

  if !init_result.is_ok() {
    eprintln!("lsp-ai-jj: Failed to initialize tracing subscriber. Logging may not work.");
    // Clean up guard if init failed but guard was created
    if log_guard.is_some() {
      drop(log_guard.take());
    }
  }

  log_guard
}

fn load_config(
  args: &Args,
  init_args: serde_json::Value,
) -> anyhow::Result<serde_json::Value> {
  if let Some(config_path) = &args.config {
    let config_data = fs::read_to_string(config_path)?;
    let mut config = serde_json::from_str(&config_data)?;
    utils::merge_json(&mut config, &init_args);
    Ok(config)
  } else {
    Ok(init_args)
  }
}

fn main() -> Result<()> {
  // let _log_guard = &*LOG_GUARD;
  let args = Args::parse();
  let _log_guard = initialize_logging(&args);
  // init_logging(&args);

  info!("lsp-ai logger initialized starting server");

  if !args.stdio {
    error!("Only stdio communication mode is supported.");
    // Maybe return an error or exit gracefully
    return Err(anyhow::anyhow!(
      "Invalid arguments: only --stdio true is supported"
    ));
  }

  info!("Setting up LSP connection via stdio.");
  let (connection, io_threads) = Connection::stdio();
  info!("LSP stdio connection established.");

  let server_capabilities = serde_json::to_value(ServerCapabilities {
    completion_provider: Some(CompletionOptions::default()),
    text_document_sync: Some(lsp_types::TextDocumentSyncCapability::Kind(
      TextDocumentSyncKind::INCREMENTAL,
    )),
    code_action_provider: Some(
      lsp_types::CodeActionProviderCapability::Options(CodeActionOptions {
        resolve_provider: Some(true),
        ..Default::default()
      }),
    ),
    ..Default::default()
  })?;
  let initialization_args = connection.initialize(server_capabilities)?;

  if let Err(e) =
    main_loop(connection, load_config(&args, initialization_args)?)
  {
    error!("{e:?}");
  }

  io_threads.join()?;
  Ok(())
}

fn main_loop(connection: Connection, args: serde_json::Value) -> Result<()> {
  info!("Starting main loop. Loaded config: {:?}", args); // Log config on start

  // Build our configuration
  let config = Config::new(args)?;

  // Wrap the connection for sharing between threads
  let connection = Arc::new(connection);

  // Our channel we use to communicate with our transformer worker
  let (transformer_tx, transformer_rx) = mpsc::channel();

  // The channel we use to communicate with our memory worker
  let (memory_tx, memory_rx) = mpsc::channel();

  // Setup the transformer worker
  let memory_backend: Box<dyn MemoryBackend + Send + Sync> =
    config.clone().try_into()?;
  let memory_worker_thread =
    thread::spawn(move || memory_worker::run(memory_backend, memory_rx));

  // Setup our transformer worker
  let transformer_backends: HashMap<
    String,
    Box<dyn TransformerBackend + Send + Sync>,
  > = config
    .config
    .models
    .clone()
    .into_iter()
    .map(|(key, value)| Ok((key, value.try_into()?)))
    .collect::<anyhow::Result<
      HashMap<String, Box<dyn TransformerBackend + Send + Sync>>,
    >>()?;
  let thread_connection = connection.clone();
  let thread_memory_tx = memory_tx.clone();
  let thread_config = config.clone();
  let transformer_worker_thread = thread::spawn(move || {
    transformer_worker::run(
      transformer_backends,
      thread_memory_tx,
      transformer_rx,
      thread_connection,
      thread_config,
    )
  });

  for msg in &connection.receiver {
    match msg {
      Message::Request(req) => {
        if request_is::<Shutdown>(&req) {
          memory_tx.send(memory_worker::WorkerRequest::Shutdown)?;
          if let Err(e) = memory_worker_thread.join() {
            std::panic::resume_unwind(e)
          }
          transformer_tx.send(WorkerRequest::Shutdown)?;
          if let Err(e) = transformer_worker_thread.join() {
            std::panic::resume_unwind(e)
          }
          connection.handle_shutdown(&req)?;
          return Ok(());
        } else if request_is::<Completion>(&req) {
          match cast::<Completion>(req) {
            Ok((id, params)) => {
              let completion_request = CompletionRequest::new(id, params);
              transformer_tx
                .send(WorkerRequest::Completion(completion_request))?;
            }
            Err(err) => error!("{err:?}"),
          }
        } else if request_is::<Generation>(&req) {
          match cast::<Generation>(req) {
            Ok((id, params)) => {
              info!("Received Generation request (id: {:?}). Parsed GenerationParams:",id);
              info!("{:?}", params);

              info!(
                "GenerationParams.parameters field contents:\n{}",
                serde_json::to_string_pretty(&params.parameters)
                  .unwrap_or_else(|e| format!(
                    "Failed to serialize parameters field: {}",
                    e
                  ))
              );

              info!("main_loop: Sending WorkerRequest::Generation to transformer_worker.");
              let generation_request = GenerationRequest::new(id, params);
              transformer_tx
                .send(WorkerRequest::Generation(generation_request))?;
            }
            Err(err) => {
              error!(
                "Failed to cast received request to Generation: {:?}",
                err
              );
            }
          }
        } else if request_is::<GenerationStream>(&req) {
          match cast::<GenerationStream>(req) {
            Ok((id, params)) => {
              let generation_stream_request =
                GenerationStreamRequest::new(id, params);
              transformer_tx.send(WorkerRequest::GenerationStream(
                generation_stream_request,
              ))?;
            }
            Err(err) => error!("{err:?}"),
          }
        } else if request_is::<CodeActionRequest>(&req) {
          match cast::<CodeActionRequest>(req) {
            Ok((id, params)) => {
              let code_action_request =
                transformer_worker::CodeActionRequest::new(id, params);
              transformer_tx
                .send(WorkerRequest::CodeActionRequest(code_action_request))?;
            }
            Err(err) => error!("{err:?}"),
          }
        } else if request_is::<CodeActionResolveRequest>(&req) {
          match cast::<CodeActionResolveRequest>(req) {
            Ok((id, params)) => {
              let code_action_request =
                transformer_worker::CodeActionResolveRequest::new(id, params);
              transformer_tx.send(WorkerRequest::CodeActionResolveRequest(
                code_action_request,
              ))?;
            }
            Err(err) => error!("{err:?}"),
          }
        } else {
          error!("Unsupported command - see the wiki for a list of supported commands: {req:?}")
        }
      }
      Message::Notification(not) => {
        if notification_is::<lsp_types::notification::DidOpenTextDocument>(&not)
        {
          let params: DidOpenTextDocumentParams =
            serde_json::from_value(not.params)?;
          memory_tx
            .send(memory_worker::WorkerRequest::DidOpenTextDocument(params))?;
        } else if notification_is::<
          lsp_types::notification::DidChangeTextDocument,
        >(&not)
        {
          let params: DidChangeTextDocumentParams =
            serde_json::from_value(not.params)?;
          memory_tx.send(
            memory_worker::WorkerRequest::DidChangeTextDocument(params),
          )?;
        } else if notification_is::<lsp_types::notification::DidRenameFiles>(
          &not,
        ) {
          let params: RenameFilesParams = serde_json::from_value(not.params)?;
          memory_tx
            .send(memory_worker::WorkerRequest::DidRenameFiles(params))?;
        }
      }
      _ => (),
    }
  }
  Ok(())
}
