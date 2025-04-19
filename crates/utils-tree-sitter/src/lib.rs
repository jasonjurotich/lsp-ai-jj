use thiserror::Error;
use tree_sitter::{LanguageError, Parser};

#[derive(Error, Debug)]
pub enum GetParserError {
    #[error("no parser found for extension")]
    NoParserFoundForExtension(String),
    #[error("no parser found for extension")]
    NoLanguageFoundForExtension(String),
    #[error("loading grammer")]
    LoadingGrammer(#[from] LanguageError),
}

fn get_extension_for_language(extension: &str) -> Result<String, GetParserError> {
    Ok(match extension {
        "py" => "Python",
        "rs" => "Rust",
        // "zig" => "Zig",
        "sh" => "Bash",
        "c" => "C",
        "cpp" => "C++",
        "cs" => "C#",
        "css" => "CSS",
        "ex" => "Elixir",
        "erl" => "Erlang",
        "go" => "Go",
        "html" => "HTML",
        "java" => "Java",
        "js" => "JavaScript",
        "json" => "JSON",
        "hs" => "Haskell",
        "lua" => "Lua",
        "ml" => "OCaml",
        _ => {
            return Err(GetParserError::NoLanguageFoundForExtension(
                extension.to_string(),
            ))
        }
    }
    .to_string())
}

pub fn get_parser_for_extension(extension: &str) -> Result<Parser, GetParserError> {
    let language = get_extension_for_language(extension)?;
    let mut parser = Parser::new();
    match language.as_str() {
        #[cfg(any(feature = "all", feature = "python"))]
        "Python" => parser.set_language(&tree_sitter_python::LANGUAGE.into())?,
        #[cfg(any(feature = "all", feature = "rust"))]
        "Rust" => parser.set_language(&tree_sitter_rust::LANGUAGE.into())?,
        // #[cfg(any(feature = "all", feature = "zig"))]
        // "Zig" => parser.set_language(&tree_sitter_zig::LANGUAGE.into())?,
        #[cfg(any(feature = "all", feature = "bash"))]
        "Bash" => parser.set_language(&tree_sitter_bash::LANGUAGE.into())?,
        #[cfg(any(feature = "all", feature = "c"))]
        "C" => parser.set_language(&tree_sitter_c::LANGUAGE.into())?,
        #[cfg(any(feature = "all", feature = "cpp"))]
        "C++" => parser.set_language(&tree_sitter_cpp::LANGUAGE.into())?,
        #[cfg(any(feature = "all", feature = "csharp"))]
        "C#" => parser.set_language(&tree_sitter_c_sharp::LANGUAGE.into())?,
        #[cfg(any(feature = "all", feature = "css"))]
        "CSS" => parser.set_language(&tree_sitter_css::LANGUAGE.into())?,
        #[cfg(any(feature = "all", feature = "elixir"))]
        "Elixir" => parser.set_language(&tree_sitter_elixir::LANGUAGE.into())?,
        #[cfg(any(feature = "all", feature = "erlang"))]
        "Erlang" => parser.set_language(&tree_sitter_erlang::LANGUAGE.into())?,
        #[cfg(any(feature = "all", feature = "go"))]
        "Go" => parser.set_language(&tree_sitter_go::LANGUAGE.into())?,
        #[cfg(any(feature = "all", feature = "html"))]
        "HTML" => parser.set_language(&tree_sitter_html::LANGUAGE.into())?,
        #[cfg(any(feature = "all", feature = "java"))]
        "Java" => parser.set_language(&tree_sitter_java::LANGUAGE.into())?,
        #[cfg(any(feature = "all", feature = "javascript"))]
        "JavaScript" => parser.set_language(&tree_sitter_javascript::LANGUAGE.into())?,
        #[cfg(any(feature = "all", feature = "json"))]
        "JSON" => parser.set_language(&tree_sitter_json::LANGUAGE.into())?,
        #[cfg(any(feature = "all", feature = "haskell"))]
        "Haskell" => parser.set_language(&tree_sitter_haskell::LANGUAGE.into())?,
        #[cfg(any(feature = "all", feature = "lua"))]
        "Lua" => parser.set_language(&tree_sitter_lua::LANGUAGE.into())?,
        #[cfg(any(feature = "all", feature = "ocaml"))]
        "OCaml" => parser.set_language(&tree_sitter_ocaml::LANGUAGE_OCAML.into())?,
        _ => {
            return Err(GetParserError::NoParserFoundForExtension(
                language.to_string(),
            ))
        }
    }
    Ok(parser)
}
