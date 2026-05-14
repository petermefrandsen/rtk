//! Token reader for Claude Code session files.
//!
//! Reads `~/.claude/projects/**/*.jsonl` and sums `input + output` tokens
//! from every `"assistant"` event that carries a `message.usage` object.

use super::DailyTokens;
use anyhow::{Context, Result};
use chrono::NaiveDate;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use walkdir::WalkDir;

/// Reads token usage from Claude Code session files.
pub struct ClaudeCodeReader;

impl ClaudeCodeReader {
    pub fn read() -> Result<DailyTokens> {
        let home = match dirs::home_dir() {
            Some(h) => h,
            None => return Ok(HashMap::new()),
        };
        let projects_dir = home.join(".claude").join("projects");
        if !projects_dir.exists() {
            return Ok(HashMap::new());
        }

        let mut daily: DailyTokens = HashMap::new();
        for entry in WalkDir::new(&projects_dir)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "jsonl"))
        {
            let _ = accumulate_claude_file(entry.path(), &mut daily);
        }
        Ok(daily)
    }
}

#[derive(Deserialize)]
struct ClaudeEvent {
    #[serde(rename = "type")]
    event_type: String,
    timestamp: Option<String>,
    message: Option<ClaudeMessage>,
}

#[derive(Deserialize)]
struct ClaudeMessage {
    usage: Option<ClaudeUsage>,
}

#[derive(Deserialize)]
struct ClaudeUsage {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
}

fn accumulate_claude_file(path: &Path, daily: &mut DailyTokens) -> Result<()> {
    let content =
        fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let Ok(event) = serde_json::from_str::<ClaudeEvent>(line) else {
            continue;
        };
        if event.event_type != "assistant" {
            continue;
        }
        let Some(ts) = &event.timestamp else {
            continue;
        };
        let Some(date_str) = ts.get(..10) else {
            continue;
        };
        let Ok(date) = NaiveDate::parse_from_str(date_str, "%Y-%m-%d") else {
            continue;
        };
        if let Some(msg) = &event.message {
            if let Some(usage) = &msg.usage {
                let tokens = usage.input_tokens.unwrap_or(0) + usage.output_tokens.unwrap_or(0);
                *daily.entry(date).or_insert(0) += tokens;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_temp(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        write!(f, "{}", content).unwrap();
        f
    }

    #[test]
    fn test_claude_accumulates_input_and_output_tokens() {
        let content = r#"{"type":"assistant","timestamp":"2025-01-15T10:00:00.000Z","message":{"usage":{"input_tokens":1000,"output_tokens":200}}}
{"type":"assistant","timestamp":"2025-01-15T11:00:00.000Z","message":{"usage":{"input_tokens":500,"output_tokens":100}}}
{"type":"user","timestamp":"2025-01-15T10:05:00.000Z","message":{"content":"hello"}}
"#;
        let f = write_temp(content);
        let mut daily = DailyTokens::new();
        accumulate_claude_file(f.path(), &mut daily).unwrap();

        let date = NaiveDate::from_ymd_opt(2025, 1, 15).unwrap();
        assert_eq!(
            daily[&date], 1800,
            "input+output from two turns on same day"
        );
    }

    #[test]
    fn test_claude_skips_non_assistant_events() {
        let content = r#"{"type":"user","timestamp":"2025-01-15T10:00:00.000Z","message":{"content":"hi"}}
{"type":"system","timestamp":"2025-01-15T10:01:00.000Z"}
"#;
        let f = write_temp(content);
        let mut daily = DailyTokens::new();
        accumulate_claude_file(f.path(), &mut daily).unwrap();
        assert!(daily.is_empty(), "non-assistant events must be ignored");
    }

    #[test]
    fn test_claude_handles_missing_usage_gracefully() {
        let content = r#"{"type":"assistant","timestamp":"2025-01-15T10:00:00.000Z","message":{}}"#;
        let f = write_temp(content);
        let mut daily = DailyTokens::new();
        accumulate_claude_file(f.path(), &mut daily).unwrap();
        assert!(daily.is_empty());
    }

    #[test]
    fn test_claude_accumulates_across_days() {
        let content = r#"{"type":"assistant","timestamp":"2025-01-14T10:00:00.000Z","message":{"usage":{"input_tokens":100,"output_tokens":50}}}
{"type":"assistant","timestamp":"2025-01-15T10:00:00.000Z","message":{"usage":{"input_tokens":200,"output_tokens":80}}}
"#;
        let f = write_temp(content);
        let mut daily = DailyTokens::new();
        accumulate_claude_file(f.path(), &mut daily).unwrap();

        let d14 = NaiveDate::from_ymd_opt(2025, 1, 14).unwrap();
        let d15 = NaiveDate::from_ymd_opt(2025, 1, 15).unwrap();
        assert_eq!(daily[&d14], 150);
        assert_eq!(daily[&d15], 280);
    }

    #[test]
    fn test_claude_skips_malformed_lines() {
        let content = r#"{"type":"assistant","timestamp":"2025-01-15T10:00:00.000Z","message":{"usage":{"input_tokens":100,"output_tokens":50}}}
{ broken json %%
"#;
        let f = write_temp(content);
        let mut daily = DailyTokens::new();
        accumulate_claude_file(f.path(), &mut daily).unwrap();

        let date = NaiveDate::from_ymd_opt(2025, 1, 15).unwrap();
        assert_eq!(
            daily[&date], 150,
            "valid line still parsed despite malformed neighbour"
        );
    }
}
