//! Token reader for Gemini CLI session files.
//!
//! Sums `tokens.output + tokens.thoughts` per turn only. The `tokens.input`
//! field stores the full cumulative context re-sent every turn, so summing it
//! would overcount total tokens consumed by 100–300×.
//!
//! Handles three path layouts produced by different Gemini CLI versions:
//! - `~/.gemini/tmp/<project>/chats/session-*.json`  (single JSON object)
//! - `~/.gemini/tmp/<project>/chats/session-*.jsonl` (line-delimited)
//! - `~/.gemini/tmp/<project>/chats/<uuid>/<shortid>.json` (nested UUID dir)

use super::DailyTokens;
use anyhow::{Context, Result};
use chrono::NaiveDate;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use walkdir::WalkDir;

/// Reads token usage from Gemini CLI session files.
pub struct GeminiReader;

impl GeminiReader {
    pub fn read() -> Result<DailyTokens> {
        let home = match dirs::home_dir() {
            Some(h) => h,
            None => return Ok(HashMap::new()),
        };
        let tmp_dir = home.join(".gemini").join("tmp");
        if !tmp_dir.exists() {
            return Ok(HashMap::new());
        }

        let mut daily: DailyTokens = HashMap::new();
        for entry in WalkDir::new(&tmp_dir)
            .max_depth(4)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file() && is_gemini_session_file(e.path()))
        {
            let _ = accumulate_gemini_file(entry.path(), &mut daily);
        }
        Ok(daily)
    }
}

/// Returns true for Gemini CLI session files in both known path layouts:
/// - `~/.gemini/tmp/<project>/chats/session-*.{json,jsonl}` (flat format)
/// - `~/.gemini/tmp/<project>/chats/<uuid>/<shortid>.json` (nested format)
fn is_gemini_session_file(path: &Path) -> bool {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    if ext != "json" && ext != "jsonl" {
        return false;
    }
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    if name.starts_with("session-") {
        return true;
    }
    // Nested format: any .json inside a UUID-named directory
    path.parent()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .map(|parent| parent.len() == 36 && parent.chars().filter(|&c| c == '-').count() == 4)
        .unwrap_or(false)
}

#[derive(Deserialize)]
struct GeminiSessionJson {
    messages: Vec<GeminiMessage>,
}

#[derive(Deserialize)]
struct GeminiMessage {
    #[serde(rename = "type")]
    msg_type: String,
    timestamp: Option<String>,
    tokens: Option<GeminiTokens>,
}

#[derive(Deserialize)]
struct GeminiTokens {
    output: u64,
    thoughts: u64,
}

fn accumulate_gemini_file(path: &Path, daily: &mut DailyTokens) -> Result<()> {
    let content =
        fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;

    // Try the older `.json` format first: a single object with a `messages` array.
    if path.extension().is_some_and(|ext| ext == "json") {
        if let Ok(session) = serde_json::from_str::<GeminiSessionJson>(&content) {
            accumulate_gemini_messages(session.messages.iter(), daily);
            return Ok(());
        }
    }

    // Fall back to line-delimited format (`.jsonl` and newer `.json` variants).
    let messages: Vec<GeminiMessage> = content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    accumulate_gemini_messages(messages.iter(), daily);
    Ok(())
}

fn accumulate_gemini_messages<'a>(
    messages: impl Iterator<Item = &'a GeminiMessage>,
    daily: &mut DailyTokens,
) {
    for msg in messages {
        if msg.msg_type != "gemini" {
            continue;
        }
        let Some(ts) = &msg.timestamp else {
            continue;
        };
        let Some(date_str) = ts.get(..10) else {
            continue;
        };
        let Ok(date) = NaiveDate::parse_from_str(date_str, "%Y-%m-%d") else {
            continue;
        };
        if let Some(tokens) = &msg.tokens {
            *daily.entry(date).or_insert(0) += tokens.output + tokens.thoughts;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn write_temp_with_ext(content: &str, ext: &str) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(format!("session.{}", ext));
        fs::write(&path, content).unwrap();
        (dir, path)
    }

    #[test]
    fn test_gemini_json_sums_output_and_thoughts_only() {
        let content = r#"{
  "sessionId": "test-fixture",
  "startTime": "2026-04-13T18:37:28.144Z",
  "messages": [
    {
      "id": "msg-1",
      "timestamp": "2026-04-13T18:38:06.616Z",
      "type": "gemini",
      "content": "",
      "tokens": {"input": 15698, "output": 38, "cached": 0, "thoughts": 200, "tool": 0, "total": 15936}
    },
    {
      "id": "msg-2",
      "timestamp": "2026-04-13T18:38:10.052Z",
      "type": "gemini",
      "content": "",
      "tokens": {"input": 16255, "output": 61, "cached": 15071, "thoughts": 142, "tool": 0, "total": 16458}
    }
  ]
}"#;
        let (_dir, path) = write_temp_with_ext(content, "json");
        let mut daily = DailyTokens::new();
        accumulate_gemini_file(&path, &mut daily).unwrap();

        let date = NaiveDate::from_ymd_opt(2026, 4, 13).unwrap();
        // (38+200) + (61+142) = 238 + 203 = 441 — input NOT included
        assert_eq!(daily[&date], 441, "input tokens must be excluded");
    }

    #[test]
    fn test_gemini_jsonl_parses_line_delimited_format() {
        // First line: session metadata (no `type` key — parsed as unknown)
        // Subsequent lines: individual message events
        let content = r#"{"sessionId":"s1","startTime":"2026-04-21T06:52:00.000Z","kind":"session","summary":""}
{"id":"ev-1","timestamp":"2026-04-21T06:57:53.770Z","type":"gemini","content":"","tokens":{"input":19085,"output":10,"cached":0,"thoughts":342,"tool":0,"total":19437}}
{"id":"ev-2","timestamp":"2026-04-21T06:58:10.000Z","type":"user","content":"more"}
{"id":"ev-3","timestamp":"2026-04-21T06:59:00.000Z","type":"gemini","content":"","tokens":{"input":20000,"output":25,"cached":0,"thoughts":100,"tool":0,"total":20125}}
"#;
        let (_dir, path) = write_temp_with_ext(content, "jsonl");
        let mut daily = DailyTokens::new();
        accumulate_gemini_file(&path, &mut daily).unwrap();

        let date = NaiveDate::from_ymd_opt(2026, 4, 21).unwrap();
        // (10+342) + (25+100) = 352 + 125 = 477
        assert_eq!(daily[&date], 477);
    }

    #[test]
    fn test_gemini_excludes_non_gemini_message_types() {
        let content = r#"{
  "sessionId": "test-fixture",
  "startTime": "2026-04-13T18:37:28.144Z",
  "messages": [
    {"id": "u1", "timestamp": "2026-04-13T18:38:00.000Z", "type": "user", "content": "hello"},
    {"id": "i1", "timestamp": "2026-04-13T18:38:01.000Z", "type": "info", "content": "tool call"},
    {"id": "g1", "timestamp": "2026-04-13T18:38:06.000Z", "type": "gemini", "content": "",
     "tokens": {"input": 5000, "output": 50, "cached": 0, "thoughts": 30, "tool": 0, "total": 5080}}
  ]
}"#;
        let (_dir, path) = write_temp_with_ext(content, "json");
        let mut daily = DailyTokens::new();
        accumulate_gemini_file(&path, &mut daily).unwrap();

        let date = NaiveDate::from_ymd_opt(2026, 4, 13).unwrap();
        assert_eq!(daily[&date], 80, "only gemini-type messages counted");
    }

    #[test]
    fn test_gemini_accumulates_across_days() {
        let content = r#"{
  "sessionId": "multi-day",
  "startTime": "2026-04-12T08:00:00.000Z",
  "messages": [
    {"id":"m1","timestamp":"2026-04-12T08:01:00.000Z","type":"gemini","content":"",
     "tokens":{"input":1000,"output":40,"cached":0,"thoughts":60,"tool":0,"total":1100}},
    {"id":"m2","timestamp":"2026-04-13T09:00:00.000Z","type":"gemini","content":"",
     "tokens":{"input":2000,"output":80,"cached":0,"thoughts":20,"tool":0,"total":2100}}
  ]
}"#;
        let (_dir, path) = write_temp_with_ext(content, "json");
        let mut daily = DailyTokens::new();
        accumulate_gemini_file(&path, &mut daily).unwrap();

        let d12 = NaiveDate::from_ymd_opt(2026, 4, 12).unwrap();
        let d13 = NaiveDate::from_ymd_opt(2026, 4, 13).unwrap();
        assert_eq!(daily[&d12], 100); // 40+60
        assert_eq!(daily[&d13], 100); // 80+20
    }

    #[test]
    fn test_gemini_skips_malformed_lines_in_jsonl() {
        let content = r#"{"id":"ev-1","timestamp":"2026-04-21T06:57:00.000Z","type":"gemini","content":"","tokens":{"input":100,"output":10,"cached":0,"thoughts":5,"tool":0,"total":115}}
not valid json at all %%%
{"id":"ev-3","timestamp":"2026-04-21T06:58:00.000Z","type":"gemini","content":"","tokens":{"input":200,"output":20,"cached":0,"thoughts":10,"tool":0,"total":230}}
"#;
        let (_dir, path) = write_temp_with_ext(content, "jsonl");
        let mut daily = DailyTokens::new();
        accumulate_gemini_file(&path, &mut daily).unwrap();

        let date = NaiveDate::from_ymd_opt(2026, 4, 21).unwrap();
        // (10+5) + (20+10) = 15 + 30 = 45
        assert_eq!(
            daily[&date], 45,
            "valid lines parsed despite malformed neighbour"
        );
    }

    // ── is_gemini_session_file ─────────────────────────────────────────────────

    #[test]
    fn test_session_file_detection_flat_format() {
        let path =
            PathBuf::from("/home/user/.gemini/tmp/proj/chats/session-2026-04-21T06-52-56.json");
        assert!(is_gemini_session_file(&path));

        let jsonl_path =
            PathBuf::from("/home/user/.gemini/tmp/proj/chats/session-2026-04-21.jsonl");
        assert!(is_gemini_session_file(&jsonl_path));
    }

    #[test]
    fn test_session_file_detection_nested_uuid_format() {
        let path = PathBuf::from(
            "/home/user/.gemini/tmp/proj/chats/42b19f97-ec32-4cbb-8dea-583ade9f6f6d/z18mgh.json",
        );
        assert!(is_gemini_session_file(&path));
    }

    #[test]
    fn test_session_file_detection_rejects_unrelated_files() {
        assert!(!is_gemini_session_file(&PathBuf::from(
            "/home/user/.gemini/tmp/proj/config.json"
        )));
        assert!(!is_gemini_session_file(&PathBuf::from(
            "/home/user/.gemini/settings.yaml"
        )));
        assert!(!is_gemini_session_file(&PathBuf::from(
            "/home/user/.gemini/tmp/proj/chats/42b19f97-ec32-4cbb-8dea-583ade9f6f6d/z18mgh.txt"
        )));
    }
}
