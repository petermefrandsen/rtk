//! Token reader for GitHub Copilot session event files.
//!
//! Reads `~/.copilot/session-state/*/events.jsonl` and sums `data.outputTokens`
//! from every `"assistant.message"` event. Input tokens are not present in
//! Copilot's event format.

use super::DailyTokens;
use anyhow::{Context, Result};
use chrono::NaiveDate;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Reads token usage from GitHub Copilot session event files.
///
/// Only output tokens are tracked — Copilot's `events.jsonl` does not
/// expose input tokens in `assistant.message` events.
pub struct CopilotReader;

impl CopilotReader {
    pub fn read() -> Result<DailyTokens> {
        let home = match dirs::home_dir() {
            Some(h) => h,
            None => return Ok(HashMap::new()),
        };
        let sessions_dir = home.join(".copilot").join("session-state");
        if !sessions_dir.exists() {
            return Ok(HashMap::new());
        }

        let mut daily: DailyTokens = HashMap::new();
        for entry in fs::read_dir(&sessions_dir)
            .with_context(|| format!("reading {}", sessions_dir.display()))?
        {
            let events_file = entry?.path().join("events.jsonl");
            if events_file.exists() {
                let _ = accumulate_copilot_file(&events_file, &mut daily);
            }
        }
        Ok(daily)
    }
}

#[derive(Deserialize)]
struct CopilotEvent {
    #[serde(rename = "type")]
    event_type: String,
    timestamp: Option<String>,
    data: Option<CopilotMessageData>,
}

#[derive(Deserialize)]
struct CopilotMessageData {
    #[serde(rename = "outputTokens")]
    output_tokens: Option<u64>,
}

fn accumulate_copilot_file(path: &Path, daily: &mut DailyTokens) -> Result<()> {
    let content =
        fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let Ok(event) = serde_json::from_str::<CopilotEvent>(line) else {
            continue;
        };
        if event.event_type != "assistant.message" {
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
        if let Some(data) = &event.data {
            *daily.entry(date).or_insert(0) += data.output_tokens.unwrap_or(0);
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
    fn test_copilot_accumulates_output_tokens() {
        let content = r#"{"type":"session.start","timestamp":"2026-03-14T20:28:36.012Z"}
{"type":"assistant.message","timestamp":"2026-03-14T20:33:12.062Z","data":{"outputTokens":856}}
{"type":"assistant.message","timestamp":"2026-03-14T20:33:26.921Z","data":{"outputTokens":820}}
"#;
        let f = write_temp(content);
        let mut daily = DailyTokens::new();
        accumulate_copilot_file(f.path(), &mut daily).unwrap();

        let date = NaiveDate::from_ymd_opt(2026, 3, 14).unwrap();
        assert_eq!(daily[&date], 1676, "two assistant.message events summed");
    }

    #[test]
    fn test_copilot_ignores_non_message_events() {
        let content = r#"{"type":"session.start","timestamp":"2026-03-14T20:28:36.012Z"}
{"type":"assistant.turn_end","timestamp":"2026-03-14T20:34:00.000Z","turnId":"abc"}
{"type":"user.message","timestamp":"2026-03-14T20:30:00.000Z","data":{"content":"hi"}}
"#;
        let f = write_temp(content);
        let mut daily = DailyTokens::new();
        accumulate_copilot_file(f.path(), &mut daily).unwrap();
        assert!(
            daily.is_empty(),
            "only assistant.message events carry token data"
        );
    }

    #[test]
    fn test_copilot_handles_missing_output_tokens() {
        // Events where `data` exists but `outputTokens` is absent (e.g. streaming chunks)
        let content =
            r#"{"type":"assistant.message","timestamp":"2026-03-14T20:33:00.000Z","data":{}}"#;
        let f = write_temp(content);
        let mut daily = DailyTokens::new();
        accumulate_copilot_file(f.path(), &mut daily).unwrap();

        let date = NaiveDate::from_ymd_opt(2026, 3, 14).unwrap();
        assert_eq!(daily[&date], 0, "missing outputTokens defaults to 0");
    }

    #[test]
    fn test_copilot_accumulates_across_sessions_same_day() {
        // Simulates two separate session files both writing to the same daily bucket
        let content_a = r#"{"type":"assistant.message","timestamp":"2026-03-14T09:00:00.000Z","data":{"outputTokens":400}}"#;
        let content_b = r#"{"type":"assistant.message","timestamp":"2026-03-14T15:00:00.000Z","data":{"outputTokens":600}}"#;

        let f_a = write_temp(content_a);
        let f_b = write_temp(content_b);
        let mut daily = DailyTokens::new();
        accumulate_copilot_file(f_a.path(), &mut daily).unwrap();
        accumulate_copilot_file(f_b.path(), &mut daily).unwrap();

        let date = NaiveDate::from_ymd_opt(2026, 3, 14).unwrap();
        assert_eq!(daily[&date], 1000);
    }
}
