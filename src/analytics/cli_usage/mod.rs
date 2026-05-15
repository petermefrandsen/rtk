//! Native readers for real LLM token usage from agent session files.
//!
//! Reads token usage from local session files written by AI coding agents.
//! Each reader returns per-day accumulated token counts and degrades
//! gracefully when the agent is not installed (returns an empty map).
//!
//! ## Token counting rationale per agent
//!
//! - **Claude Code**: `input + output` tokens per turn — both sides of the exchange
//! - **Copilot**: `outputTokens` only — input is absent from `assistant.message` events
//! - **Gemini CLI**: `output + thoughts` only — `input` grows cumulatively with full
//!   context history each turn, causing ~100–300× overcounting if included
//!
//! This module is a standalone reader library consumed by `gain` (PR 2) and
//! `cc_economics` (PR 3); the public API is unused until those integrations land.
#![allow(dead_code)]

use chrono::NaiveDate;
use std::collections::HashMap;

pub mod claude_code;
pub mod copilot;
pub mod gemini;

pub use claude_code::ClaudeCodeReader;
pub use copilot::CopilotReader;
pub use gemini::GeminiReader;

// ── Public types ──────────────────────────────────────────────────────────────

/// Per-day accumulated token count for one agent.
pub type DailyTokens = HashMap<NaiveDate, u64>;

pub const AGENT_CLAUDE_CODE: &str = "claude_code";
pub const AGENT_COPILOT: &str = "copilot";
pub const AGENT_GEMINI: &str = "gemini";

// ── Aggregated reader ─────────────────────────────────────────────────────────

/// Read token usage from all locally-installed agents.
///
/// Agents without session files (not installed) are silently omitted.
/// Returns an empty map when no agents are found.
pub fn read_all_agents() -> HashMap<String, DailyTokens> {
    let mut result = HashMap::new();
    for (name, tokens) in [
        (AGENT_CLAUDE_CODE, ClaudeCodeReader::read()),
        (AGENT_COPILOT, CopilotReader::read()),
        (AGENT_GEMINI, GeminiReader::read()),
    ] {
        if let Ok(daily) = tokens {
            if !daily.is_empty() {
                result.insert(name.to_string(), daily);
            }
        }
    }
    result
}
