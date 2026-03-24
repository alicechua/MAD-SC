"""Utilities for persisting enriched debate-trail logs to debate_logs.json.

Each entry in the log is keyed by the target word and stores enough
information to diagnose incorrect or surprising Judge verdicts:

  - The full argument texts produced by Team Support and Team Refuse
  - The complete JudgeVerdict (verdict label, change_type, causal_driver,
    break_point_year, reasoning)
  - The exact corpus sentences that were fed to the agents
  - The word's part-of-speech tag
  - A timestamp and the active LLM backend/model for reproducibility
  - The debate mode ("single_round" or "multi_round") and round count
  - The full per-round transcript when running in multi-round mode

Usage
-----
    from mad_sc.log_utils import append_debate_log
    append_debate_log(state, log_path="debate_logs.json")
    append_debate_log(state, log_path="debate_logs.json", debate_mode="multi_round", num_rounds=3)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DEFAULT_LOG_PATH = Path(__file__).resolve().parent.parent / "debate_logs.json"


def _extract_text(content: Any) -> str:
    """Normalise an LLM response content value to a plain string.

    Handles three formats that different LangChain / SDK versions may return:
      - str                          → returned as-is
      - object with .content attr   → recurse on .content
      - list[dict] with 'text' keys → join all 'text' values (Anthropic blocks)
    """
    if isinstance(content, str):
        return content
    if hasattr(content, "content"):
        return _extract_text(content.content)
    if isinstance(content, list):
        texts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and "text" in block
        ]
        return "\n\n".join(texts)
    return str(content)


def _get_backend_info() -> dict:
    """Return the active LLM backend and resolved model name from env vars."""
    backend = os.getenv("LLM_BACKEND", "google_ai_studio").strip().lower()
    model_env_map = {
        "google_ai_studio": ("DEFAULT_MODEL_GAS", "gemini-2.5-flash"),
        "vertex_ai": ("DEFAULT_MODEL_VAI", "gemini-2.5-flash"),
        "groq": ("DEFAULT_MODEL_GROQ", "llama3-70b-8192"),
    }
    env_key, fallback = model_env_map.get(backend, ("DEFAULT_MODEL_GAS", "unknown"))
    model = os.getenv(env_key, fallback)
    return {"backend": backend, "model": model}


def append_debate_log(
    state: dict,
    log_path: str | Path | None = None,
    debate_mode: str = "single_round",
    num_rounds: int = 1,
) -> None:
    """Append (or overwrite) one word's enriched debate record in the log file.

    Parameters
    ----------
    state:
        The final ``GraphState`` dict returned by ``graph.invoke()``.
        Must contain ``word``, ``word_type``, ``arg_change``, ``arg_stable``,
        ``verdict``, ``sentences_old``, and ``sentences_new``.
    log_path:
        Path to the JSON log file.  Defaults to ``debate_logs.json`` next to
        the repository root.
    debate_mode:
        ``"single_round"`` (default) or ``"multi_round"``.
    num_rounds:
        Number of rebuttal rounds executed (meaningful only for multi-round mode).
    """
    path = Path(log_path) if log_path else _DEFAULT_LOG_PATH

    # Load existing log (or start fresh).
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            try:
                data: dict = json.load(fh)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    word: str = state["word"]
    verdict_dict: dict = state.get("verdict") or {}

    # Build the enriched record.
    record: dict = {
        # ── Debate arguments ──────────────────────────────────────────────
        "arg_change": _extract_text(state.get("arg_change", "")),
        "arg_stable": _extract_text(state.get("arg_stable", "")),
        # ── Full structured verdict ───────────────────────────────────────
        "verdict": verdict_dict.get("verdict"),
        "change_type": verdict_dict.get("change_type"),
        "causal_driver": verdict_dict.get("causal_driver"),
        "break_point_year": verdict_dict.get("break_point_year"),
        "reasoning": verdict_dict.get("reasoning"),
        # ── Input evidence ────────────────────────────────────────────────
        "word_type": state.get("word_type", ""),
        "t_old": state.get("t_old", ""),
        "t_new": state.get("t_new", ""),
        "sentences_old": state.get("sentences_old", []),
        "sentences_new": state.get("sentences_new", []),
        # ── Run metadata ──────────────────────────────────────────────────
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "debate_mode": debate_mode,
        "num_rounds": num_rounds,
        # ── Tool calls made by each team agent ────────────────────────────
        "tool_calls_support": state.get("tool_calls_support") or [],
        "tool_calls_refuse": state.get("tool_calls_refuse") or [],
        # ── Multi-round full transcript (empty list for single-round) ─────
        "debate_history": state.get("debate_history", []),
        **_get_backend_info(),
    }

    data[word] = record

    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
