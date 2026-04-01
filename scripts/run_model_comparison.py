#!/usr/bin/env python3
"""
Multi-model smoke test comparison for MAD-SC.

Runs evaluate_lsc.py as a fresh subprocess per model configuration to avoid
import-time env var caching in mad_sc/nodes.py (_JUDGE_MODEL_* are captured
at module import, so patching os.environ after import has no effect).

Usage:
    python scripts/run_model_comparison.py            # run all models sequentially
    python scripts/run_model_comparison.py --dry-run  # print commands, don't execute
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATE_LSC = PROJECT_ROOT / "scripts" / "evaluate_lsc.py"

_venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
PYTHON = str(_venv_python) if _venv_python.exists() else sys.executable

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
SMOKE_WORDS = ["mouse", "corn", "horn", "canine", "fast"]

GROUND_TRUTH = {
    "mouse":  "Metaphor",
    "corn":   "Specialization",
    "horn":   "Metonymy",
    "canine": "Ellipsis",
    "fast":   "Analogy",
}


@dataclass
class ModelConfig:
    short_name: str
    backend: str             # "google_ai_studio" | "nebius"
    debate_model: str
    judge_model: str
    delay: float             # seconds between words (--delay)
    inter_call_delay: float  # seconds between API calls (INTER_CALL_DELAY env)
    timeout: int = 1800      # subprocess timeout in seconds
    skip: bool = False       # True = read existing result without re-running


MODELS: list[ModelConfig] = [
    # ── Historical reference only (used lexicographer + old prompt) ────────
    ModelConfig(
        short_name="gemini_lite_with_lex",
        backend="google_ai_studio",
        debate_model="gemini-3.1-flash-lite-preview",
        judge_model="gemini-3.1-flash-lite-preview",
        delay=5.0,
        inter_call_delay=2.0,
        skip=True,
    ),
    # ── GAS: gemini-flash-lite, no-lex, revised prompt (fair baseline) ────
    ModelConfig(
        short_name="gemini_flash_lite_nolex",
        backend="google_ai_studio",
        debate_model="gemini-3.1-flash-lite-preview",
        judge_model="gemini-3.1-flash-lite-preview",
        delay=5.0,
        inter_call_delay=2.0,
        timeout=1200,
    ),
    # ── Google AI Studio: stronger model (skipped — rate limits) ──────────
    ModelConfig(
        short_name="gemini_25_flash",
        backend="google_ai_studio",
        debate_model="gemini-2.5-flash",
        judge_model="gemini-2.5-flash",
        delay=15.0,
        inter_call_delay=5.0,
        timeout=2400,
        skip=True,
    ),
    # ── Nebius: DeepSeek-V3 ────────────────────────────────────────────────
    ModelConfig(
        short_name="deepseek_v3",
        backend="nebius",
        debate_model="deepseek-ai/DeepSeek-V3-0324",
        judge_model="deepseek-ai/DeepSeek-V3-0324",
        delay=3.0,
        inter_call_delay=2.0,
        timeout=1200,   # ~113s/word observed × 5 words = ~565s; 1200s is comfortable
    ),
    # ── Nebius: DeepSeek-R1 (reasoning model) ─────────────────────────────
    ModelConfig(
        short_name="deepseek_r1",
        backend="nebius",
        debate_model="deepseek-ai/DeepSeek-R1-0528",
        judge_model="deepseek-ai/DeepSeek-R1-0528",
        delay=5.0,
        inter_call_delay=3.0,
        timeout=3600,   # reasoning model generates long CoT; allow 60 min
    ),
    # ── Nebius: Nemotron-Ultra-253B ────────────────────────────────────────
    ModelConfig(
        short_name="nemotron_253b",
        backend="nebius",
        debate_model="nvidia/Llama-3_1-Nemotron-Ultra-253B-v1",
        judge_model="nvidia/Llama-3_1-Nemotron-Ultra-253B-v1",
        delay=5.0,
        inter_call_delay=3.0,
        timeout=1800,   # ~174s/word observed × 5 words = ~870s; 1800s is safe
    ),
    # ── Nebius: Qwen3-235B MoE ─────────────────────────────────────────────
    ModelConfig(
        short_name="qwen3_235b",
        backend="nebius",
        debate_model="Qwen/Qwen3-235B-A22B-Instruct-2507",
        judge_model="Qwen/Qwen3-235B-A22B-Instruct-2507",
        delay=5.0,
        inter_call_delay=3.0,
    ),
]

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    config: ModelConfig
    output_dir: Path
    status: str                          # "ok" | "failed" | "timeout" | "missing" | "corrupt" | "reference"
    fine_acc: Optional[float] = None
    coarse_acc: Optional[float] = None
    per_word: list[dict] = field(default_factory=list)
    returncode: Optional[int] = None
    elapsed: Optional[float] = None      # wall-clock seconds for the subprocess


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_output_dir(config: ModelConfig) -> Path:
    if config.short_name == "gemini_lite_with_lex":
        return PROJECT_ROOT / "evals" / "eval_results_multiround_closing_smoke_v2"
    return PROJECT_ROOT / "evals" / f"smoke_{config.short_name}"


def build_env(config: ModelConfig) -> dict[str, str]:
    """Build child process env with model-specific overrides."""
    env = os.environ.copy()
    env["LLM_BACKEND"] = config.backend
    env["INTER_CALL_DELAY"] = str(config.inter_call_delay)

    # Clear judge/lex vars that might bleed across backends
    for key in ("JUDGE_MODEL_OR", "LEXICOGRAPHER_MODEL_NEBIUS"):
        env[key] = ""

    if config.backend == "google_ai_studio":
        env["DEFAULT_MODEL_GAS"] = config.debate_model
        env["JUDGE_MODEL_GAS"] = config.judge_model
        env["NEBIUS_MODEL"] = ""
        env["JUDGE_MODEL_NEBIUS"] = ""
    elif config.backend == "nebius":
        env["NEBIUS_MODEL"] = config.debate_model
        env["JUDGE_MODEL_NEBIUS"] = config.judge_model
        env["DEFAULT_MODEL_GAS"] = ""
        env["JUDGE_MODEL_GAS"] = ""

    return env


def build_cmd(config: ModelConfig, output_dir: Path, resume: bool = False) -> list[str]:
    cmd = [
        PYTHON, str(EVALUATE_LSC),
        "--words", *SMOKE_WORDS,
        "--output-dir", str(output_dir),
        "--mode", "multi",
        "--rounds", "3",
        "--no-grounding",
        "--no-lexicographer",
        "--delay", str(config.delay),
    ]
    if resume:
        cmd.append("--resume")
    return cmd


def read_summary(output_dir: Path, config: ModelConfig) -> RunResult:
    summary_path = output_dir / "eval_summary.json"
    if not summary_path.exists():
        return RunResult(config=config, output_dir=output_dir, status="missing")
    try:
        with open(summary_path) as f:
            data = json.load(f)
        return RunResult(
            config=config,
            output_dir=output_dir,
            status="ok",
            fine_acc=data["fine_grained_metrics"]["accuracy"],
            coarse_acc=data["coarse_grained_metrics"]["accuracy"],
            per_word=data.get("per_word", []),
        )
    except (KeyError, json.JSONDecodeError):
        return RunResult(config=config, output_dir=output_dir, status="corrupt")


def run_model(config: ModelConfig, dry_run: bool, resume: bool = False) -> RunResult:
    output_dir = get_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_cmd(config, output_dir, resume=resume)
    env = build_env(config)

    # Show what we're about to run
    overrides = {k: v for k, v in env.items()
                 if k in ("LLM_BACKEND", "NEBIUS_MODEL", "DEFAULT_MODEL_GAS",
                          "JUDGE_MODEL_NEBIUS", "JUDGE_MODEL_GAS", "INTER_CALL_DELAY")
                 and v}
    print(f"\n{'='*65}")
    print(f"Model: {config.short_name}  ({config.backend})")
    print(f"  debate={config.debate_model}")
    print(f"  judge ={config.judge_model}")
    print(f"  output={output_dir.relative_to(PROJECT_ROOT)}")
    print(f"  cmd   ={' '.join(cmd[2:])}")  # skip python + script path
    for k, v in overrides.items():
        print(f"  env   {k}={v}")
    print(f"{'='*65}")

    if dry_run:
        print("[DRY RUN] Skipping execution.")
        return read_summary(output_dir, config)

    import time
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            cwd=str(PROJECT_ROOT),
            timeout=config.timeout,
        )
        elapsed = time.monotonic() - t0
        result = read_summary(output_dir, config)
        result.returncode = proc.returncode
        result.elapsed = elapsed
        if proc.returncode != 0:
            print(f"[WARN] {config.short_name} exited with code {proc.returncode}")
            if result.status != "ok":
                result.status = "failed"
        return result
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        print(f"[TIMEOUT] {config.short_name} exceeded {config.timeout}s — skipping.")
        r = RunResult(config=config, output_dir=output_dir, status="timeout")
        r.elapsed = elapsed
        return r


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def _cell(word: str, per_word: list[dict]) -> str:
    """Return 'Label✓' / 'Label✗' / 'STABLE✗' / '—' for a word."""
    row = next((r for r in per_word if r.get("word") == word), None)
    if row is None:
        return "—"
    predicted = row.get("predicted") or "STABLE"
    ok = "✓" if row.get("correct_fine") else "✗"
    label = predicted[:9]  # truncate for column width
    return f"{label}{ok}"


def print_comparison_table(results: list[RunResult]) -> None:
    col_word = 11

    name_w = max(len(r.config.short_name) for r in results) + 1
    backend_w = 12

    header_words = "  ".join(w.ljust(col_word) for w in SMOKE_WORDS)
    gt_cells = "  ".join(GROUND_TRUTH[w][:9].ljust(col_word) for w in SMOKE_WORDS)

    sep = "-" * (name_w + backend_w + 8 + 8 + 10 + 10 + len(header_words) + 4)

    print("\n")
    print("=" * len(sep))
    print("Model Comparison — Smoke Test (5 words, multi/3 rounds, no-grounding, no-lexicographer)")
    print("=" * len(sep))
    print(
        f"{'Model':<{name_w}} {'Backend':<{backend_w}} {'Fine':>5}  {'Coarse':>6}  {'Total':>7}  {'Avg/word':>8}  {header_words}"
    )
    print(
        f"{'(ground truth)':<{name_w}} {'':<{backend_w}} {'':>5}  {'':>6}  {'':>7}  {'':>8}  {gt_cells}"
    )
    print(sep)

    for r in results:
        fine = f"{r.fine_acc:.0%}" if r.fine_acc is not None else "—"
        coarse = f"{r.coarse_acc:.0%}" if r.coarse_acc is not None else "—"
        n_done = len(r.per_word)
        if r.elapsed is not None:
            total = f"{r.elapsed/60:.1f}m"
            avg = f"{r.elapsed/n_done:.0f}s" if n_done else "—"
        else:
            total = avg = "—"
        word_cells = "  ".join(_cell(w, r.per_word).ljust(col_word) for w in SMOKE_WORDS)
        tag = " [REF]" if r.status == "reference" else (f" [{r.status.upper()}]" if r.status not in ("ok", "reference") else "")
        print(
            f"{r.config.short_name:<{name_w}} {r.config.backend:<{backend_w}} {fine:>5}  {coarse:>6}  {total:>7}  {avg:>8}  {word_cells}{tag}"
        )

    print(sep)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-model smoke test comparison for MAD-SC.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands and env overrides without executing.")
    parser.add_argument("--resume", action="store_true",
                        help="Pass --resume to evaluate_lsc.py — skip words with existing traces.")
    args = parser.parse_args()

    results: list[RunResult] = []

    for config in MODELS:
        output_dir = get_output_dir(config)

        if config.skip:
            print(f"\n[SKIP] {config.short_name} — reading reference from {output_dir.relative_to(PROJECT_ROOT)}")
            result = read_summary(output_dir, config)
            result.status = "reference"
            results.append(result)
            continue

        result = run_model(config, dry_run=args.dry_run, resume=args.resume)
        results.append(result)

    print_comparison_table(results)


if __name__ == "__main__":
    main()
