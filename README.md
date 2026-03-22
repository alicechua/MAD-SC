# MAD-SC: Multi-Agent Debate for Semantic Change

A LangGraph-based agentic pipeline that classifies **diachronic semantic change** using an adversarial multi-agent debate architecture. Corpus evidence comes from SemEval-2020 Task 1 (COHA) for benchmark words, with automatic fallback to Oxford English Dictionary (OED) quotations for arbitrary English words.

*Course project — CSC2611: Computational Models of Semantic Change, University of Toronto.*

---

## Overview

Static embedding methods quantify *that* a word shifted without explaining *how* or *why*. MAD-SC addresses this by framing change classification as a structured debate backed by historical corpus evidence:

1. **Lexicographer Agent** *(optional)* — consults OED dated quotations and Wiktionary etymology to produce a **Definition Dossier**: historical sense, modern sense, estimated shift year, and mechanism. Anchors both debate teams to etymological ground truth before the debate begins.
2. **Team Support** — argues that genuine semantic change occurred. Mines corpus evidence for divergent usages and proposes a Change Type and Causal Driver.
3. **Team Refuse** — defends the null hypothesis of semantic stability. Retrieves examples showing the core meaning persists and frames new usages as situational polysemy.
4. **LLM Judge** — weighs the comparative evidence from both teams and renders a structured JSON verdict.

The pipeline validates quantitative embedding signals with granular, auditable natural-language reasoning grounded in real corpus evidence.

---

## Architecture

The graph topology adapts based on which optional pre-debate nodes are enabled:

```
No flags:

    START ──► team_support ──┐
          │                   ▼
          └──► team_refuse ──► judge ──► END

With Lexicographer Agent (--lexicographer):

    START ──► lexicographer ──► team_support ──┐
                            │                   ▼
                            └──► team_refuse ──► judge ──► END

With Grounding (--grounding):

    START ──► grounding ──► team_support ──┐
                        │                   ▼
                        └──► team_refuse ──► judge ──► END

With both:

    START ──► grounding ──► lexicographer ──► team_support ──┐
                                          │                   ▼
                                          └──► team_refuse ──► judge ──► END
```

Team Support and Team Refuse always run in parallel (fan-out). The Judge fires only after both complete (fan-in).

### Node Summary

| Node | Role | State keys written |
|---|---|---|
| `grounding` *(optional)* | Runs BERT-based SED/TD analysis; injects quantitative signal into agent prompts | `grounding_block` |
| `lexicographer` *(optional)* | Queries OED + Wiktionary; synthesises Definition Dossier via chain-of-thought | `lexicographer_dossier` |
| `team_support` | Argues semantic change occurred; cites corpus evidence | `arg_change` |
| `team_refuse` | Argues semantic stability; argues new senses are polysemy | `arg_stable` |
| `judge` | Renders structured verdict from both arguments | `verdict` |

### Lexicographer Agent — OED Data Pipeline

The Lexicographer Agent fetches real corpus evidence before synthesising definitions:

1. **OED (primary)** — Playwright headless Chromium loads the OED entry page (JS-rendered), parses `li.quotation` elements, and extracts up to 6 historical (pre-1900) and 6 modern (1900+) dated quotations. Requires a valid UofT library session cookie at `scripts/oed_cookie.json`.
2. **Wiktionary (fallback)** — If OED is unavailable (no cookie, expired session), fetches the Etymology section via the Wiktionary REST API (no auth).
3. **Parametric only** — If both external sources fail, the LLM reasons from training knowledge.

The LLM uses a **chain-of-thought scratchpad** (`synthesis_reasoning`): it labels each quote individually before committing to the definitions, following Blank's taxonomy decision guide for mechanism selection.

### Taxonomy

The pipeline enforces constrained decoding over a fixed taxonomy. The Change Type uses Blank's (1999) full mechanism inventory:

| Dimension | Values |
|---|---|
| **Verdict** | `CHANGE DETECTED` \| `STABLE` |
| **Change Type** | `Metaphor` \| `Metonymy` \| `Analogy` \| `Generalization` \| `Specialization` \| `Ellipsis` \| `Antiphrasis` \| `Auto-Antonym` \| `Synecdoche` |
| **Causal Driver** | `Cultural Shift` \| `Linguistic Drift` |

### Judge Output Schema (JSON)

```json
{
  "word": "edge_nn",
  "verdict": "CHANGE DETECTED",
  "change_type": "Generalization",
  "causal_driver": "Cultural Shift",
  "break_point_year": 1990,
  "reasoning": "..."
}
```

---

## Installation

```bash
git clone <repo-url>
cd MAD-SC

python3.12 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# Install Playwright's headless Chromium (required for OED scraping)
playwright install chromium
```

---

## Configuration

Create a `.env` file at the repo root:

```bash
# ── LLM Backend ──────────────────────────────────────────────────────────────
LLM_BACKEND=openrouter          # "openrouter" (default) or "google_ai_studio"

# ── OpenRouter (default) ─────────────────────────────────────────────────────
# Obtain at: https://openrouter.ai/keys
OPENROUTER_API_KEY=sk-or-...
DEFAULT_MODEL_OR=google/gemini-2.5-flash

# ── Google AI Studio (alternative) ───────────────────────────────────────────
# Obtain at: https://aistudio.google.com/apikey
GOOGLE_AI_STUDIO_KEY=AIza...
DEFAULT_MODEL_GAS=gemini-2.5-flash

# ── Optional: use a stronger model for the Judge only ────────────────────────
# JUDGE_MODEL_OR=google/gemini-2.5-pro
# JUDGE_MODEL_GAS=gemini-2.5-pro

# ── Optional: enable pre-debate nodes by default ─────────────────────────────
USE_GROUNDING=false
USE_LEXICOGRAPHER=true

# ── Optional: rate-limit delay between LLM calls (seconds) ───────────────────
INTER_CALL_DELAY=2.0
```

Switch backends by changing `LLM_BACKEND`. Model overrides are per-backend via `DEFAULT_MODEL_OR` / `DEFAULT_MODEL_GAS`. The Judge can independently use a stronger reasoning model via `JUDGE_MODEL_OR` / `JUDGE_MODEL_GAS`.

---

## Data Setup

### SemEval-2020 Task 1 (benchmark words)

MAD-SC uses the **SemEval-2020 Task 1 (English)** dataset for the 37 expert-annotated benchmark words.

| Corpus | Time period |
|---|---|
| Corpus 1 | 1810–1860 (CCOHA) |
| Corpus 2 | 1960–2010 (CCOHA) |

Download and unzip `semeval2020_ulscd_eng.zip` from the [IMS Stuttgart resource page](https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd-eng/), then place the unzipped folder at:

```
data/semeval2020_ulscd_eng/
├── corpus1/
│   ├── lemma/ccoha1.txt.gz
│   └── token/ccoha1.txt.gz
├── corpus2/
│   ├── lemma/ccoha2.txt.gz
│   └── token/ccoha2.txt.gz
├── targets.txt
└── truth/
    ├── binary.txt
    └── graded.txt
```

`data_loader.py` streams the `.gz` files lazily — no pre-processing required.

```bash
# Optional env overrides
SEMEVAL_DIR=data/semeval2020_ulscd_eng
SEMEVAL_MAX_SAMPLES=10
```

### OED Session Cookie (arbitrary words + Lexicographer Agent)

OED scraping requires a valid UofT library session. To set it up:

1. Log in to OED via the UofT library proxy in Chrome.
2. Install the [Cookie-Editor](https://cookie-editor.com/) browser extension.
3. On the OED site, open Cookie-Editor → **Export** → **Export as JSON**.
4. Save the file to `scripts/oed_cookie.json`.

> **Note:** OED session cookies (`JSESSIONID` + `AWSALBCORS`) expire on browser close or after ~30–60 min of inactivity. Re-export if scraping silently returns zero quotes. Keep the OED tab open during long evaluation runs.

The cookie file is git-ignored. Without it, the Lexicographer Agent falls back to Wiktionary.

---

## Usage

### Streamlit web UI

```bash
source .venv/bin/activate
streamlit run app.py
```

The sidebar provides:

- **Target word dropdown** — populated from `targets.txt` (SemEval benchmark words).
- **Custom word input** — type any English word. When no SemEval data exists, the app automatically fetches OED historical and modern quotations as corpus evidence and runs the full debate on those.
- **Max sentences per period** — slider (3–20).
- **Pre-debate grounding (BERT)** toggle — enables the grounding node.
- **Lexicographer Agent** toggle — enables the Definition Dossier pre-processing step.

Team Support and Team Refuse arguments stream side-by-side as the graph executes, followed by the Judge's structured verdict.

### CLI

```bash
source .venv/bin/activate
python main.py edge_nn        # run for a specific benchmark word
python main.py                # defaults to the first word in targets.txt
```

### Evaluation script

```bash
source .venv/bin/activate
python scripts/evaluate_pipeline.py \
    --lexicographer \
    --output eval_results/ \
    --max-samples 10
```

---

## Project Structure

```
MAD-SC/
├── mad_sc/
│   ├── state.py              GraphState (TypedDict) + JudgeVerdict + EtymologyResult (Pydantic)
│   ├── nodes.py              All node functions: lexicographer, team_support, team_refuse, judge
│   ├── graph.py              compile_graph() — builds topology from use_grounding / use_lexicographer flags
│   ├── data_loader.py        SemEval-2020 loader: streams .gz corpus files lazily
│   ├── etymology.py          OED (Playwright) + Wiktionary fetchers; fetch_etymology_context()
│   └── pre_debate_grounding.py  BERT-based SED/TD computation; HypothesisDocument
├── scripts/
│   ├── evaluate_pipeline.py  Batch evaluation against LSC-CTD / SemEval ground truth
│   ├── lsc_data_pipeline.py  OED data enrichment for LSC-CTD benchmark words
│   └── helsinki_wrapper.py   Helsinki NLP model wrapper (auxiliary)
├── data/
│   └── semeval2020_ulscd_eng/   Place downloaded SemEval data here (git-ignored)
├── app.py                    Streamlit frontend with real-time streaming + OED fallback
├── main.py                   CLI entry point
├── requirements.txt
└── .env                      API keys and backend selector (git-ignored)
```

---

## Evaluation

Ground-truth labels are in `truth/binary.txt` (Subtask 1: binary change/stable) and `truth/graded.txt` (Subtask 2: graded change score) from SemEval-2020 Task 1.

The pipeline is also validated against the **LSC-CTD Benchmark** (657 expert-annotated word–period pairs with Change Type and Causal Driver labels), which provides fine-grained accuracy beyond binary classification.

Representative evaluation results (fine-grained / coarse accuracy on LSC-CTD):

| Run | Configuration | Fine | Coarse |
|---|---|---|---|
| Baseline | Debate only | 28% | 40% |
| Run 4 | Two-stage judge + few-shot | 44% | 68% |
| Run 8 | + Lexicographer Agent (mock dossier) | 60% | 80% |
| Run 11 | + Lexicographer (OED Playwright + CoT synthesis) | 56% | 72% |
