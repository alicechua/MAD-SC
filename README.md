# MAD-SC: Multi-Agent Debate for Semantic Change

A LangGraph-based agentic pipeline that classifies **diachronic semantic change** using an adversarial tri-agent debate architecture backed by SemEval-2020 Task 1 corpus evidence.

*Course project — CSC2611: Computational Models of Semantic Change, University of Toronto.*

---

## Overview

Static embedding methods for detecting lexical semantic change are powerful but opaque — they quantify *that* a word shifted without explaining *how* or *why*. MAD-SC addresses this by framing change classification as a structured debate:

1. **Team Support** argues that genuine semantic change occurred, citing corpus evidence and proposing a Change Type and Causal Driver.
2. **Team Refuse** defends the null hypothesis of semantic stability, reframing new usages as situational polysemy.
3. **LLM Judge** weighs the comparative evidence and renders a structured JSON verdict.

The pipeline validates quantitative embedding signals with granular, auditable natural-language reasoning.

---

## Architecture

```
                    ┌─────────────────────┐
                    │    GraphState        │
                    │  word, t_old, t_new  │
                    │  sentences_old/new   │
                    └──────────┬──────────┘
                               │ START
               ┌───────────────┴───────────────┐
               ▼                               ▼
   ┌───────────────────────┐     ┌───────────────────────┐
   │    Team Support Node   │     │    Team Refuse Node    │
   │                        │     │                        │
   │  Hypothesis: change    │     │  Hypothesis: stable    │
   │  occurred.             │     │  (null hypothesis).    │
   │                        │     │                        │
   │  • Cites incompatible  │     │  • Cites stable        │
   │    new-period usages   │     │    new-period usages   │
   │  • Classifies Change   │     │  • Argues new senses   │
   │    Type + Causal Driver│     │    are polysemy only   │
   │                        │     │                        │
   │  → writes arg_change   │     │  → writes arg_stable   │
   └───────────┬───────────┘     └───────────┬───────────┘
               │  (parallel, fan-in)          │
               └───────────────┬─────────────┘
                               ▼
                   ┌───────────────────────┐
                   │      Judge Node        │
                   │                        │
                   │  Evaluates arg_change  │
                   │  vs arg_stable by      │
                   │  comparative weight    │
                   │  of evidence.          │
                   │                        │
                   │  Structured output →   │
                   │  JudgeVerdict (JSON)   │
                   └───────────┬───────────┘
                               │ END
```

### Strict Taxonomy

The pipeline enforces constrained decoding over a fixed taxonomy:

| Dimension | Values |
|---|---|
| **Verdict** | `CHANGE DETECTED` \| `STABLE` |
| **Change Type** | `Generalization` \| `Specialization` \| `Co-hyponymous Transfer` |
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
```

---

## Configuration

Copy the template and fill in your keys:

```bash
# .env
LLM_BACKEND=google_ai_studio      # or "vertex_ai"

# Google AI Studio (key starts with AIza)
# Obtain at: https://aistudio.google.com/apikey
GOOGLE_AI_STUDIO_KEY=AIza...
DEFAULT_MODEL_GAS=gemini-2.5-flash

# Vertex AI Express (key starts with AQ.)
# Requires Vertex AI API enabled + billing in Google Cloud Console
VERTEX_AI_KEY=AQ...
DEFAULT_MODEL_VAI=gemini-2.5-flash
```

Switch backends by changing `LLM_BACKEND`. Models can be changed independently per backend via `DEFAULT_MODEL_GAS` / `DEFAULT_MODEL_VAI`.

---

## Data Setup

MAD-SC uses the **SemEval-2020 Task 1 (English)** dataset — a pair of historical English corpora covering two non-overlapping time periods, with 37 expert-annotated target words for lexical semantic change.

| Corpus | Time period | Size |
|---|---|---|
| Corpus 1 | 1810–1860 | ~6 million tokens |
| Corpus 2 | 1960–2010 | ~6 million tokens |

### Download

The dataset is freely available for education and research:

**[https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd-eng/](https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd-eng/)**

Download and unzip `semeval2020_ulscd_eng.zip`, then place the unzipped folder at:

```
data/semeval2020_ulscd_eng/
```

The expected directory layout after extraction is:

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

No further pre-processing is required — `data_loader.py` streams the `.gz` files directly at query time.

### Environment override

```bash
SEMEVAL_DIR=data/semeval2020_ulscd_eng   # path to the dataset directory
SEMEVAL_MAX_SAMPLES=10                   # sentences retrieved per corpus
```

---

## Usage

### CLI

```bash
source .venv/bin/activate
python main.py edge_nn        # run for a specific target word
python main.py                # defaults to the first word in targets.txt
```

### Streamlit web UI

```bash
source .venv/bin/activate
streamlit run app.py
```

Open the sidebar, select a target word from the dropdown (populated from `targets.txt`), and click **Run Debate**. The UI streams Team Support and Team Refuse arguments side-by-side as the graph executes, then displays the Judge verdict at the bottom.

---

## Project Structure

```
MAD-SC/
├── mad_sc/
│   ├── state.py          GraphState (TypedDict) + JudgeVerdict (Pydantic)
│   ├── nodes.py          team_support_node, team_refuse_node, judge_node
│   ├── graph.py          compile_graph() — LangGraph StateGraph
│   └── data_loader.py    SemEval-2020 loader: streams corpus gz files
├── data/
│   └── semeval2020_ulscd_eng/   Place downloaded SemEval data here
├── app.py                Streamlit frontend with real-time streaming
├── main.py               CLI entry point
├── requirements.txt
└── .env                  API keys and backend selector
```

---

## Evaluation

The 37 target words include binary and graded ground-truth labels in `truth/binary.txt` and `truth/graded.txt` (SemEval-2020 Task 1 subtasks 1 and 2). Predicted verdicts from the Judge node can be compared directly to these ground-truth labels. The system is also compatible with the **LSC-CTD Benchmark** (657 expert-validated word–period pairs with Change Type and Causal Driver annotations).
