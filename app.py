"""MAD-SC Streamlit application.

Provides an interactive debate-style frontend that streams the multi-agent
debate round by round. All debate rounds are pre-rendered as fixed-height
scrollable containers, so the entire debate is visible on screen simultaneously.
A progress bar and per-agent status badges show pipeline state in real time.

Run
---
    source .venv/bin/activate
    streamlit run app.py
"""

import json
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from mad_sc.data_loader import (
    CORPUS1_LABEL,
    CORPUS2_LABEL,
    get_semeval_contexts,
    get_targets,
)
from mad_sc.graph import compile_graph
from mad_sc.graph_multi import compile_multi_round_graph
from mad_sc.log_utils import append_debate_log

# Make scripts/ importable (lsc_data_pipeline is not a package)
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from lsc_data_pipeline import HistoricalScraper, ModernScraper  # noqa: E402

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LSC_CTD_JSON = Path("data/lsc_context_data_engl.json")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_text(content) -> str:
    """Normalise an LLM response content value to a plain string."""
    if isinstance(content, str):
        return content
    if hasattr(content, "content"):
        return _extract_text(content.content)
    if isinstance(content, list):
        return "\n\n".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and "text" in block
        )
    return str(content)


def _stream_text(text: str, delay: float = 0.015):
    """Yield text word-by-word for st.write_stream() typing animation."""
    words = text.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")
        time.sleep(delay)


def _total_steps(
    debate_mode: str,
    num_rounds: int,
    use_grounding: bool,
    use_lexicographer: bool,
) -> int:
    """Compute total pipeline steps for progress bar calculation."""
    n = 0
    if use_grounding:
        n += 1
    if use_lexicographer:
        n += 1
    if debate_mode == "Multi-round":
        n += 2               # opening support + refuse
        n += 2 * num_rounds  # rebuttal rounds
        n += 2               # closing refuse + support
    else:
        n += 2               # team_support + team_refuse
    n += 1                   # judge
    return n


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MAD-SC · Multi-Agent Debate for Semantic Change",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------


@st.cache_resource
def get_single_graph(use_grounding: bool, use_lexicographer: bool):
    return compile_graph(use_grounding=use_grounding, use_lexicographer=use_lexicographer)


@st.cache_resource
def get_multi_graph(num_rounds: int, use_grounding: bool, use_lexicographer: bool):
    return compile_multi_round_graph(
        num_rounds=num_rounds,
        use_grounding=use_grounding,
        use_lexicographer=use_lexicographer,
    )


@st.cache_data
def load_lsc_ctd_data() -> dict:
    """Load lsc_context_data_engl.json → {word: entry dict}."""
    with open(LSC_CTD_JSON, encoding="utf-8") as f:
        entries = json.load(f)
    return {e["word"]: e for e in entries}


@st.cache_data(show_spinner=False)
def fetch_custom_sentences(word: str, max_samples: int) -> tuple:
    """Run HistoricalScraper + ModernScraper for an arbitrary word.

    Returns (sentences_old, sentences_new, historical_source).
    Cached per (word, max_samples) so re-running the same word is instant.
    """
    hist = HistoricalScraper(target_quotes=max_samples, use_llm_filter=True)
    modern = ModernScraper(target_sentences=max_samples)
    hist_sentences, hist_source = hist.collect(word)
    mod_sentences = modern.collect(word)
    return hist_sentences[:max_samples], mod_sentences[:max_samples], hist_source


# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("MAD-SC")
    st.caption("Multi-Agent Debate for Semantic Change")
    st.divider()

    # ── Mode selector ────────────────────────────────────────────────────────
    mode = st.radio(
        "Data source",
        options=["LSC-CTD Benchmark", "SemEval Benchmark", "Custom Word"],
        index=0,
        help=(
            "**LSC-CTD Benchmark**: pre-cached OED/Helsinki corpus data for the 25 "
            "Blank-taxonomy English words.\n\n"
            "**SemEval Benchmark**: CCOHA corpus (1810–1860 vs 1960–2010) for the 37 "
            "SemEval-2020 Task 1 English words.\n\n"
            "**Custom Word**: live-scrape OED/Wiktionary (historical) + web (modern) "
            "for any English word you provide."
        ),
    )

    # ── Word selector (mode-dependent) ──────────────────────────────────────
    if mode == "LSC-CTD Benchmark":
        lsc_data = load_lsc_ctd_data()
        word = st.selectbox("Target word (LSC-CTD)", sorted(lsc_data.keys()))
        t_old_default = "Historical (pre-1900)"
        t_new_default = "Modern (post-1900)"

    elif mode == "SemEval Benchmark":
        targets = get_targets()
        if not targets:
            targets = ["edge_nn", "record_nn", "attack_nn", "plane_nn", "gas_nn"]
            st.warning("targets.txt not found — showing example words.", icon="⚠️")
        word = st.selectbox("Target word (SemEval)", options=targets)
        t_old_default = CORPUS1_LABEL
        t_new_default = CORPUS2_LABEL

    else:  # Custom Word
        word = st.text_input(
            "Enter any English word",
            placeholder="e.g. awful, gossip, nice…",
            help="Type any English word. Historical sentences are fetched from OED "
                 "(requires oed_cookie.json) or Wiktionary. Modern sentences are "
                 "scraped from the open web.",
        ).strip().lower()

        st.caption("Time frames — used as agent prompt labels")
        col1, col2 = st.columns(2)
        with col1:
            old_start = st.number_input(
                "Old period start", value=1810, min_value=800, max_value=1950, step=10
            )
            old_end = st.number_input(
                "Old period end", value=1860, min_value=800, max_value=1950, step=10
            )
        with col2:
            new_start = st.number_input(
                "New period start", value=1960, min_value=1900, max_value=2020, step=10
            )
            new_end = st.number_input(
                "New period end", value=2010, min_value=1900, max_value=2020, step=10
            )
        t_old_default = f"{int(old_start)}–{int(old_end)}"
        t_new_default = f"{int(new_start)}–{int(new_end)}"

    # ── Shared controls ──────────────────────────────────────────────────────
    max_samples = st.slider(
        "Max sentences per period", min_value=3, max_value=20, value=10, step=1
    )

    st.divider()

    debate_mode = st.radio(
        "Debate mode",
        options=["Multi-round", "Single round"],
        index=0,
        help=(
            "Multi-round: opening + rebuttal rounds + closing statements → judge. "
            "Single round: both teams argue once → judge."
        ),
    )

    num_rounds = 3
    if debate_mode == "Multi-round":
        num_rounds = st.slider(
            "Rebuttal rounds",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
            help="Number of rebuttal rounds after the opening exchange.",
        )

    st.divider()

    use_grounding = st.toggle(
        "Pre-debate grounding (BERT)",
        value=False,
        help="Run BERT-based SED/TD analysis before the debate and inject quantitative evidence into agent prompts.",
    )
    use_lexicographer = st.toggle(
        "Lexicographer Agent",
        value=True,
        help="Run the Lexicographer Agent before the debate to produce a Definition Dossier.",
    )

    run_btn = st.button("Run Debate", type="primary", use_container_width=True)

    st.divider()
    st.markdown(f"**Time periods**  \n{t_old_default}  \n{t_new_default}")
    st.divider()
    st.caption(
        "🔴 **Team Support** (change hypothesis) vs 🔵 **Team Refuse** "
        "(stability hypothesis), adjudicated by ⚖️ **Judge**."
    )

# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------

st.title("MAD-SC: Multi-Agent Debate for Semantic Change")
st.markdown(
    "An adversarial agentic pipeline that classifies diachronic semantic change "
    "using historical and modern corpus evidence."
)

if not run_btn:
    st.info("Select a data source and target word in the sidebar, then click **Run Debate**.")
    st.stop()

# ---------------------------------------------------------------------------
# Step 1 — Load corpus sentences
# ---------------------------------------------------------------------------

t_old_label = t_old_default
t_new_label = t_new_default
hist_source = ""

with st.status("Loading corpus sentences…", expanded=False) as load_status:

    if mode == "LSC-CTD Benchmark":
        entry = lsc_data.get(word, {})
        sentences_old = entry.get("historical_context", [])[:max_samples]
        sentences_new = entry.get("modern_context", [])[:max_samples]
        hist_source = entry.get("historical_source", "cached")
        if not sentences_old and not sentences_new:
            load_status.update(label="No LSC-CTD data found.", state="error")
            st.error(f"No corpus data found for **{word}** in lsc_context_data_engl.json.")
            st.stop()
        load_status.update(
            label=f"Loaded {len(sentences_old)} + {len(sentences_new)} sentences "
                  f"(LSC-CTD / {hist_source}).",
            state="complete",
        )

    elif mode == "SemEval Benchmark":
        sentences_old, sentences_new = get_semeval_contexts(word, max_samples=max_samples)
        if not sentences_old and not sentences_new:
            load_status.update(label="No SemEval data found.", state="error")
            st.error(f"No SemEval corpus sentences found for **{word}**.")
            st.stop()
        load_status.update(
            label=f"Loaded {len(sentences_old)} + {len(sentences_new)} sentences (SemEval).",
            state="complete",
        )

    else:  # Custom Word
        if not word:
            load_status.update(label="No word entered.", state="error")
            st.warning("Please enter a word in the sidebar.")
            st.stop()
        load_status.update(
            label=f"Scraping sentences for '{word}'… (may take ~30 s)",
            expanded=True,
        )
        sentences_old, sentences_new, hist_source = fetch_custom_sentences(word, max_samples)
        if not sentences_old and not sentences_new:
            load_status.update(label="No sentences found.", state="error")
            st.error(
                f"No sentences found for **{word}**. "
                "Check your OED cookie (`scripts/oed_cookie.json`) or try another word."
            )
            st.stop()
        load_status.update(
            label=f"Loaded {len(sentences_old)} + {len(sentences_new)} sentences "
                  f"(historical: {hist_source}, modern: web).",
            state="complete",
        )

# ---------------------------------------------------------------------------
# Step 2 — Show retrieved sentences
# ---------------------------------------------------------------------------

st.markdown("**Retrieved Corpus Sentences**")
c_old, c_new = st.columns(2)

with c_old:
    st.caption(f"{t_old_label} — {len(sentences_old)} sentence(s)")
    if sentences_old:
        st.markdown(f"- {sentences_old[0]}")
    if len(sentences_old) > 1:
        with st.expander(f"View {len(sentences_old) - 1} more sentence(s)…"):
            for s in sentences_old[1:]:
                st.markdown(f"- {s}")

with c_new:
    st.caption(f"{t_new_label} — {len(sentences_new)} sentence(s)")
    if sentences_new:
        st.markdown(f"- {sentences_new[0]}")
    if len(sentences_new) > 1:
        with st.expander(f"View {len(sentences_new) - 1} more sentence(s)…"):
            for s in sentences_new[1:]:
                st.markdown(f"- {s}")

# Mode banner
if mode == "LSC-CTD Benchmark":
    st.info(
        f"**LSC-CTD mode** — pre-cached corpus data for '{word}' "
        f"(source: {hist_source}).",
        icon="📚",
    )
elif mode == "Custom Word":
    st.info(
        f"**Custom mode** — '{word}': historical sentences from {hist_source or 'web'}, "
        "modern sentences from web.",
        icon="🌐",
    )

st.divider()

# ---------------------------------------------------------------------------
# Step 3 — Debate UI: progress bar + status badges + pre-created round containers
# ---------------------------------------------------------------------------

st.subheader("Adversarial Debate")

# ── Progress bar ─────────────────────────────────────────────────────────────
total_steps = _total_steps(debate_mode, num_rounds, use_grounding, use_lexicographer)
_step = 0
progress_bar = st.progress(0, text="Pipeline ready.")

# ── Agent status badges ───────────────────────────────────────────────────────
s_col, r_col, j_col = st.columns(3)
with s_col:
    st.markdown("🔴 **Team Support**")
    support_status = st.empty()
    support_status.caption("⏳ Waiting…")
with r_col:
    st.markdown("🔵 **Team Refuse**")
    refuse_status = st.empty()
    refuse_status.caption("⏳ Waiting…")
with j_col:
    st.markdown("⚖️ **Judge**")
    judge_status = st.empty()
    judge_status.caption("⏳ Waiting…")

# ── Lexicographer dossier slot (above rounds, filled during stream) ───────────
lexi_placeholder = st.empty()

# ── Pre-create all round containers ──────────────────────────────────────────
if debate_mode == "Multi-round":
    round_defs = (
        [("Opening Statements", 0)]
        + [(f"Rebuttal {i} / {num_rounds}", i) for i in range(1, num_rounds + 1)]
        + [("Closing Statements", "closing")]
    )
else:
    round_defs = [("Debate", 0)]

round_containers: dict = {}  # key → (support_container, refuse_container)

for label, key in round_defs:
    st.markdown(f"**{label}**")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("🔴 Change Hypothesis")
        s_box = st.container(height=220)
    with c2:
        st.caption("🔵 Stability Hypothesis")
        r_box = st.container(height=220)
    round_containers[key] = (s_box, r_box)

# Verdict slot at the bottom
judge_placeholder = st.empty()

# ---------------------------------------------------------------------------
# Step 4 — Build and run the graph, streaming into pre-created containers
# ---------------------------------------------------------------------------

word_type = "verb" if word.endswith("_vb") else "noun"

initial_state = {
    "word": word,
    "word_type": word_type,
    "t_old": t_old_label,
    "t_new": t_new_label,
    "sentences_old": sentences_old,
    "sentences_new": sentences_new,
    "arg_change": "",
    "arg_stable": "",
    "verdict": None,
    "num_rounds": num_rounds,
    "current_round": 1,
    "debate_history": [],
}

if debate_mode == "Multi-round":
    graph = get_multi_graph(num_rounds, use_grounding, use_lexicographer)
else:
    graph = get_single_graph(use_grounding, use_lexicographer)

# Accumulate for log
_arg_change: str = ""
_arg_stable: str = ""
_debate_history: list = []
_verdict_dict: dict = {}
_dossier: str = ""


def _advance(label: str) -> None:
    """Increment progress bar by one step."""
    global _step
    _step += 1
    progress_bar.progress(
        min(_step / total_steps, 1.0),
        text=f"Step {_step} / {total_steps} — {label}",
    )


for chunk in graph.stream(initial_state, stream_mode="updates"):
    for node_name, update in chunk.items():

        # ── Lexicographer ──────────────────────────────────────────────────
        if node_name == "lexicographer":
            _dossier = update.get("lexicographer_dossier", "") or ""
            if _dossier:
                with lexi_placeholder.container():
                    with st.expander("📖 Lexicographer Dossier", expanded=False):
                        st.code(_dossier, language=None)
            _advance("Lexicographer")

        # ── Grounding ─────────────────────────────────────────────────────
        elif node_name == "grounding":
            _advance("BERT Grounding")

        # ── Opening / single-round Support ────────────────────────────────
        elif node_name in ("team_support", "opening_support"):
            _arg_change = _extract_text(update.get("arg_change", ""))
            support_status.caption("💭 Generating…")
            s_box, _ = round_containers[0]
            with s_box:
                st.write_stream(_stream_text(_arg_change))
            support_status.caption("✅ Done")
            _advance("Team Support — opening")

        # ── Opening / single-round Refuse ─────────────────────────────────
        elif node_name in ("team_refuse", "opening_refuse", "opening_refuse_record"):
            _arg_stable = _extract_text(update.get("arg_stable", ""))
            refuse_status.caption("💭 Generating…")
            _, r_box = round_containers[0]
            with r_box:
                st.write_stream(_stream_text(_arg_stable))
            refuse_status.caption("✅ Done")
            _advance("Team Refuse — opening")

        # ── Rebuttal Support ──────────────────────────────────────────────
        elif node_name == "rebuttal_support":
            _arg_change = _extract_text(update.get("arg_change", ""))
            history = update.get("debate_history", [])
            rnd = history[-1]["round"] if history else 1
            support_status.caption(f"💭 Rebuttal {rnd}…")
            s_box, _ = round_containers.get(rnd, list(round_containers.values())[1])
            with s_box:
                st.write_stream(_stream_text(_arg_change))
            support_status.caption("✅ Done")
            _advance(f"Support — rebuttal {rnd}")

        # ── Rebuttal Refuse ───────────────────────────────────────────────
        elif node_name == "rebuttal_refuse":
            _arg_stable = _extract_text(update.get("arg_stable", ""))
            history = update.get("debate_history", [])
            rnd = history[-1]["round"] if history else 1
            refuse_status.caption(f"💭 Rebuttal {rnd}…")
            _, r_box = round_containers.get(rnd, list(round_containers.values())[1])
            with r_box:
                st.write_stream(_stream_text(_arg_stable))
            refuse_status.caption("✅ Done")
            _advance(f"Refuse — rebuttal {rnd}")

        # ── Closing Refuse ────────────────────────────────────────────────
        elif node_name == "closing_refuse":
            _arg_stable = _extract_text(update.get("arg_stable", ""))
            refuse_status.caption("💭 Closing…")
            _, r_box = round_containers["closing"]
            with r_box:
                st.write_stream(_stream_text(_arg_stable))
            refuse_status.caption("✅ Done")
            _advance("Refuse — closing")

        # ── Closing Support ───────────────────────────────────────────────
        elif node_name == "closing_support":
            _arg_change = _extract_text(update.get("arg_change", ""))
            support_status.caption("💭 Closing…")
            s_box, _ = round_containers["closing"]
            with s_box:
                st.write_stream(_stream_text(_arg_change))
            support_status.caption("✅ Done")
            _advance("Support — closing")

        # ── Judge ─────────────────────────────────────────────────────────
        elif node_name == "judge":
            _verdict_dict = update.get("verdict", {}) or {}
            judge_status.caption("💭 Deliberating…")
            _advance("Judge")
            judge_status.caption("✅ Verdict rendered")

progress_bar.progress(1.0, text="Pipeline complete.")

# ---------------------------------------------------------------------------
# Step 5 — Judge verdict (rendered into pre-created slot)
# ---------------------------------------------------------------------------

if _verdict_dict:
    verdict_label = _verdict_dict.get("verdict", "UNKNOWN")
    change_detected = verdict_label == "CHANGE DETECTED"

    with judge_placeholder.container():
        st.markdown("---")
        with st.chat_message("Judge", avatar="⚖️"):
            if change_detected:
                st.error(f"**{verdict_label}**", icon="🔴")
            else:
                st.success(f"**{verdict_label}**", icon="🟢")

            m1, m2, m3 = st.columns(3)
            m1.metric("Change Type", _verdict_dict.get("change_type") or "N/A")
            m2.metric("Causal Driver", _verdict_dict.get("causal_driver") or "N/A")
            m3.metric("Break-point Year", _verdict_dict.get("break_point_year") or "N/A")

            st.markdown("**Reasoning**")
            st.write_stream(_stream_text(_verdict_dict.get("reasoning", "")))

            with st.expander("Raw JSON output"):
                st.json(_verdict_dict)

# ---------------------------------------------------------------------------
# Step 6 — Persist the full debate trail
# ---------------------------------------------------------------------------

_final_state = {
    **initial_state,
    "arg_change": _arg_change,
    "arg_stable": _arg_stable,
    "debate_history": _debate_history,
    "verdict": _verdict_dict,
}
append_debate_log(_final_state)
st.caption("✅ Debate trail saved to `debate_logs.json`.")
