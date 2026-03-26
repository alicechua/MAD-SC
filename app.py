"""MAD-SC Streamlit application.

Provides an interactive debate-style frontend that streams the multi-agent
debate round by round, with chat bubbles and typing animation, followed by
the Judge's structured verdict.

Run
---
    source .venv/bin/activate
    streamlit run app.py
"""

import json
import time

import streamlit as st
from dotenv import load_dotenv

from mad_sc.data_loader import (
    CORPUS1_LABEL,
    CORPUS2_LABEL,
    get_semeval_contexts,
    get_targets,
)
from mad_sc.etymology import fetch_etymology_context
from mad_sc.graph import compile_graph
from mad_sc.graph_multi import compile_multi_round_graph
from mad_sc.log_utils import append_debate_log

load_dotenv()


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


def _stream_text(text: str, delay: float = 0.02):
    """Yield text word-by-word for st.write_stream() typing animation."""
    words = text.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")
        time.sleep(delay)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MAD-SC · Multi-Agent Debate for Semantic Change",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Cached graphs
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


# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("MAD-SC")
    st.caption("Multi-Agent Debate for Semantic Change")
    st.divider()

    targets = get_targets()
    if not targets:
        targets = ["edge_nn", "record_nn", "attack_nn", "plane_nn", "gas_nn"]
        st.warning("targets.txt not found — showing example words.", icon="⚠️")

    custom_word = st.text_input(
        "Or enter any English word",
        placeholder="e.g. bead, canine, awful…",
        help="Type any word to analyse it using OED historical + modern quotes as corpus evidence.",
    )

    word = custom_word.strip().lower() if custom_word.strip() else st.selectbox(
        "Target word (LSC-CTD / SemEval)",
        options=targets,
        help="Select from the benchmark words, or type a custom word above.",
    )

    max_samples = st.slider(
        "Max sentences per period", min_value=3, max_value=20, value=10, step=1
    )

    st.divider()

    debate_mode = st.radio(
        "Debate mode",
        options=["Multi-round", "Single round"],
        index=0,
        help="Multi-round: opening + rebuttal rounds + closing statements → judge. Single round: both teams argue once → judge.",
    )

    num_rounds = 3
    if debate_mode == "Multi-round":
        num_rounds = st.slider(
            "Rebuttal rounds", min_value=1, max_value=5, value=3, step=1,
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
    st.markdown(f"**Time periods**  \n{CORPUS1_LABEL}  \n{CORPUS2_LABEL}")
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
    f"using SemEval-2020 Task 1 corpus evidence ({CORPUS1_LABEL} vs. {CORPUS2_LABEL})."
)

if not run_btn:
    st.info("Select a target word in the sidebar, then click **Run Debate**.")
    st.stop()

# ---------------------------------------------------------------------------
# Step 1 — Load corpus sentences
# ---------------------------------------------------------------------------

oed_mode = False
t_old_label = CORPUS1_LABEL
t_new_label = CORPUS2_LABEL

with st.status("Loading corpus sentences…", expanded=False) as load_status:
    sentences_old, sentences_new = get_semeval_contexts(word, max_samples=max_samples)

    if not sentences_old and not sentences_new:
        load_status.update(label="No SemEval data — fetching OED quotes…")
        oed_ctx = fetch_etymology_context(word)
        if oed_ctx["historical"] or oed_ctx["modern"]:
            sentences_old = [f"[{yr}] {txt}" for yr, txt in oed_ctx["historical"]]
            sentences_new = [f"[{yr}] {txt}" for yr, txt in oed_ctx["modern"]]
            t_old_label = "OED pre-1900 (historical)"
            t_new_label = "OED post-1900 (modern)"
            oed_mode = True
        else:
            load_status.update(label="No corpus data found.", state="error")
            st.error(
                f"No corpus sentences found for **{word}** in SemEval, "
                "and OED returned no quotes. Check the OED cookie or try another word."
            )
            st.stop()

    load_status.update(
        label=f"Loaded {len(sentences_old)} + {len(sentences_new)} sentences"
        + (" (OED)" if oed_mode else "") + ".",
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
        with st.expander(f"View {len(sentences_old) - 1} more corpus 1 sentence(s)…"):
            for s in sentences_old[1:]:
                st.markdown(f"- {s}")

with c_new:
    st.caption(f"{t_new_label} — {len(sentences_new)} sentence(s)")
    if sentences_new:
        st.markdown(f"- {sentences_new[0]}")
    if len(sentences_new) > 1:
        with st.expander(f"View {len(sentences_new) - 1} more corpus 2 sentence(s)…"):
            for s in sentences_new[1:]:
                st.markdown(f"- {s}")

if oed_mode:
    st.info(
        f"**OED mode** — '{word}' is not in the SemEval benchmark. "
        "Using Oxford English Dictionary historical and modern quotes as corpus evidence.",
        icon="📖",
    )

st.divider()

# ---------------------------------------------------------------------------
# Step 3 — Debate (chat-bubble layout, streamed round by round)
# ---------------------------------------------------------------------------

st.subheader("Adversarial Debate")

# Accumulate for log
_arg_change: str = ""
_arg_stable: str = ""
_debate_history: list = []
_verdict_dict: dict = {}
_dossier: str = ""

# State for tracking which round header to print next
_printed_round_headers: set = set()
_pending_support: str | None = None  # buffer Support text until Refuse arrives (multi-round)


def _round_label(rnd) -> str:
    if rnd == 0:
        return "Opening Statements"
    if rnd == "closing":
        return "Closing Statements"
    total = num_rounds if debate_mode == "Multi-round" else 1
    return f"Rebuttal Round {rnd} / {total}"


def _print_round_header(rnd):
    if rnd not in _printed_round_headers:
        _printed_round_headers.add(rnd)
        label = _round_label(rnd)
        st.markdown(f"---\n**{label}**")


# Build and run the graph
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

with st.status(f"Running debate for '{word}'…", expanded=True) as debate_status:

    for chunk in graph.stream(initial_state, stream_mode="updates"):
        for node_name, update in chunk.items():

            # ── Lexicographer ──────────────────────────────────────────────
            if node_name == "lexicographer":
                _dossier = update.get("lexicographer_dossier", "") or ""
                if _dossier:
                    with st.expander("📖 Lexicographer Dossier", expanded=False):
                        st.code(_dossier, language=None)
                    st.write("Lexicographer Agent completed.")

            # ── Grounding ─────────────────────────────────────────────────
            elif node_name == "grounding":
                st.write("BERT grounding completed.")

            # ── Opening / single-round Support ────────────────────────────
            elif node_name in ("team_support", "opening_support"):
                _arg_change = _extract_text(update.get("arg_change", ""))
                rnd = 0
                _print_round_header(rnd)
                with st.chat_message("Team Support", avatar="🔴"):
                    st.caption("Change Hypothesis")
                    st.write_stream(_stream_text(_arg_change))
                st.write("🔴 Team Support completed.")

            # ── Opening / single-round Refuse ─────────────────────────────
            elif node_name in ("team_refuse", "opening_refuse_record"):
                _arg_stable = _extract_text(update.get("arg_stable", ""))
                with st.chat_message("Team Refuse", avatar="🔵"):
                    st.caption("Stability Hypothesis")
                    st.write_stream(_stream_text(_arg_stable))
                st.write("🔵 Team Refuse completed.")

            # ── Rebuttal Support ──────────────────────────────────────────
            elif node_name == "rebuttal_support":
                _arg_change = _extract_text(update.get("arg_change", ""))
                history = update.get("debate_history", [])
                rnd = history[-1]["round"] if history else "?"
                _print_round_header(rnd)
                with st.chat_message("Team Support", avatar="🔴"):
                    st.caption(f"Rebuttal Round {rnd}")
                    st.write_stream(_stream_text(_arg_change))
                st.write(f"🔴 Team Support rebuttal {rnd} completed.")

            # ── Rebuttal Refuse ───────────────────────────────────────────
            elif node_name == "rebuttal_refuse":
                _arg_stable = _extract_text(update.get("arg_stable", ""))
                history = update.get("debate_history", [])
                rnd = history[-1]["round"] if history else "?"
                with st.chat_message("Team Refuse", avatar="🔵"):
                    st.caption(f"Rebuttal Round {rnd}")
                    st.write_stream(_stream_text(_arg_stable))
                st.write(f"🔵 Team Refuse rebuttal {rnd} completed.")

            # ── Closing Refuse ────────────────────────────────────────────
            elif node_name == "closing_refuse":
                _arg_stable = _extract_text(update.get("arg_stable", ""))
                _print_round_header("closing")
                with st.chat_message("Team Refuse", avatar="🔵"):
                    st.caption("Closing Statement")
                    st.write_stream(_stream_text(_arg_stable))
                st.write("🔵 Team Refuse closing completed.")

            # ── Closing Support ───────────────────────────────────────────
            elif node_name == "closing_support":
                _arg_change = _extract_text(update.get("arg_change", ""))
                with st.chat_message("Team Support", avatar="🔴"):
                    st.caption("Closing Statement")
                    st.write_stream(_stream_text(_arg_change))
                st.write("🔴 Team Support closing completed.")

            # ── Judge ─────────────────────────────────────────────────────
            elif node_name == "judge":
                _verdict_dict = update.get("verdict", {}) or {}
                st.write("⚖️ Judge rendered verdict.")

    debate_status.update(label="Debate complete.", state="complete", expanded=False)

# ---------------------------------------------------------------------------
# Step 4 — Judge verdict (rendered outside status so it stays visible)
# ---------------------------------------------------------------------------

if _verdict_dict:
    verdict_label = _verdict_dict.get("verdict", "UNKNOWN")
    change_detected = verdict_label == "CHANGE DETECTED"

    st.markdown("---")
    with st.chat_message("Judge", avatar="⚖️"):
        if change_detected:
            st.error(f"**{verdict_label}**", icon="🔴")
        else:
            st.success(f"**{verdict_label}**", icon="🟢")

        m1, m2, m3 = st.columns(3)
        m1.metric("Change Type", _verdict_dict.get("change_type") or "N/A")
        m2.metric("Causal Driver", _verdict_dict.get("causal_driver") or "N/A")
        m3.metric(
            "Break-point Year",
            _verdict_dict.get("break_point_year") or "N/A",
        )

        st.markdown("**Reasoning**")
        st.write_stream(_stream_text(_verdict_dict.get("reasoning", "")))

        with st.expander("Raw JSON output"):
            st.json(_verdict_dict)

# ---------------------------------------------------------------------------
# Step 5 — Persist the full debate trail
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
