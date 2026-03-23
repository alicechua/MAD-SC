"""MAD-SC Streamlit application.

Provides an interactive frontend that streams the tri-agent debate in
real-time and renders the Judge's structured verdict.

Run
---
    source .venv/bin/activate
    streamlit run app.py
"""

import json

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
from mad_sc.log_utils import append_debate_log

load_dotenv()


def _extract_text(content) -> str:
    """Normalise an LLM response content value to a plain string.

    Handles three formats that different LangChain / Anthropic SDK versions
    may return:
      - str                          → returned as-is
      - AIMessage (has .content)     → recurse on .content
      - list[dict] with 'text' keys  → join all 'text' values (Anthropic blocks)
    """
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


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MAD-SC · Multi-Agent Debate for Semantic Change",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Cached graph (avoids recompilation on every Streamlit rerun)
# ---------------------------------------------------------------------------


@st.cache_resource
def get_graph(use_grounding: bool, use_lexicographer: bool):
    return compile_graph(use_grounding=use_grounding, use_lexicographer=use_lexicographer)


# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("MAD-SC")
    st.caption("Multi-Agent Debate for Semantic Change")
    st.divider()

    # Populate dropdown from targets.txt; fall back to a hardcoded list.
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

    use_grounding = st.toggle(
        "Pre-debate grounding (BERT)",
        value=False,
        help="Run BERT-based SED/TD analysis before the debate and inject quantitative evidence into agent prompts. Slower but adds embedding-distance signal.",
    )
    use_lexicographer = st.toggle(
        "Lexicographer Agent",
        value=True,
        help="Run the Lexicographer Agent before the debate to produce a Definition Dossier (historical/modern senses + change mechanism). Anchors teams to etymological ground truth.",
    )

    run_btn = st.button("Run Debate", type="primary", use_container_width=True)

    st.divider()
    st.markdown(
        f"**Time periods**  \n{CORPUS1_LABEL}  \n{CORPUS2_LABEL}"
    )
    st.divider()
    st.caption(
        "Agents: **Team Support** (change hypothesis) vs **Team Refuse** "
        "(stability hypothesis), adjudicated by an **LLM Judge**."
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

oed_mode = False  # set True when OED quotes are used as corpus sentences
t_old_label = CORPUS1_LABEL
t_new_label = CORPUS2_LABEL

with st.status("Loading corpus sentences…", expanded=False) as load_status:
    sentences_old, sentences_new = get_semeval_contexts(word, max_samples=max_samples)

    if not sentences_old and not sentences_new:
        # Fall back to OED quotes for any arbitrary word.
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
# Step 3 — Debate columns  (populated by streaming)
# ---------------------------------------------------------------------------

st.subheader("Adversarial Debate")

col_support, col_refuse = st.columns(2, gap="large")

with col_support:
    st.markdown("##### Team Support")
    st.caption("Hypothesis: semantic change HAS occurred")
    support_box = st.empty()
    support_box.info("Waiting for Team Support agent…")

with col_refuse:
    st.markdown("##### Team Refuse")
    st.caption("Null hypothesis: semantic stability")
    refuse_box = st.empty()
    refuse_box.info("Waiting for Team Refuse agent…")

st.divider()

# ---------------------------------------------------------------------------
# Step 4 — Judge verdict placeholder
# ---------------------------------------------------------------------------

st.subheader("Judge Verdict")
verdict_box = st.empty()
verdict_box.info("Awaiting Judge verdict…")

# ---------------------------------------------------------------------------
# Step 5 — Stream the LangGraph pipeline
# ---------------------------------------------------------------------------

# Derive word_type from the trailing _nn / _vb suffix.
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
}

graph = get_graph(use_grounding, use_lexicographer)

# Accumulate debate content across streamed node updates so we can persist
# the full trail after streaming completes (without a second LLM call).
_arg_change: str = ""
_arg_stable: str = ""
_verdict_dict: dict = {}

with st.status(
    f"Running debate for '{word}'…",
    expanded=True,
) as debate_status:

    for chunk in graph.stream(initial_state, stream_mode="updates"):
        for node_name, update in chunk.items():

            # ── Team Support ──────────────────────────────────────────────
            if node_name == "team_support":
                _arg_change = _extract_text(update.get("arg_change", ""))
                support_box.markdown(_arg_change)
                st.write("Team Support completed.")

            # ── Team Refuse ───────────────────────────────────────────────
            elif node_name == "team_refuse":
                _arg_stable = _extract_text(update.get("arg_stable", ""))
                refuse_box.markdown(_arg_stable)
                st.write("Team Refuse completed.")

            # ── Judge ─────────────────────────────────────────────────────
            elif node_name == "judge":
                _verdict_dict = update.get("verdict", {})
                st.write("Judge rendered verdict.")

                verdict_label = _verdict_dict.get("verdict", "UNKNOWN")
                change_detected = verdict_label == "CHANGE DETECTED"

                with verdict_box.container():
                    if change_detected:
                        st.error(f"**{verdict_label}**", icon="🔴")
                    else:
                        st.success(f"**{verdict_label}**", icon="🟢")

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Verdict", verdict_label)
                    m2.metric("Change Type", _verdict_dict.get("change_type") or "N/A")
                    m3.metric("Causal Driver", _verdict_dict.get("causal_driver") or "N/A")

                    if _verdict_dict.get("break_point_year"):
                        st.metric("Estimated Break-point Year", _verdict_dict["break_point_year"])

                    st.markdown("**Reasoning**")
                    st.markdown(_verdict_dict.get("reasoning", ""))

                    with st.expander("Raw JSON output"):
                        st.json(_verdict_dict)

    debate_status.update(label="Debate complete.", state="complete", expanded=False)

# ---------------------------------------------------------------------------
# Persist the full debate trail once streaming is complete.
# All three values (_arg_change, _arg_stable, _verdict_dict) were accumulated
# per-node during the stream loop above; no second LLM call is needed.
# ---------------------------------------------------------------------------

_final_state = {
    **initial_state,
    "arg_change": _arg_change,
    "arg_stable": _arg_stable,
    "verdict": _verdict_dict,
}
append_debate_log(_final_state)
st.caption("✅ Debate trail saved to `debate_logs.json`.")
