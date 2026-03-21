"""LangGraph node functions for the MAD-SC tri-agent debate pipeline.

Nodes
-----
team_support_node  –  Team Support: argues that semantic change HAS occurred.
team_refuse_node   –  Team Refuse: argues that semantic change has NOT occurred.
judge_node         –  LLM Judge: evaluates both arguments and renders a structured verdict.
"""

import os

from typing import Any
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from mad_sc.state import GraphState, JudgeVerdict

load_dotenv()

# Per-backend default models — override in .env with DEFAULT_MODEL_GAS / DEFAULT_MODEL_VAI / DEFAULT_MODEL_GROQ
_BACKEND_GAS = "google_ai_studio"
_BACKEND_VAI = "vertex_ai"
_BACKEND_GROQ = "groq"

_DEFAULT_MODEL_GAS = "gemini-2.5-flash"
_DEFAULT_MODEL_VAI = "gemini-2.5-flash"
_DEFAULT_MODEL_GROQ = "llama3-70b-8192"


def _get_llm(model: str | None = None, temperature: float = 0.7) -> Any:
    """Instantiate a Gemini or Groq LLM using the backend configured in .env.

    LLM_BACKEND=google_ai_studio  →  uses GOOGLE_AI_STUDIO_KEY
                                      default model: DEFAULT_MODEL_GAS (or gemini-2.5-flash)
                                      calls generativelanguage.googleapis.com
    LLM_BACKEND=vertex_ai         →  uses VERTEX_AI_KEY
                                      default model: DEFAULT_MODEL_VAI (or gemini-2.5-flash)
                                      calls aiplatform.googleapis.com (Express mode)
    LLM_BACKEND=groq              →  uses GROQ_API_KEY
                                      default model: DEFAULT_MODEL_GROQ (or llama3-70b-8192)

    Pass `model` explicitly to override the per-backend default for a single call.
    """
    backend = os.getenv("LLM_BACKEND", _BACKEND_GAS).strip().lower()

    if backend == _BACKEND_VAI:
        api_key = os.getenv("VERTEX_AI_KEY")
        if not api_key:
            raise EnvironmentError(
                "LLM_BACKEND=vertex_ai but VERTEX_AI_KEY is not set in .env."
            )
        resolved_model = model or os.getenv("DEFAULT_MODEL_VAI", _DEFAULT_MODEL_VAI)
        return ChatGoogleGenerativeAI(
            model=resolved_model,
            google_api_key=api_key,
            temperature=temperature,
            vertexai=True,
            location="global",
        )

    if backend == _BACKEND_GROQ:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "LLM_BACKEND=groq but GROQ_API_KEY is not set in .env."
            )
        resolved_model = model or os.getenv("DEFAULT_MODEL_GROQ", _DEFAULT_MODEL_GROQ)
        return ChatGroq(
            model=resolved_model,
            api_key=api_key,
            temperature=temperature,
        )

    # Default: Google AI Studio
    if backend != _BACKEND_GAS:
        raise ValueError(
            f"Unknown LLM_BACKEND '{backend}'. "
            f"Use '{_BACKEND_GAS}', '{_BACKEND_VAI}', or '{_BACKEND_GROQ}'."
        )
    api_key = os.getenv("GOOGLE_AI_STUDIO_KEY")
    if not api_key:
        raise EnvironmentError(
            "LLM_BACKEND=google_ai_studio but GOOGLE_AI_STUDIO_KEY is not set in .env."
        )
    resolved_model = model or os.getenv("DEFAULT_MODEL_GAS", _DEFAULT_MODEL_GAS)
    return ChatGoogleGenerativeAI(
        model=resolved_model,
        google_api_key=api_key,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Team Support Node
# ---------------------------------------------------------------------------

_SUPPORT_SYSTEM = """You are a computational linguist on **Team Support**.
Your sole mission is to build the strongest possible argument that the target word
has undergone genuine *diachronic* semantic change between the two time periods.

Requirements
------------
1. Identify specific sentences from the NEW period whose semantics are **incompatible**
   with the dominant meaning established in the OLD period.
2. Classify the shift into EXACTLY ONE Change Type:
   - **Generalization** (Broadening): the word's meaning expanded to cover more concepts.
   - **Specialization** (Narrowing): the word's meaning narrowed to a more specific domain.
   - **Co-hyponymous Transfer**: the word shifted to a semantically related but distinct
     concept at the same level of abstraction.
3. Hypothesize EXACTLY ONE Causal Driver:
   - **Cultural Shift** (Global): driven by broad societal, technological, or cultural changes.
   - **Linguistic Drift** (Local): driven by internal linguistic processes (metaphor, metonymy, analogy).

Be specific, cite corpus evidence directly, and construct a compelling argument."""

_SUPPORT_USER = """Analyze the semantic change of the word: "{word}" (used as a {word_type})

SENTENCES FROM {t_old}:
{sentences_old}

SENTENCES FROM {t_new}:
{sentences_new}

Build Arg_change — your argument that a genuine semantic shift has occurred between \
{t_old} and {t_new}. Cite specific sentence evidence, name the Change Type, and \
identify the Causal Driver."""


def team_support_node(state: GraphState) -> dict:
    """Team Support: constructs Arg_change arguing for semantic shift."""
    llm = _get_llm()

    sentences_old = "\n".join(f"  • {s}" for s in state["sentences_old"])
    sentences_new = "\n".join(f"  • {s}" for s in state["sentences_new"])

    response = llm.invoke(
        [
            SystemMessage(content=_SUPPORT_SYSTEM),
            HumanMessage(
                content=_SUPPORT_USER.format(
                    word=state["word"],
                    word_type=state.get("word_type", "word"),
                    t_old=state["t_old"],
                    t_new=state["t_new"],
                    sentences_old=sentences_old,
                    sentences_new=sentences_new,
                )
            ),
        ]
    )
    return {"arg_change": response.content}


# ---------------------------------------------------------------------------
# Team Refuse Node
# ---------------------------------------------------------------------------

_REFUSE_SYSTEM = """You are a computational linguist on **Team Refuse**.
Your sole mission is to build the strongest possible argument that the target word
has NOT undergone genuine diachronic semantic change — i.e., defend the null hypothesis
of semantic stability.

Requirements
------------
1. Identify sentences from the NEW period that **align perfectly** with the core
   meaning established in the OLD period.
2. Argue that any apparently new usages represent **situational polysemy** —
   context-dependent senses that do not constitute a fundamental shift in the word's
   core semantic content.
3. Demonstrate continuity of the word's primary prototypical meaning across periods.
4. Pre-emptively counter broadening/narrowing claims by showing stable semantic features.

Be specific, cite corpus evidence directly, and construct a compelling argument."""

_REFUSE_USER = """Analyze the semantic stability of the word: "{word}" (used as a {word_type})

SENTENCES FROM {t_old}:
{sentences_old}

SENTENCES FROM {t_new}:
{sentences_new}

Build Arg_stable — your counter-argument that NO fundamental semantic shift has \
occurred between {t_old} and {t_new}. Cite specific sentence evidence and argue \
that any new usages are situational polysemy rather than true diachronic change."""


def team_refuse_node(state: GraphState) -> dict:
    """Team Refuse: constructs Arg_stable arguing for semantic stability."""
    llm = _get_llm()

    sentences_old = "\n".join(f"  • {s}" for s in state["sentences_old"])
    sentences_new = "\n".join(f"  • {s}" for s in state["sentences_new"])

    response = llm.invoke(
        [
            SystemMessage(content=_REFUSE_SYSTEM),
            HumanMessage(
                content=_REFUSE_USER.format(
                    word=state["word"],
                    word_type=state.get("word_type", "word"),
                    t_old=state["t_old"],
                    t_new=state["t_new"],
                    sentences_old=sentences_old,
                    sentences_new=sentences_new,
                )
            ),
        ]
    )
    return {"arg_stable": response.content}


# ---------------------------------------------------------------------------
# Judge Node
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """You are an impartial **LLM Judge** specialising in diachronic semantics.
You will evaluate two arguments — one for semantic change and one for semantic stability —
and render a final, structured verdict.

Taxonomy reference
------------------
Change Types (only when CHANGE DETECTED):
  - Generalization  –  meaning broadened to cover more concepts.
  - Specialization  –  meaning narrowed to a more specific domain.
  - Co-hyponymous Transfer  –  shifted to a semantically related but distinct concept at
    the same level of abstraction.

Causal Drivers (only when CHANGE DETECTED):
  - Cultural Shift  –  broad societal, technological, or cultural changes.
  - Linguistic Drift  –  internal linguistic processes (metaphor, metonymy, analogy).

Verdict rules
-------------
- Weigh the COMPARATIVE QUALITY of evidence, not the volume.
- If the evidence for change outweighs stability: verdict = "CHANGE DETECTED".
  Supply change_type, causal_driver, and an estimated break_point_year.
- Otherwise: verdict = "STABLE". Set change_type, causal_driver, and break_point_year to null.
- Always provide detailed reasoning citing both arguments."""

_JUDGE_USER = """Evaluate the following debate about the word "{word}" (used as a {word_type}) \
({t_old} vs. {t_new}):

--- ARGUMENT FOR CHANGE (Team Support) ---
{arg_change}

--- ARGUMENT FOR STABILITY (Team Refuse) ---
{arg_stable}

Based on the comparative weight of evidence, render your final structured verdict."""


def judge_node(state: GraphState) -> dict:
    """LLM Judge: evaluates Arg_change vs Arg_stable and emits a JudgeVerdict."""
    # Use structured output to enforce the strict JSON schema.
    structured_llm = _get_llm(temperature=0.2).with_structured_output(JudgeVerdict)

    verdict: JudgeVerdict = structured_llm.invoke(
        [
            SystemMessage(content=_JUDGE_SYSTEM),
            HumanMessage(
                content=_JUDGE_USER.format(
                    word=state["word"],
                    word_type=state.get("word_type", "word"),
                    t_old=state["t_old"],
                    t_new=state["t_new"],
                    arg_change=state["arg_change"],
                    arg_stable=state["arg_stable"],
                )
            ),
        ]
    )
    return {"verdict": verdict.model_dump()}


# ---------------------------------------------------------------------------
# Multi-Round Rebuttal Nodes
# ---------------------------------------------------------------------------

_REBUTTAL_SUPPORT_SYSTEM = """You are a computational linguist on **Team Support**.
Your goal is to argue that the target word HAS undergone genuine diachronic semantic change.

This is a REBUTTAL round. You have just read Team Refuse's counter-argument claiming the
word is semantically stable. Your job is to:
1. Directly address and refute the weakest points in their argument.
2. Reinforce your strongest evidence with new angles or additional detail.
3. Name the Change Type (Generalization / Specialization / Co-hyponymous Transfer).
4. Name the Causal Driver (Cultural Shift / Linguistic Drift).

Be direct, adversarial, and cite corpus evidence."""

_REBUTTAL_SUPPORT_USER = """Debate round {current_round} of {num_rounds}.
Word: "{word}" (used as a {word_type}) — {t_old} vs {t_new}

--- CORPUS EVIDENCE ---
SENTENCES FROM {t_old}:
{sentences_old}

SENTENCES FROM {t_new}:
{sentences_new}

--- TEAM REFUSE'S LATEST ARGUMENT (what you must rebut) ---
{arg_stable}

Write your rebuttal Arg_change, directly countering Team Refuse's claims above."""


_REBUTTAL_REFUSE_SYSTEM = """You are a computational linguist on **Team Refuse**.
Your goal is to argue that the target word has NOT undergone genuine diachronic semantic change.

This is a REBUTTAL round. You have just read Team Support's counter-argument claiming the
word's meaning has shifted. Your job is to:
1. Directly address and refute the weakest points in their argument.
2. Reinforce your stability evidence with new angles or additional detail.
3. Demonstrate that any apparent shift is situational polysemy, not diachronic change.

Be direct, adversarial, and cite corpus evidence."""

_REBUTTAL_REFUSE_USER = """Debate round {current_round} of {num_rounds}.
Word: "{word}" (used as a {word_type}) — {t_old} vs {t_new}

--- CORPUS EVIDENCE ---
SENTENCES FROM {t_old}:
{sentences_old}

SENTENCES FROM {t_new}:
{sentences_new}

--- TEAM SUPPORT'S LATEST ARGUMENT (what you must rebut) ---
{arg_change}

Write your rebuttal Arg_stable, directly countering Team Support's claims above."""


def rebuttal_support_node(state: GraphState) -> dict:
    """Multi-round: Team Support rebuttal — reads opponent's arg_stable, writes new arg_change."""
    llm = _get_llm()
    sentences_old = "\n".join(f"  • {s}" for s in state["sentences_old"])
    sentences_new = "\n".join(f"  • {s}" for s in state["sentences_new"])

    current_round = state.get("current_round", 1)
    num_rounds = state.get("num_rounds", 1)

    response = llm.invoke(
        [
            SystemMessage(content=_REBUTTAL_SUPPORT_SYSTEM),
            HumanMessage(
                content=_REBUTTAL_SUPPORT_USER.format(
                    current_round=current_round,
                    num_rounds=num_rounds,
                    word=state["word"],
                    word_type=state.get("word_type", "word"),
                    t_old=state["t_old"],
                    t_new=state["t_new"],
                    sentences_old=sentences_old,
                    sentences_new=sentences_new,
                    arg_stable=state.get("arg_stable", ""),
                )
            ),
        ]
    )
    new_arg_change = response.content
    # Append this round's Support argument to the history (keyed for this round).
    history = list(state.get("debate_history", []))
    # Start a new round entry; refuse will complete it.
    history.append({"round": current_round, "arg_change": new_arg_change, "arg_stable": ""})
    return {"arg_change": new_arg_change, "debate_history": history}


def rebuttal_refuse_node(state: GraphState) -> dict:
    """Multi-round: Team Refuse rebuttal — reads opponent's arg_change, writes new arg_stable."""
    llm = _get_llm()
    sentences_old = "\n".join(f"  • {s}" for s in state["sentences_old"])
    sentences_new = "\n".join(f"  • {s}" for s in state["sentences_new"])

    current_round = state.get("current_round", 1)
    num_rounds = state.get("num_rounds", 1)

    response = llm.invoke(
        [
            SystemMessage(content=_REBUTTAL_REFUSE_SYSTEM),
            HumanMessage(
                content=_REBUTTAL_REFUSE_USER.format(
                    current_round=current_round,
                    num_rounds=num_rounds,
                    word=state["word"],
                    word_type=state.get("word_type", "word"),
                    t_old=state["t_old"],
                    t_new=state["t_new"],
                    sentences_old=sentences_old,
                    sentences_new=sentences_new,
                    arg_change=state.get("arg_change", ""),
                )
            ),
        ]
    )
    new_arg_stable = response.content

    # Complete the current round entry in history.
    history = list(state.get("debate_history", []))
    if history and history[-1]["round"] == current_round:
        history[-1]["arg_stable"] = new_arg_stable
    else:
        history.append({"round": current_round, "arg_change": state.get("arg_change", ""), "arg_stable": new_arg_stable})

    return {
        "arg_stable": new_arg_stable,
        "debate_history": history,
        "current_round": current_round + 1,
    }


def should_continue(state: GraphState) -> str:
    """Conditional router: loop for another rebuttal round, or exit to judge.

    Returns
    -------
    "rebuttal_support"  – if more rounds remain (current_round <= num_rounds)
    "judge"             – once all rounds are exhausted
    """
    current_round = state.get("current_round", 1)
    num_rounds = state.get("num_rounds", 1)
    if current_round <= num_rounds:
        return "rebuttal_support"
    return "judge"
