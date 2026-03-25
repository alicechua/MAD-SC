"""LangGraph node functions for the MAD-SC tri-agent debate pipeline.

Nodes
-----
team_support_node    –  Team Support: argues that semantic change HAS occurred.
team_refuse_node     –  Team Refuse: argues that semantic change has NOT occurred.
judge_node           –  LLM Judge: evaluates arguments and renders a structured verdict.
closing_refuse_node  –  Team Refuse closing statement (multi-round only).
closing_support_node –  Team Support closing statement (multi-round only, last word fix).
rebuttal_support_node / rebuttal_refuse_node  –  adversarial rebuttal rounds.
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
# Shared helpers
# ---------------------------------------------------------------------------

def _extract_text(content: Any) -> str:
    """Normalise LLM response content to a plain string.

    Gemini can return either a bare string or a list of content-block dicts
    (``[{"type": "text", "text": "…"}]``).  Both are handled transparently.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif hasattr(item, "text"):
                parts.append(item.text)
        return "\n".join(parts)
    return str(content)


def _format_debate_history(debate_history: list[dict]) -> str:
    """Render all rounds of the debate history as readable text for the judge."""
    if not debate_history:
        return "(no debate history available)"
    blocks: list[str] = []
    for entry in debate_history:
        r = entry.get("round", "?")
        arg_c = _extract_text(entry.get("arg_change", ""))
        arg_s = _extract_text(entry.get("arg_stable", ""))
        if arg_c:
            blocks.append(f"=== Round {r} — Team Support ===\n{arg_c}")
        if arg_s:
            blocks.append(f"=== Round {r} — Team Refuse ===\n{arg_s}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Team Support Node
# ---------------------------------------------------------------------------

_SUPPORT_SYSTEM = """You are a computational linguist on **Team Support**.
Your goal is to determine whether the target word has undergone genuine diachronic
semantic change, and argue for change only if the corpus evidence supports it.

STRICT RULES — the judge will penalise violations:
1. Use ONLY the provided corpus sentences. Do NOT cite dictionaries, etymologies,
   historical background knowledge, or any source outside the given sentences.
2. First establish the DOMINANT OLD sense from the OLD sentences alone.
3. Argue for change only if you can show at least two NEW sentences that the OLD
   sense cannot comfortably explain.
4. The following do NOT constitute semantic change on their own:
   - Change in topic or subject matter (e.g., horses → cars)
   - Change in register or genre (formal vs. informal text)
   - New collocations or compound formations
   - Change in cultural salience, frequency, or discourse domain
5. A valid change claim MUST involve at least one of:
   - A new denotation or referent class not licensed by the OLD sense
   - Loss or gain of a named semantic feature (e.g., [+animate], [+physical])
   - Changed selectional restrictions not explainable by metaphor/metonymy
   - A sense that was clearly marginal or absent in OLD becoming dominant in NEW
6. Identify EXACTLY ONE Change Type: Generalization, Specialization, or
   Co-hyponymous Transfer.
7. Identify EXACTLY ONE Causal Driver: Cultural Shift or Linguistic Drift.
8. End with a CONFIDENCE score (0–100) reflecting your honest appraisal of the evidence."""

_SUPPORT_USER = """Analyze the word "{word}" (used as a {word_type}) for diachronic semantic change.

SENTENCES FROM {t_old} (OLD period):
{sentences_old}

SENTENCES FROM {t_new} (NEW period):
{sentences_new}

Structure your Arg_change as follows:
1. **OLD dominant sense** — derived from OLD sentences only; name 1–2 core semantic features.
2. **Evidence for change** — quote 2–3 NEW sentences and explain, feature by feature, why
   the OLD sense does not fully account for them.
3. **Best counterevidence** — quote 1–2 sentences (OLD or NEW) that appear to continue the
   OLD sense; explain why they do not undermine your claim.
4. **Why this is not topic/register/polysemy** — a specific, evidence-based explanation.
5. **Change Type & Causal Driver**
6. **Confidence: NN/100**"""


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
Your goal is to determine whether the target word is semantically stable, and argue
for stability only if the corpus evidence supports it.

STRICT RULES — the judge will penalise violations:
1. Use ONLY the provided corpus sentences. Do NOT cite dictionaries, etymologies,
   historical background knowledge, or any source outside the given sentences.
2. Identify the OLD core sense from OLD sentences alone; explicitly name its key
   semantic features (e.g., [+animate], [+physical], [+bounded]).
3. For every apparently novel NEW use, you MUST:
   a. Name the shared semantic feature(s) present in BOTH the old and new usage.
   b. Point to at least one OLD sentence that already licenses that feature.
4. "Situational polysemy" is a valid defence only when you supply (3a) and (3b).
   Merely asserting "situational polysemy" without evidence is not acceptable.
5. Do NOT ignore Team Support's strongest evidence — address it directly.
6. If a NEW use genuinely introduces a referent class or semantic feature absent
   from the OLD corpus, acknowledge it as serious counterevidence rather than
   dismissing it.
7. End with a CONFIDENCE score (0–100) reflecting your honest appraisal."""

_REFUSE_USER = """Analyze the word "{word}" (used as a {word_type}) for semantic stability.

SENTENCES FROM {t_old} (OLD period):
{sentences_old}

SENTENCES FROM {t_new} (NEW period):
{sentences_new}

Structure your Arg_stable as follows:
1. **OLD core sense** — from OLD sentences only; list 2–3 specific semantic features.
2. **NEW uses consistent with that sense** — quote 2–3 NEW sentences; for each, name
   the shared feature and cite the OLD sentence that already licenses it.
3. **Strongest evidence for change** — quote the 1–2 NEW sentences that most challenge
   stability; do not ignore them.
4. **Why they are still compatible with the OLD sense** — name the specific feature
   that carries over; distinguish between referent shift and sense shift.
5. **Why the difference is polysemy/register, not diachronic change**
6. **Confidence: NN/100**"""


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

Your task is to evaluate the full debate and render a final, structured verdict.

Taxonomy reference
------------------
Change Types (only when CHANGE DETECTED):
  - Generalization  –  meaning broadened to cover more concepts or referent classes.
  - Specialization  –  meaning narrowed to a more specific domain or referent class.
  - Co-hyponymous Transfer  –  shifted to a semantically related but distinct concept at
    the same level of abstraction.

Causal Drivers (only when CHANGE DETECTED):
  - Cultural Shift  –  broad societal, technological, or cultural changes.
  - Linguistic Drift  –  internal linguistic processes (metaphor, metonymy, analogy).

Verdict rules
-------------
1. Weigh EVIDENCE QUALITY, not rhetorical volume or which team sounded more confident.
2. Discount arguments based on topic/register/domain shifts alone — those do not
   constitute semantic change unless accompanied by a loss or gain of core semantic
   features or a genuinely new referent class.
3. Discount arguments that cite dictionaries, etymologies, or facts not in the corpus
   sentences — both teams are restricted to corpus evidence.
4. Discount "situational polysemy" defences that assert continuity without naming the
   shared semantic feature or citing an OLD-period analogue for the allegedly new use.
5. If genuine new denotation, lost/gained features, or changed selectional restrictions
   are demonstrated: verdict = "CHANGE DETECTED". Supply change_type, causal_driver,
   and an estimated break_point_year.
6. Otherwise: verdict = "STABLE". Set change_type, causal_driver, break_point_year to null.
7. Always provide detailed reasoning that cites specific sentences from the debate."""

# Used for single-round (parallel) debates — only the two opening arguments are available.
_JUDGE_USER = """Evaluate the following debate about the word "{word}" (used as a {word_type}) \
({t_old} vs. {t_new}):

--- ARGUMENT FOR CHANGE (Team Support) ---
{arg_change}

--- ARGUMENT FOR STABILITY (Team Refuse) ---
{arg_stable}

Based on the comparative quality of evidence, render your final structured verdict."""

# Used for multi-round debates — the judge receives the complete transcript.
_JUDGE_MULTI_USER = """Evaluate the complete {num_rounds}-round debate about the word \
"{word}" (used as a {word_type}), {t_old} vs. {t_new}.

{history}

The closing statements represent each team's final position after reading all of \
the rebuttals. Base your verdict on the full transcript above, not just \
the final round. Apply the verdict rules in your system prompt strictly."""


def judge_node(state: GraphState) -> dict:
    """LLM Judge: evaluates the full debate and emits a JudgeVerdict.

    In multi-round mode the judge receives the complete debate history so it
    can weigh evidence across all rounds, not just the final overwritten
    arg_change / arg_stable pair.
    """
    structured_llm = _get_llm(temperature=0.2).with_structured_output(JudgeVerdict)

    debate_history = state.get("debate_history", [])

    if debate_history:
        # Multi-round path: pass full transcript.
        history_text = _format_debate_history(debate_history)
        user_content = _JUDGE_MULTI_USER.format(
            num_rounds=state.get("num_rounds", len(debate_history)),
            word=state["word"],
            word_type=state.get("word_type", "word"),
            t_old=state["t_old"],
            t_new=state["t_new"],
            history=history_text,
        )
    else:
        # Single-round path: original behaviour.
        user_content = _JUDGE_USER.format(
            word=state["word"],
            word_type=state.get("word_type", "word"),
            t_old=state["t_old"],
            t_new=state["t_new"],
            arg_change=_extract_text(state.get("arg_change", "")),
            arg_stable=_extract_text(state.get("arg_stable", "")),
        )

    verdict: JudgeVerdict = structured_llm.invoke(
        [
            SystemMessage(content=_JUDGE_SYSTEM),
            HumanMessage(content=user_content),
        ]
    )
    return {"verdict": verdict.model_dump()}


# ---------------------------------------------------------------------------
# Multi-Round Rebuttal Nodes
# ---------------------------------------------------------------------------

_REBUTTAL_SUPPORT_SYSTEM = """You are a computational linguist on **Team Support**.
Your goal is to argue that the target word HAS undergone genuine diachronic semantic change.

This is a REBUTTAL round. Rules:
1. Use ONLY the provided corpus sentences — no dictionaries, etymologies, or outside facts.
2. Identify the single strongest point in Team Refuse's argument and rebut it with a
   specific corpus sentence that contradicts their claimed continuity.
3. Do NOT merely reassert your opening claim — advance the argument with new evidence
   or a tighter feature-level analysis.
4. Restate Change Type and Causal Driver concisely at the end.
5. Focus on whether core semantic FEATURES have changed, not just topics or contexts."""

_REBUTTAL_SUPPORT_USER = """Rebuttal round {current_round} of {num_rounds}.
Word: "{word}" ({word_type}) — {t_old} vs {t_new}

--- CORPUS SENTENCES ---
OLD ({t_old}):
{sentences_old}

NEW ({t_new}):
{sentences_new}

--- TEAM REFUSE'S LATEST ARGUMENT (rebut this) ---
{arg_stable}

Write your rebuttal. Lead with Team Refuse's weakest claim, refute it with a specific
corpus sentence, then reinforce your strongest feature-level evidence for change."""


_REBUTTAL_REFUSE_SYSTEM = """You are a computational linguist on **Team Refuse**.
Your goal is to argue that the target word has NOT undergone genuine diachronic semantic change.

This is a REBUTTAL round. Rules:
1. Use ONLY the provided corpus sentences — no dictionaries, etymologies, or outside facts.
2. Identify the single strongest piece of evidence in Team Support's argument and address
   it directly: name the shared semantic feature that bridges OLD and NEW usage, and cite
   an OLD sentence that already licenses that feature.
3. Do NOT merely re-assert that the word is stable — show the shared feature explicitly.
4. If Team Support cited a NEW sentence that genuinely has no OLD analogue, acknowledge
   it but explain why it represents polysemy rather than a new lexical sense.
5. Do not invoke "situational polysemy" without naming the shared feature."""

_REBUTTAL_REFUSE_USER = """Rebuttal round {current_round} of {num_rounds}.
Word: "{word}" ({word_type}) — {t_old} vs {t_new}

--- CORPUS SENTENCES ---
OLD ({t_old}):
{sentences_old}

NEW ({t_new}):
{sentences_new}

--- TEAM SUPPORT'S LATEST ARGUMENT (rebut this) ---
{arg_change}

Write your rebuttal. Lead with Team Support's strongest sentence-level evidence, name
the semantic feature that persists from OLD to NEW, then cite an OLD analogue."""


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


# ---------------------------------------------------------------------------
# Closing Refuse Node (multi-round only)
# ---------------------------------------------------------------------------

_CLOSING_REFUSE_SYSTEM = """You are a computational linguist on **Team Refuse**.
You are delivering a concise CLOSING STATEMENT after all rebuttal rounds.

Rules:
1. Use ONLY the provided corpus sentences — no dictionaries, etymologies, or outside facts.
2. Summarise your 1–2 strongest corpus-grounded evidence points for stability.
3. Directly counter Team Support's single most effective argument in 2–3 sentences,
   citing a specific corpus sentence.
4. Keep the closing under 200 words — brevity and precision beat length."""

_CLOSING_REFUSE_USER = """Final closing statement for "{word}" ({word_type}), {t_old} vs {t_new}.

Team Support's final argument (what you must address):
{arg_change}

Write a concise closing statement (≤200 words) using only corpus evidence."""


def closing_refuse_node(state: GraphState) -> dict:
    """Team Refuse closing statement — runs after all rebuttals, before Team Support closing."""
    llm = _get_llm()

    response = llm.invoke(
        [
            SystemMessage(content=_CLOSING_REFUSE_SYSTEM),
            HumanMessage(
                content=_CLOSING_REFUSE_USER.format(
                    word=state["word"],
                    word_type=state.get("word_type", "word"),
                    t_old=state["t_old"],
                    t_new=state["t_new"],
                    arg_change=_extract_text(state.get("arg_change", "")),
                )
            ),
        ]
    )
    closing = response.content

    history = list(state.get("debate_history", []))
    history.append({"round": "closing", "arg_change": "", "arg_stable": closing})
    return {"arg_stable": closing, "debate_history": history}


# ---------------------------------------------------------------------------
# Closing Support Node  (multi-round only)
# ---------------------------------------------------------------------------

_CLOSING_SUPPORT_SYSTEM = """You are a computational linguist on **Team Support**.
You are delivering a concise CLOSING STATEMENT after all rebuttal rounds.

Rules:
1. Use ONLY the provided corpus sentences — no dictionaries, etymologies, or outside facts.
2. Summarise your 1–2 strongest corpus-grounded evidence points for change.
3. Directly counter Team Refuse's single most effective argument in 2–3 sentences,
   citing a specific corpus sentence.
4. Restate Change Type and Causal Driver in one line.
5. Keep the closing under 200 words — brevity and precision beat length."""

_CLOSING_SUPPORT_USER = """Final closing statement for "{word}" ({word_type}), {t_old} vs {t_new}.

Team Refuse's final argument (what you must address):
{arg_stable}

Write a concise closing statement (≤200 words) using only corpus evidence."""


def closing_support_node(state: GraphState) -> dict:
    """Team Support closing statement — runs after all Refuse rebuttals.

    This node exists solely to neutralise the last-word advantage that Team
    Refuse accumulates in the default rebuttal loop (Support → Refuse × N).
    The closing is recorded in debate_history and becomes the ``arg_change``
    that the judge receives as Support's final position.
    """
    llm = _get_llm()

    response = llm.invoke(
        [
            SystemMessage(content=_CLOSING_SUPPORT_SYSTEM),
            HumanMessage(
                content=_CLOSING_SUPPORT_USER.format(
                    word=state["word"],
                    word_type=state.get("word_type", "word"),
                    t_old=state["t_old"],
                    t_new=state["t_new"],
                    arg_stable=_extract_text(state.get("arg_stable", "")),
                )
            ),
        ]
    )
    closing = response.content

    history = list(state.get("debate_history", []))
    if history and history[-1].get("round") == "closing":
        history[-1]["arg_change"] = closing
    else:
        history.append({"round": "closing", "arg_change": closing, "arg_stable": ""})
        
    return {"arg_change": closing, "debate_history": history}


def should_continue(state: GraphState) -> str:
    """Conditional router: loop for another rebuttal round, or exit to closing.

    Returns
    -------
    "rebuttal_support"  – if more rounds remain (current_round <= num_rounds)
    "judge"             – once all rounds are exhausted (graph re-maps to closing)
    """
    current_round = state.get("current_round", 1)
    num_rounds = state.get("num_rounds", 1)
    if current_round <= num_rounds:
        return "rebuttal_support"
    return "judge"
