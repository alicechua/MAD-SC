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

import json
import os
import re
import time

from typing import Any
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal

from mad_sc.etymology import fetch_etymology_context, format_etymology_context_for_prompt
from mad_sc.state import EtymologyResult, GraphState, JudgeVerdict

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_DEFAULT_MODEL_OR   = "google/gemini-2.5-flash"
_DEFAULT_MODEL_GAS  = "gemini-2.0-flash-lite"
_DEFAULT_MODEL_VAI  = "gemini-2.5-flash"
_DEFAULT_MODEL_GROQ = "llama3-70b-8192"
# Judge can use a stronger/reasoning model independently of the team agents.
# Set JUDGE_MODEL_GAS / JUDGE_MODEL_OR in .env to override; falls back to the team default.
_JUDGE_MODEL_OR  = os.getenv("JUDGE_MODEL_OR")   # e.g. "google/gemini-2.5-flash"
_JUDGE_MODEL_GAS = os.getenv("JUDGE_MODEL_GAS")  # e.g. "gemini-2.5-flash"
# Inter-call delay to stay under free-tier rate limits
_INTER_CALL_DELAY = float(os.getenv("INTER_CALL_DELAY", "2.0"))

# Fine-grained valid labels (must match JudgeVerdict Literal exactly)
_VALID_LABELS = [
    "Metaphor", "Metonymy", "Analogy", "Generalization",
    "Specialization", "Ellipsis", "Antiphrasis", "Auto-Antonym", "Synecdoche",
]

# Aliases for fuzzy label normalization
_LABEL_ALIASES: dict[str, str] = {
    "metaphor": "Metaphor",
    "metaphorical": "Metaphor",
    "metonym": "Metonymy",
    "metonymic": "Metonymy",
    "analogy": "Analogy",
    "analogical": "Analogy",
    "generali": "Generalization",
    "broadening": "Generalization",
    "speciali": "Specialization",
    "narrowing": "Specialization",
    "ellipsis": "Ellipsis",
    "antiphrasis": "Antiphrasis",
    "auto-antonym": "Auto-Antonym",
    "autantonym": "Auto-Antonym",
    "synecdoche": "Synecdoche",
}


def _get_llm(model: str | None = None, temperature: float = 0.7) -> Any:
    """Instantiate an LLM based on the LLM_BACKEND env variable.

    Backends
    --------
    google_ai_studio  Uses langchain-google-genai + GOOGLE_AI_STUDIO_KEY.
                      Model resolved from DEFAULT_MODEL_GAS env var.
    vertex_ai         Uses langchain-google-genai + VERTEX_AI_KEY (Express mode).
                      Model resolved from DEFAULT_MODEL_VAI env var.
    groq              Uses langchain-groq + GROQ_API_KEY.
                      Model resolved from DEFAULT_MODEL_GROQ env var.
    openrouter        Uses langchain-openai pointed at OpenRouter.
                      Model resolved from DEFAULT_MODEL_OR env var.

    Pass ``model`` explicitly to override the default for a specific call site
    (e.g. the judge using a stronger reasoning model).
    """
    backend = os.getenv("LLM_BACKEND", "openrouter").lower()

    if backend == "google_ai_studio":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_AI_STUDIO_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_AI_STUDIO_KEY is not set in .env")
        resolved_model = model or os.getenv("DEFAULT_MODEL_GAS", _DEFAULT_MODEL_GAS)
        return ChatGoogleGenerativeAI(
            model=resolved_model,
            temperature=temperature,
            google_api_key=api_key,
            max_retries=3,
            max_output_tokens=2048,
        )

    if backend == "vertex_ai":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("VERTEX_AI_KEY")
        if not api_key:
            raise EnvironmentError("LLM_BACKEND=vertex_ai but VERTEX_AI_KEY is not set in .env.")
        resolved_model = model or os.getenv("DEFAULT_MODEL_VAI", _DEFAULT_MODEL_VAI)
        return ChatGoogleGenerativeAI(
            model=resolved_model,
            google_api_key=api_key,
            temperature=temperature,
            vertexai=True,
            location="global",
        )

    if backend == "groq":
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("LLM_BACKEND=groq but GROQ_API_KEY is not set in .env.")
        resolved_model = model or os.getenv("DEFAULT_MODEL_GROQ", _DEFAULT_MODEL_GROQ)
        return ChatGroq(
            model=resolved_model,
            api_key=api_key,
            temperature=temperature,
        )

    # Default: openrouter
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set in .env")
    resolved_model = model or os.getenv("DEFAULT_MODEL_OR", _DEFAULT_MODEL_OR)
    return ChatOpenAI(
        model=resolved_model,
        temperature=temperature,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        max_retries=3,
        max_tokens=2048,
        default_headers={
            "HTTP-Referer": "https://github.com/torontoui/mad-sc",
            "X-Title": "MAD-SC Evaluation",
        }
    )


def _get_judge_llm(temperature: float = 0.1):
    """Return an LLM for the judge, optionally using a stronger reasoning model.

    Reads JUDGE_MODEL_GAS / JUDGE_MODEL_OR from the environment.  If unset,
    falls back to the same model as the team agents.

    Example .env entries
    --------------------
    JUDGE_MODEL_GAS=gemini-2.5-flash   # reasoning model for judge on GAS backend
    JUDGE_MODEL_OR=google/gemini-2.5-flash
    """
    backend = os.getenv("LLM_BACKEND", "openrouter").lower()
    judge_model = _JUDGE_MODEL_GAS if backend == "google_ai_studio" else _JUDGE_MODEL_OR
    return _get_llm(model=judge_model, temperature=temperature)


def _extract_text(response) -> str:
    """Normalise LLM response content to a plain string.

    Gemini (via langchain-google-genai) returns a list of content parts
    such as ``[{'type': 'text', 'text': '...', 'extras': {...}}]``.
    OpenAI-style responses return a plain ``str``.  This helper handles both.
    """
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        ).strip()
    return str(content)


def _robust_invoke(llm, messages, max_retries=10):
    """Invoke the LLM with rate-limit throttle and exponential backoff.

    A pre-call sleep keeps us under free-tier RPM limits.
    On a 429 we additionally apply exponential backoff on top.
    """
    # Rate-limit throttle: wait before every call
    time.sleep(_INTER_CALL_DELAY)

    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "quota" in error_str or "rate" in error_str:
                sleep_time = 15 + (attempt * 15)   # 15s, 30s, 45s …
                print(f"[RETRY] Rate limit hit. Sleeping {sleep_time}s "
                      f"(attempt {attempt+1}/{max_retries})...")
                time.sleep(sleep_time)
            else:
                raise e
    raise Exception(f"Max retries ({max_retries}) exceeded for LLM invocation.")


def _normalize_label(raw: str) -> str | None:
    """Map freeform model text to a canonical Blank's taxonomy label."""
    raw_lower = raw.lower()
    for alias, canonical in _LABEL_ALIASES.items():
        if alias in raw_lower:
            return canonical
    return None


def _parse_verdict_from_text(word: str, raw_text: str) -> JudgeVerdict:
    """
    Fallback parser: extract structured verdict from freeform LLM output.
    Tries JSON extraction first, then exact label scan, then fuzzy normalization.
    """
    # Attempt 1: JSON block in text
    json_match = re.search(r"\{.*?\}", raw_text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return JudgeVerdict(
                word=data.get("word", word),
                verdict=data.get("verdict", "CHANGE DETECTED"),
                change_type=data.get("change_type"),
                causal_driver=data.get("causal_driver"),
                break_point_year=data.get("break_point_year"),
                reasoning=data.get("reasoning", raw_text),
            )
        except Exception:
            pass

    # Attempt 2: exact label anywhere in free text
    for label in _VALID_LABELS:
        if label.lower() in raw_text.lower():
            return JudgeVerdict(
                word=word,
                verdict="CHANGE DETECTED",
                change_type=label,
                causal_driver=None,
                break_point_year=None,
                reasoning=raw_text,
            )

    # Attempt 3: fuzzy normalization
    normalized = _normalize_label(raw_text)
    if normalized:
        return JudgeVerdict(
            word=word,
            verdict="CHANGE DETECTED",
            change_type=normalized,
            causal_driver=None,
            break_point_year=None,
            reasoning=raw_text,
        )

    # Final fallback
    return JudgeVerdict(
        word=word,
        verdict="CHANGE DETECTED",
        change_type=None,
        causal_driver=None,
        break_point_year=None,
        reasoning=raw_text,
    )


# ---------------------------------------------------------------------------
# Grounding Node
# ---------------------------------------------------------------------------

def grounding_node(state: GraphState) -> dict:
    """Pre-debate grounding: compute SED/TD metrics and build a HypothesisDocument.

    Runs the pre_debate_grounding pipeline (WordNet substitutes → BERT SED →
    Time Difference) and stores the formatted prompt block in state.
    If grounding fails for any reason the pipeline continues without it —
    the team nodes simply receive an empty grounding_block.
    """
    try:
        from mad_sc.pre_debate_grounding import run_grounding_pipeline
        doc = run_grounding_pipeline(
            word=state["word"],
            sentences_old=state["sentences_old"],
            sentences_new=state["sentences_new"],
            t_old=state["t_old"],
            t_new=state["t_new"],
            top_k=3,
            n_per_type=2,
            max_sentences=10,
        )
        block = doc.to_prompt_block()
        print(f"[GROUNDING] Hypothesis document generated for '{state['word']}'")
        return {"grounding_block": block}
    except Exception as exc:
        print(f"[GROUNDING] Skipped for '{state['word']}': {exc}")
        return {"grounding_block": ""}


# ---------------------------------------------------------------------------
# Lexicographer Node
# ---------------------------------------------------------------------------

_LEXICOGRAPHER_SYSTEM = """\
You are an expert historical lexicographer specialising in diachronic English semantics.

Your task is to analyse DATED QUOTATIONS from the Oxford English Dictionary and produce \
a precise Definition Dossier for a target word. You must reason directly from the \
provided quote evidence — not from general knowledge.

CRITICAL: Work in two stages before writing your definitions.

STAGE 1 — Quote-by-quote sense analysis
  For each historical quote (pre-1900), write one line identifying the sense of the \
word in that quote. For each modern quote (post-1900), write one line identifying the \
sense of the word in that quote.

STAGE 2 — Synthesis
  What is the DOMINANT sense across historical quotes? That is the OLD SENSE.
  What is the DOMINANT sense across modern quotes? That is the NEW SENSE.
  Is the NEW SENSE genuinely different from the OLD SENSE, or is it the same sense \
in a new context?

MECHANISM DECISION GUIDE (Blank's taxonomy)
  Metaphor    : meaning extended by SIMILARITY across semantic domains
                (e.g. mouse: animal → computer peripheral, both small and scurrying)
  Metonymy    : meaning extended by CONTIGUITY / association in the same domain
                (e.g. bead: prayer-ball → decorative ball, via prayer→ornament context)
  Ellipsis    : a COMPOUND is shortened; the modifier alone takes the compound's meaning
                (e.g. "motor car" → "car"; "canine unit" → "canine")
  Specialization : meaning NARROWS to a subset of the original referents
                (e.g. corn: any grain → specifically maize in American English)
  Generalization : meaning BROADENS to cover a superset of original referents
  Antiphrasis : word takes on the OPPOSITE evaluative polarity in a fixed phrase
  Auto-Antonym: word gains a sense that is the OPPOSITE of its core meaning
                (e.g. bad → "excellent" in slang; sanction → permit AND prohibit)
  Analogy     : meaning shifts by structural parallel with another word/concept
  Synecdoche  : part-for-whole or whole-for-part substitution
  STABLE      : old and new senses are functionally identical — no genuine shift

KEY RULES
  - Ground every definition in what the QUOTES ACTUALLY SHOW, not general knowledge.
  - The corpus period is 1810–2009 (American English). Prioritise shifts visible within \
this window. Ancient shifts (pre-1800) that are already complete by 1810 may appear \
STABLE in corpus data even if the word has an interesting etymology.
  - Be decisive: choose one mechanism. Do not hedge with "possibly" or leave null.
  - Fill synthesis_reasoning FIRST — it is your scratchpad. Your definitions must \
follow from your reasoning."""

_LEXICOGRAPHER_USER = """\
Target word: "{word}"

--- OED evidence ---
{etymology_context}
--- End of evidence ---

Follow the two-stage process:

STAGE 1: For EACH quote above, write one line:
  "[year] HISTORICAL/MODERN — sense: <what the word means in this quote>"

STAGE 2: Synthesise:
  OLD SENSE (dominant across historical quotes): ...
  NEW SENSE (dominant across modern quotes): ...
  Change between periods: ...
  Best-fit mechanism: ... (because ...)

Then produce the structured EtymologyResult fields:
  synthesis_reasoning  : your full Stage 1 + Stage 2 notes
  target_word          : "{word}"
  old_sense_definition : ONE sentence grounded in Stage 2
  new_sense_definition : ONE sentence grounded in Stage 2
  year_of_shift        : year if determinable from the quotes, else null
  mechanism_of_change  : the mechanism from Stage 2"""


def lexicographer_node(state: GraphState) -> dict:
    """Lexicographer Agent: fetches OED/Wiktionary evidence and produces a Definition Dossier.

    Step 1: fetch_etymology_context(word) → OED dated quotes + Wiktionary etymology.
    Step 2: LLM synthesises an EtymologyResult from the evidence via structured output.
    Step 3: to_dossier_block() formats it for injection into team system prompts.

    Falls back gracefully — if fetching or LLM call fails, teams receive no dossier.
    """
    word = state["word"]
    try:
        # Step 1 — fetch real etymological evidence
        ctx = fetch_etymology_context(word)
        etymology_str = format_etymology_context_for_prompt(ctx)

        # Step 2 — LLM synthesises structured EtymologyResult from evidence
        llm = _get_llm(temperature=0.1).with_structured_output(EtymologyResult)
        messages = [
            SystemMessage(content=_LEXICOGRAPHER_SYSTEM),
            HumanMessage(content=_LEXICOGRAPHER_USER.format(
                word=word,
                etymology_context=etymology_str,
            )),
        ]
        result: EtymologyResult = _robust_invoke(llm, messages)
        if result is None:
            raise ValueError("LLM returned None for EtymologyResult")

        dossier = result.to_dossier_block()
        print(f"[LEXICOGRAPHER] Dossier produced for '{word}' "
              f"(source: {ctx['source']}, mechanism: {result.mechanism_of_change}, "
              f"year: {result.year_of_shift})")
        return {"lexicographer_dossier": dossier}

    except Exception as exc:
        print(f"[LEXICOGRAPHER] Skipped for '{word}': {exc}")
        return {"lexicographer_dossier": ""}


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

Requirements
------------
1. Identify specific sentences from the NEW period whose semantics are **incompatible**
   with the dominant meaning established in the OLD period.
2. Classify the shift into EXACTLY ONE Change Type from Blank's taxonomy:
   - **Metaphor**: meaning extended via conceptual *similarity* (resemblance across domains).
   - **Metonymy**: meaning shifted via *contiguity* or *association* (part-for-whole, cause-for-effect, container-for-content).
   - **Analogy**: meaning extended via structural resemblance across semantic domains.
   - **Generalization**: scope *broadened* to cover a wider range of referents.
   - **Specialization**: scope *narrowed* to a specific domain or subset.
   - **Ellipsis**: a compound phrase shortened; meaning transferred to the head noun alone (e.g. "motor car" → "car").
   - **Antiphrasis**: meaning shifted to its *opposite* through ironic or euphemistic usage.
   - **Auto-Antonym**: word acquired a sense directly *opposite* to its original meaning.
   - **Synecdoche**: part-for-whole or whole-for-part meaning transfer.
3. Hypothesize EXACTLY ONE Causal Driver:
   - **Cultural Shift**: driven by broad societal, technological, or cultural changes.
   - **Linguistic Drift**: driven by internal linguistic processes (metaphor, metonymy, analogy)."""

_SUPPORT_USER = """Analyze the word "{word}" (used as a {word_type}) for diachronic semantic change.

SENTENCES FROM {t_old} (OLD period):
{sentences_old}

SENTENCES FROM {t_new} (NEW period):
{sentences_new}

Build Arg_change — your argument that a genuine semantic shift has occurred between \
{t_old} and {t_new}. Cite specific sentence evidence, name the Change Type from \
Blank's taxonomy (Metaphor / Metonymy / Analogy / Generalization / Specialization / \
Ellipsis / Antiphrasis / Auto-Antonym / Synecdoche), and identify the Causal Driver."""


def team_support_node(state: GraphState) -> dict:
    """Team Support: constructs Arg_change arguing for semantic shift."""
    llm = _get_llm()

    sentences_old = "\n".join(f"  • {s}" for s in state["sentences_old"])
    sentences_new = "\n".join(f"  • {s}" for s in state["sentences_new"])

    # Inject lexicographer dossier first (highest priority context), then BERT grounding.
    dossier = state.get("lexicographer_dossier", "") or ""
    system_content = f"{dossier}\n\n{_SUPPORT_SYSTEM}" if dossier else _SUPPORT_SYSTEM

    response = _robust_invoke(
        llm,
        [
            SystemMessage(content=system_content),
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
    return {"arg_change": _extract_text(response)}


# ---------------------------------------------------------------------------
# Team Refuse Node
# ---------------------------------------------------------------------------

_REFUSE_SYSTEM = """You are a computational linguist on **Team Refuse**.
Your goal is to determine whether the target word is semantically stable, and argue
for stability only if the corpus evidence supports it.

Requirements
------------
1. Identify sentences from the NEW period that **align perfectly** with the core
   meaning established in the OLD period.
2. Argue that any apparently new usages represent **situational polysemy** —
   context-dependent senses that do not constitute a fundamental shift in the word's
   core semantic content.
3. Demonstrate continuity of the word's primary prototypical meaning across periods.
4. Pre-emptively counter broadening/narrowing/transfer claims by showing stable semantic features."""

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

    # Inject lexicographer dossier first (highest priority context), then BERT grounding.
    dossier = state.get("lexicographer_dossier", "") or ""
    system_content = f"{dossier}\n\n{_REFUSE_SYSTEM}" if dossier else _REFUSE_SYSTEM

    response = _robust_invoke(
        llm,
        [
            SystemMessage(content=system_content),
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
    return {"arg_stable": _extract_text(response)}


# ---------------------------------------------------------------------------
# Two-stage Judge helpers
# ---------------------------------------------------------------------------

_TRANSFER_TYPES = [
    "Metaphor", "Metonymy", "Analogy",
    "Ellipsis", "Antiphrasis", "Auto-Antonym", "Synecdoche",
]


class _CoarseVerdict(BaseModel):
    """Stage-1 output: coarse category only."""
    coarse_category: Literal["STABLE", "Transfer", "Broadening", "Narrowing"] = Field(
        description=(
            "Coarse semantic-change category.\n"
            "  STABLE     – no genuine diachronic shift; any variation is situational.\n"
            "  Transfer   – meaning moved to a new referent/domain (covers Metaphor, Metonymy,\n"
            "               Analogy, Ellipsis, Antiphrasis, Auto-Antonym, Synecdoche).\n"
            "  Broadening – referential scope widened (Generalization).\n"
            "  Narrowing  – referential scope narrowed (Specialization)."
        )
    )
    confidence: float = Field(
        default=0.75,
        description=(
            "Confidence in the coarse category, 0.0–1.0. "
            "Use ≤0.6 for ambiguous/contested evidence, ≥0.85 for clear-cut cases."
        ),
    )
    reasoning: str = Field(description="Brief justification for the coarse category choice.")


# ── Stage 1: Coarse prompt ───────────────────────────────────────────────────

_JUDGE_COARSE_SYSTEM = """\
You are an expert judge in diachronic semantics. Given two debaters' arguments about \
a target word, decide the COARSE category of semantic change.

Categories
----------
STABLE     — No genuine change. Any variation is situational polysemy or register, \
             not a true shift in the word's core meaning.
Broadening — The word's referential scope WIDENED to cover more referents.
             Diagnostic: the old meaning is a proper subset of the new meaning.
Narrowing  — The word's referential scope NARROWED to a more specific subset.
             Diagnostic: the new meaning is a proper subset of the old meaning.
Transfer   — The meaning was transferred to a new referent or domain via some \
             mechanism (resemblance, association, irony, shortening, etc.).
             Diagnostic: the new referent/domain is NOT simply a wider or narrower \
             version of the old — it is a qualitatively different one.

Few-shot examples
-----------------
Word: "dog"  →  BROADENING
  OLD: male canine only  |  NEW: any canine regardless of sex
  Why: old meaning is a subset of new meaning (scope widened).

Word: "corn"  →  NARROWING
  OLD: any cereal grain  |  NEW: maize specifically
  Why: new meaning is a subset of old meaning (scope narrowed).

Word: "mouse"  →  TRANSFER
  OLD: small rodent  |  NEW: computer pointing device
  Why: rodent ≠ subset/superset of computer device; it is a qualitatively different \
  referent reached via resemblance.

Word: "bead"  →  TRANSFER
  OLD: a prayer (counting prayers on a rosary)  |  NEW: small sphere object
  Why: prayer ≠ subset/superset of sphere; meaning transferred via association.

Word: "water"  →  STABLE
  OLD: the liquid H₂O  |  NEW: the liquid H₂O
  Why: core referential scope unchanged across contexts.

Decision rules
--------------
1. If BOTH arguments agree there is no change → lean STABLE.
2. If the new referents are strictly a larger set → BROADENING.
3. If the new referents are strictly a smaller set → NARROWING.
4. If the new referent is qualitatively different (different domain, reversed polarity, \
   ironic inversion, compound shortening) → TRANSFER.
5. Weigh QUALITY of evidence, not volume.

Output ONLY valid JSON matching this schema:
{"coarse_category": "<STABLE|Transfer|Broadening|Narrowing>", "confidence": <0.0-1.0>, "reasoning": "<brief justification>"}\
"""

_JUDGE_COARSE_USER = """\
Evaluate the debate about the word "{word}" ({t_old} vs. {t_new}).

--- ARGUMENT FOR CHANGE ---
{arg_change}

--- ARGUMENT FOR STABILITY ---
{arg_stable}

Which COARSE category best describes the semantic trajectory? \
Output ONLY a valid JSON object.\
"""


# ── Stage 2: Fine-grained Transfer prompt ───────────────────────────────────

_JUDGE_TRANSFER_SYSTEM = """\
You are an expert judge in diachronic semantics. Stage 1 already determined that \
the word underwent a TRANSFER — meaning moved to a qualitatively different referent \
or domain. Your task in Stage 2 is to identify the EXACT MECHANISM of that transfer.

Transfer mechanisms
-------------------
Metaphor    – New sense connected to old via CONCEPTUAL RESEMBLANCE across domains.
              The two referents share perceptual or structural similarity but belong
              to different conceptual domains.
              ► Example: "mouse" (rodent) → "mouse" (computer device)
                Reason: shape/tail resembles the cable; different domains (nature vs tech).

Metonymy    – New sense connected to old via CONTIGUITY or ASSOCIATION within the
              same conceptual world (part-for-whole, cause-for-effect,
              container-for-content, instrument-for-action).
              ► Example: "bead" (prayer) → "bead" (small sphere)
                Reason: beads USED FOR prayer → the physical object itself.
              ► Example: "sweat" (perspiration) → "work hard"
                Reason: sweat is the PHYSICAL RESULT of hard work (effect-for-cause).
              ► Example: "shall" (deontic obligation) → "will" (temporal future)
                Reason: modal shift within the same functional domain (futurity).

Analogy     – New sense connected via STRUCTURAL resemblance across semantic domains
              (not perceptual). The mapping is abstract/functional rather than visual.
              ► Example: "fast" (firmly fixed) → "moving quickly"
                Reason: both senses share the abstract idea of "no yielding/resistance"
                applied analogically from physical anchoring to speed.
              ► Example: "hardly" (boldly/vigorously) → "scarcely/barely"
                Reason: structural shift from degree-of-effort to degree-of-occurrence.

Ellipsis    – A COMPOUND phrase was shortened; the head noun inherited the full meaning
              of the compound.
              ► Example: "motor car" → "car" (car inherited "motor car" meaning).
              ► Example: "canine tooth" → "canine" (tooth sense absorbed).
              Diagnostic: always check if the word was historically part of a longer phrase.

Antiphrasis – Meaning shifted to its OPPOSITE through IRONIC or EUPHEMISTIC usage.
              The shift happened gradually via sarcastic/ironic context.
              ► Example: "perfect lady" (noble woman) → (prostitute via irony)
              Distinguish from Auto-Antonym: the mechanism is social/ironic, not internal.

Auto-Antonym – Word acquired a sense DIRECTLY OPPOSITE to its original meaning without
               irony — a straightforward polarity reversal driven by slang/register shift.
               ► Example: "bad" (evil/wicked) → "excellent" (slang)
               Distinguish from Antiphrasis: no irony — speakers genuinely use it positively.

Synecdoche  – PART-FOR-WHOLE or WHOLE-FOR-PART meaning transfer within the same domain.
              ► Example: "sail" (piece of canvas) → "sailing vessel" (whole ship).

Critical disambiguation rules
------------------------------
Metaphor vs Metonymy (most common confusion):
  • Ask: Are the old and new referents part of the SAME conceptual world / event frame?
    YES → Metonymy.  NO (leap to a different domain) → Metaphor.
  • Ask: Is the link PERCEPTUAL SIMILARITY or FUNCTIONAL ASSOCIATION?
    Similarity → Metaphor.  Association/contiguity → Metonymy.

Ellipsis vs Specialization (both narrow):
  • Ellipsis requires a historically attested COMPOUND. If no compound existed → not Ellipsis.

Antiphrasis vs Auto-Antonym:
  • Antiphrasis: the positive form is used IRONICALLY to mean the negative (or vice versa).
  • Auto-Antonym: the word GENUINELY means both (or has shifted) — no ironic framing needed.

Output ONLY valid JSON matching the JudgeVerdict schema.\
"""

_JUDGE_TRANSFER_USER = """\
The word "{word}" ({t_old} vs. {t_new}) has been classified as a TRANSFER in Stage 1.

Stage 1 reasoning: {coarse_reasoning}

--- ARGUMENT FOR CHANGE ---
{arg_change}

--- ARGUMENT FOR STABILITY ---
{arg_stable}

Now identify the EXACT transfer mechanism (Metaphor / Metonymy / Analogy / Ellipsis / \
Antiphrasis / Auto-Antonym / Synecdoche). Work through the disambiguation rules, then \
output ONLY a valid JSON object with: word, verdict ("CHANGE DETECTED"), change_type, \
causal_driver, break_point_year, confidence (0.0-1.0), reasoning.\
"""


# ---------------------------------------------------------------------------
# Judge Node
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """You are an impartial **LLM Judge** specialising in diachronic semantics.

Your task is to evaluate the full debate and render a final, structured verdict.

Taxonomy reference (Blank's full taxonomy)
------------------------------------------
Fine-grained Change Types (pick EXACTLY ONE when CHANGE DETECTED):
  Metaphor      – meaning extended via conceptual similarity (resemblance across domains).
                  e.g. "mouse" (animal) → "mouse" (computer device)
  Metonymy      – meaning shifted via contiguity or association.
                  e.g. "bead" (prayer) → "bead" (small sphere object)
  Analogy       – meaning extended via structural resemblance across semantic domains.
  Generalization – scope broadened to cover a wider range of referents.
                   e.g. "dog" (male canine) → "dog" (any canine)
  Specialization – scope narrowed to a specific domain or subset.
                   e.g. "corn" (any grain) → "corn" (maize specifically)
  Ellipsis      – compound phrase shortened; meaning of the whole transferred to head noun.
                   e.g. "motor car" → "car"
  Antiphrasis   – meaning shifted to its opposite through ironic/euphemistic usage.
                   e.g. "perfect lady" → used ironically for unladylike behavior
  Auto-Antonym  – word acquired a sense directly opposite to its original meaning.
                   e.g. "bad" → also meaning "good" (slang reversal)
  Synecdoche    – part-for-whole or whole-for-part meaning transfer.

Diagnostic checklist (work through before classifying)
-------------------------------------------------------
1. Has the REFERENTIAL SCOPE changed?
   → Broader = Generalization | Narrower = Specialization or Ellipsis
2. Is the new sense connected to the old via SPATIAL, TEMPORAL, or CAUSAL contiguity?
   → YES = Metonymy
3. Is the new sense connected via CONCEPTUAL RESEMBLANCE across domains?
   → YES = Metaphor or Analogy
4. Has the meaning REVERSED polarity or become ironic?
   → YES = Antiphrasis or Auto-Antonym
5. Is the word a SHORTENED FORM of a longer compound?
   → YES = Ellipsis

Verdict rules
-------------
- Weigh the COMPARATIVE QUALITY of evidence, not the volume.
- If the evidence for change outweighs stability: verdict = "CHANGE DETECTED".
  Supply change_type (from the taxonomy above), causal_driver, and break_point_year.
- Otherwise: verdict = "STABLE". Set change_type, causal_driver, and break_point_year to null.
- Always work through the diagnostic checklist before choosing change_type.

IMPORTANT: You MUST output a valid JSON object matching this schema exactly:
{
  "word": "<the target word>",
  "verdict": "CHANGE DETECTED" | "STABLE",
  "change_type": "<one of the 9 labels above, or null if STABLE>",
  "causal_driver": "Cultural Shift" | "Linguistic Drift" | null,
  "break_point_year": <integer year or null>,
  "confidence": <float 0.0-1.0 — your confidence in this verdict based on evidence quality>,
  "reasoning": "<your full reasoning>"
}"""

# Used for single-round (parallel) debates — only the two opening arguments are available.
_JUDGE_USER = """Evaluate the following debate about the word "{word}" (used as a {word_type}) \
({t_old} vs. {t_new}):

--- ARGUMENT FOR CHANGE (Team Support) ---
{arg_change}

--- ARGUMENT FOR STABILITY (Team Refuse) ---
{arg_stable}

Work through the diagnostic checklist step-by-step, then output ONLY a valid JSON \
object with your final structured verdict. Use the exact JSON schema from your instructions."""

# Used for multi-round debates — the judge receives the complete transcript.
_JUDGE_MULTI_USER = """Evaluate the complete {num_rounds}-round debate about the word \
"{word}" (used as a {word_type}), {t_old} vs. {t_new}.

{history}

The closing statements represent each team's final position after reading all of \
the rebuttals. Base your verdict on the full transcript above, not just \
the final round. Apply the verdict rules in your system prompt strictly."""


def _run_coarse_stage(word: str, t_old: str, t_new: str,
                      arg_change: str, arg_stable: str,
                      lexicographer_dossier: str = "",
                      debate_history: list = None,
                      num_rounds: int = 1,
                      word_type: str = "word") -> _CoarseVerdict | None:
    """Stage 1: classify into STABLE / Transfer / Broadening / Narrowing."""
    coarse_system = (
        f"{lexicographer_dossier}\n\n{_JUDGE_COARSE_SYSTEM}"
        if lexicographer_dossier else _JUDGE_COARSE_SYSTEM
    )
    
    if debate_history and num_rounds > 1:
        history_text = _format_debate_history(debate_history)
        user_prompt = _JUDGE_MULTI_USER.format(
            num_rounds=num_rounds,
            word=word, t_old=t_old, t_new=t_new,
            word_type=word_type,
            history=history_text,
        )
    else:
        user_prompt = _JUDGE_COARSE_USER.format(
            word=word, t_old=t_old, t_new=t_new,
            arg_change=arg_change, arg_stable=arg_stable,
        )

    messages = [
        SystemMessage(content=coarse_system),
        HumanMessage(content=user_prompt),
    ]
    try:
        llm = _get_judge_llm(temperature=0.1).with_structured_output(_CoarseVerdict)
        result = _robust_invoke(llm, messages)
        if result is not None:
            print(f"[JUDGE-S1] '{word}' → {result.coarse_category}")
            return result
    except Exception as e:
        print(f"[JUDGE-S1] Structured output failed for '{word}': {e}. Trying raw.")

    # Fallback: raw text parse for coarse category
    raw_llm = _get_judge_llm(temperature=0.1)
    messages_raw = [
        SystemMessage(content=coarse_system),
        HumanMessage(content=user_prompt + "\n\nRespond with a JSON object: {\"coarse_category\": \"...\", \"reasoning\": \"...\"}"),
    ]
    try:
        resp = _robust_invoke(raw_llm, messages_raw)
        text = _extract_text(resp)
        m = re.search(r'"coarse_category"\s*:\s*"(STABLE|Transfer|Broadening|Narrowing)"', text)
        if m:
            cat = m.group(1)
            print(f"[JUDGE-S1-fallback] '{word}' → {cat}")
            return _CoarseVerdict(coarse_category=cat, reasoning=text)
    except Exception as e:
        print(f"[JUDGE-S1] Raw fallback also failed for '{word}': {e}")
    return None


def _run_transfer_stage(word: str, t_old: str, t_new: str,
                        arg_change: str, arg_stable: str,
                        coarse_reasoning: str,
                        lexicographer_dossier: str = "") -> JudgeVerdict:
    """Stage 2: identify exact Transfer mechanism."""
    transfer_system = (
        f"{lexicographer_dossier}\n\n{_JUDGE_TRANSFER_SYSTEM}"
        if lexicographer_dossier else _JUDGE_TRANSFER_SYSTEM
    )
    messages = [
        SystemMessage(content=transfer_system),
        HumanMessage(content=_JUDGE_TRANSFER_USER.format(
            word=word, t_old=t_old, t_new=t_new,
            arg_change=arg_change, arg_stable=arg_stable,
            coarse_reasoning=coarse_reasoning,
        )),
    ]
    try:
        llm = _get_judge_llm(temperature=0.2).with_structured_output(JudgeVerdict)
        verdict = _robust_invoke(llm, messages)
        if verdict is not None and verdict.change_type in _TRANSFER_TYPES:
            print(f"[JUDGE-S2] '{word}' → {verdict.change_type}")
            return verdict
    except Exception as e:
        print(f"[JUDGE-S2] Structured output failed for '{word}': {e}. Falling back.")

    # Fallback: raw text
    raw_llm = _get_judge_llm(temperature=0.1)
    resp = _robust_invoke(raw_llm, messages)
    verdict = _parse_verdict_from_text(word, _extract_text(resp))
    print(f"[JUDGE-S2-fallback] '{word}' → {verdict.change_type}")
    return verdict


def judge_node(state: GraphState) -> dict:
    """LLM Judge: two-stage coarse-then-fine verdict.

    Stage 1 — Coarse: STABLE / Transfer / Broadening / Narrowing.
    Stage 2 — Fine:   Transfer → exact mechanism (Metaphor / Metonymy / …).
               Broadening → Generalization (direct, no second call).
               Narrowing  → Specialization (direct, no second call).
    """
    word = state["word"]
    t_old, t_new = state["t_old"], state["t_new"]
    arg_change, arg_stable = state["arg_change"], state["arg_stable"]
    dossier = state.get("lexicographer_dossier", "") or ""
    debate_history = state.get("debate_history", [])
    num_rounds = state.get("num_rounds", 1)
    word_type = state.get("word_type", "word")

    # ── Stage 1: Coarse ──────────────────────────────────────────────────────
    coarse = _run_coarse_stage(word, t_old, t_new, arg_change, arg_stable,
                               lexicographer_dossier=dossier,
                               debate_history=debate_history,
                               num_rounds=num_rounds,
                               word_type=word_type)

    if coarse is None or coarse.coarse_category == "STABLE":
        return {"verdict": JudgeVerdict(
            word=word, verdict="STABLE",
            change_type=None, causal_driver=None,
            break_point_year=None,
            confidence=coarse.confidence if coarse else 0.5,
            reasoning=coarse.reasoning if coarse else "Stage 1 failed; defaulting to STABLE.",
        ).model_dump()}

    if coarse.coarse_category == "Broadening":
        print(f"[JUDGE] '{word}' → Broadening → Generalization (direct)")
        return {"verdict": JudgeVerdict(
            word=word, verdict="CHANGE DETECTED",
            change_type="Generalization",
            causal_driver="Linguistic Drift",
            break_point_year=None,
            confidence=coarse.confidence,
            reasoning=coarse.reasoning,
        ).model_dump()}

    if coarse.coarse_category == "Narrowing":
        print(f"[JUDGE] '{word}' → Narrowing → Specialization (direct)")
        return {"verdict": JudgeVerdict(
            word=word, verdict="CHANGE DETECTED",
            change_type="Specialization",
            causal_driver="Linguistic Drift",
            break_point_year=None,
            confidence=coarse.confidence,
            reasoning=coarse.reasoning,
        ).model_dump()}

    # ── Stage 2: Fine-grained Transfer ───────────────────────────────────────
    verdict = _run_transfer_stage(
        word, t_old, t_new, arg_change, arg_stable, coarse.reasoning,
        lexicographer_dossier=dossier,
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
                    arg_change=state.get("arg_change", ""),
                )
            ),
        ]
    )
    closing = _extract_text(response)

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
                    arg_stable=state.get("arg_stable", ""),
                )
            ),
        ]
    )
    closing = _extract_text(response)

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
