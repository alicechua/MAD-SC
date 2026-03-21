"""LangGraph node functions for the MAD-SC tri-agent debate pipeline.

Nodes
-----
team_support_node  –  Team Support: argues that semantic change HAS occurred.
team_refuse_node   –  Team Refuse: argues that semantic change has NOT occurred.
judge_node         –  LLM Judge: evaluates both arguments and renders a structured verdict.
"""

import json
import os
import re
import time

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal

from mad_sc.state import GraphState, JudgeVerdict

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_DEFAULT_MODEL_OR  = "google/gemini-2.5-flash"
_DEFAULT_MODEL_GAS = "gemini-2.0-flash-lite"
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


def _get_llm(model: str | None = None, temperature: float = 0.7):
    """Instantiate an LLM based on the LLM_BACKEND env variable.

    Backends
    --------
    google_ai_studio  Uses langchain-google-genai + GOOGLE_AI_STUDIO_KEY.
                      Model resolved from DEFAULT_MODEL_GAS env var.
    openrouter        Uses langchain-openai pointed at OpenRouter.
                      Model resolved from DEFAULT_MODEL_OR env var.
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
# Team Support Node
# ---------------------------------------------------------------------------

_SUPPORT_SYSTEM = """You are a computational linguist on **Team Support**.
Your sole mission is to build the strongest possible argument that the target word
has undergone genuine *diachronic* semantic change between the two time periods.

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
   - **Linguistic Drift**: driven by internal linguistic processes (metaphor, metonymy, analogy).

Be specific, cite corpus evidence directly, and construct a compelling argument."""

_SUPPORT_USER = """Analyze the semantic change of the word: "{word}"

SENTENCES FROM {t_old}:
{sentences_old}

SENTENCES FROM {t_new}:
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

    response = _robust_invoke(
        llm,
        [
            SystemMessage(content=_SUPPORT_SYSTEM),
            HumanMessage(
                content=_SUPPORT_USER.format(
                    word=state["word"],
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
4. Pre-emptively counter broadening/narrowing/transfer claims by showing stable semantic features.

Be specific, cite corpus evidence directly, and construct a compelling argument."""

_REFUSE_USER = """Analyze the semantic stability of the word: "{word}"

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

    response = _robust_invoke(
        llm,
        [
            SystemMessage(content=_REFUSE_SYSTEM),
            HumanMessage(
                content=_REFUSE_USER.format(
                    word=state["word"],
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

Output ONLY valid JSON matching the schema provided.\
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
causal_driver, break_point_year, reasoning.\
"""


# ---------------------------------------------------------------------------
# Judge Node
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """You are an impartial **LLM Judge** specialising in diachronic semantics.
You will evaluate two arguments — one for semantic change and one for semantic stability —
and render a final, structured verdict.

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
  "reasoning": "<your full reasoning>"
}"""

_JUDGE_USER = """Evaluate the following debate about the word "{word}" \
({t_old} vs. {t_new}):

--- ARGUMENT FOR CHANGE (Team Support) ---
{arg_change}

--- ARGUMENT FOR STABILITY (Team Refuse) ---
{arg_stable}

Work through the diagnostic checklist step-by-step, then output ONLY a valid JSON \
object with your final structured verdict. Use the exact JSON schema from your instructions."""


def _run_coarse_stage(word: str, t_old: str, t_new: str,
                      arg_change: str, arg_stable: str) -> _CoarseVerdict | None:
    """Stage 1: classify into STABLE / Transfer / Broadening / Narrowing."""
    messages = [
        SystemMessage(content=_JUDGE_COARSE_SYSTEM),
        HumanMessage(content=_JUDGE_COARSE_USER.format(
            word=word, t_old=t_old, t_new=t_new,
            arg_change=arg_change, arg_stable=arg_stable,
        )),
    ]
    try:
        llm = _get_llm(temperature=0.1).with_structured_output(_CoarseVerdict)
        result = _robust_invoke(llm, messages)
        if result is not None:
            print(f"[JUDGE-S1] '{word}' → {result.coarse_category}")
            return result
    except Exception as e:
        print(f"[JUDGE-S1] Structured output failed for '{word}': {e}. Trying raw.")

    # Fallback: raw text parse for coarse category
    raw_llm = _get_llm(temperature=0.1)
    messages_raw = [
        SystemMessage(content=_JUDGE_COARSE_SYSTEM),
        HumanMessage(content=_JUDGE_COARSE_USER.format(
            word=word, t_old=t_old, t_new=t_new,
            arg_change=arg_change, arg_stable=arg_stable,
        ) + "\n\nRespond with a JSON object: {\"coarse_category\": \"...\", \"reasoning\": \"...\"}"),
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
                        coarse_reasoning: str) -> JudgeVerdict:
    """Stage 2: identify exact Transfer mechanism."""
    messages = [
        SystemMessage(content=_JUDGE_TRANSFER_SYSTEM),
        HumanMessage(content=_JUDGE_TRANSFER_USER.format(
            word=word, t_old=t_old, t_new=t_new,
            arg_change=arg_change, arg_stable=arg_stable,
            coarse_reasoning=coarse_reasoning,
        )),
    ]
    try:
        llm = _get_llm(temperature=0.2).with_structured_output(JudgeVerdict)
        verdict = _robust_invoke(llm, messages)
        if verdict is not None and verdict.change_type in _TRANSFER_TYPES:
            print(f"[JUDGE-S2] '{word}' → {verdict.change_type}")
            return verdict
    except Exception as e:
        print(f"[JUDGE-S2] Structured output failed for '{word}': {e}. Falling back.")

    # Fallback: raw text
    raw_llm = _get_llm(temperature=0.1)
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

    # ── Stage 1: Coarse ──────────────────────────────────────────────────────
    coarse = _run_coarse_stage(word, t_old, t_new, arg_change, arg_stable)

    if coarse is None or coarse.coarse_category == "STABLE":
        return {"verdict": JudgeVerdict(
            word=word, verdict="STABLE",
            change_type=None, causal_driver=None,
            break_point_year=None,
            reasoning=coarse.reasoning if coarse else "Stage 1 failed; defaulting to STABLE.",
        ).model_dump()}

    if coarse.coarse_category == "Broadening":
        print(f"[JUDGE] '{word}' → Broadening → Generalization (direct)")
        return {"verdict": JudgeVerdict(
            word=word, verdict="CHANGE DETECTED",
            change_type="Generalization",
            causal_driver="Linguistic Drift",
            break_point_year=None,
            reasoning=coarse.reasoning,
        ).model_dump()}

    if coarse.coarse_category == "Narrowing":
        print(f"[JUDGE] '{word}' → Narrowing → Specialization (direct)")
        return {"verdict": JudgeVerdict(
            word=word, verdict="CHANGE DETECTED",
            change_type="Specialization",
            causal_driver="Linguistic Drift",
            break_point_year=None,
            reasoning=coarse.reasoning,
        ).model_dump()}

    # ── Stage 2: Fine-grained Transfer ───────────────────────────────────────
    verdict = _run_transfer_stage(
        word, t_old, t_new, arg_change, arg_stable, coarse.reasoning
    )
    return {"verdict": verdict.model_dump()}
