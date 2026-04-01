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
import pathlib
import random
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
# Judge few-shot examples
# ---------------------------------------------------------------------------
_EXAMPLES_FILE = pathlib.Path(__file__).parent / "judge_examples.json"

def _load_judge_examples() -> list[dict]:
    with open(_EXAMPLES_FILE, encoding="utf-8") as f:
        return json.load(f)

def _build_fewshot_block(current_word: str | None = None,
                         rng: random.Random | None = None) -> str:
    # See _build_transfer_fewshot_block below for the Stage-2 equivalent.
    """Select one example per category (BROADENING, NARROWING, STABLE) and two
    TRANSFER examples, then format them as the few-shot block for the coarse judge.

    Parameters
    ----------
    current_word:
        The word currently being evaluated. Any example whose ``word`` field
        matches (case-insensitive) is excluded so the judge never sees a
        worked answer for the target word itself.
    rng:
        Optional seeded Random instance for reproducible selection. When None,
        uses the global random state.
    """
    _target = current_word.lower() if current_word else None
    examples = [e for e in _load_judge_examples()
                if _target is None or e["word"].lower() != _target]
    _rng = rng or random

    by_category: dict[str, list[dict]] = {}
    for ex in examples:
        by_category.setdefault(ex["category"], []).append(ex)

    selected: list[dict] = []
    for cat in ("BROADENING", "NARROWING", "STABLE"):
        pool = by_category.get(cat, [])
        if pool:
            selected.append(_rng.choice(pool))
    transfer_pool = by_category.get("TRANSFER", [])
    selected.extend(_rng.sample(transfer_pool, min(2, len(transfer_pool))))

    lines = ["Few-shot examples", "-----------------"]
    for ex in selected:
        lines.append(f'Word: "{ex["word"]}"  \u2192  {ex["category"]}')
        lines.append(f'  OLD: {ex["old_sense"]}  |  NEW: {ex["new_sense"]}')
        lines.append(f'  Why: {ex["explanation"]}')
        lines.append("")
    return "\n".join(lines).rstrip()


def _build_transfer_fewshot_block(current_word: str | None = None,
                                   rng: random.Random | None = None) -> str:
    """Select one Transfer example per mechanism and format as a few-shot block
    for the Stage-2 transfer judge.

    Mechanisms: Metaphor, Metonymy, Analogy, Ellipsis, Antiphrasis,
                Auto-Antonym, Synecdoche.
    One example is drawn randomly from each mechanism's pool, excluding any
    example whose word matches ``current_word``.
    """
    _target = current_word.lower() if current_word else None
    examples = _load_judge_examples()
    _rng = rng or random

    by_mechanism: dict[str, list[dict]] = {}
    for ex in examples:
        if (ex.get("category") == "TRANSFER"
                and ex.get("transfer_mechanism")
                and (_target is None or ex["word"].lower() != _target)):
            by_mechanism.setdefault(ex["transfer_mechanism"], []).append(ex)

    mechanisms = [
        "Metaphor", "Metonymy", "Analogy", "Ellipsis",
        "Antiphrasis", "Auto-Antonym", "Synecdoche",
    ]

    lines = ["Few-shot examples (one per mechanism)", "--------------------------------------"]
    for mech in mechanisms:
        pool = by_mechanism.get(mech, [])
        if not pool:
            continue
        ex = _rng.choice(pool)
        lines.append(f'{mech}: "{ex["word"]}"')
        lines.append(f'  OLD: {ex["old_sense"]}')
        lines.append(f'  NEW: {ex["new_sense"]}')
        lines.append(f'  Why: {ex["explanation"]}')
        lines.append("")
    return "\n".join(lines).rstrip()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_DEFAULT_MODEL_OR     = "google/gemini-2.5-flash"
_DEFAULT_MODEL_GAS    = "gemini-2.0-flash-lite"
_DEFAULT_MODEL_VAI    = "gemini-2.5-flash"
_DEFAULT_MODEL_GROQ   = "llama3-70b-8192"
_DEFAULT_MODEL_NEBIUS = "meta-llama/Meta-Llama-3.1-70B-Instruct"
# Judge can use a stronger/reasoning model independently of the team agents.
# Set JUDGE_MODEL_GAS / JUDGE_MODEL_OR / JUDGE_MODEL_NEBIUS in .env to override; falls back to the team default.
_JUDGE_MODEL_OR     = os.getenv("JUDGE_MODEL_OR")     # e.g. "google/gemini-2.5-flash"
_JUDGE_MODEL_GAS    = os.getenv("JUDGE_MODEL_GAS")    # e.g. "gemini-2.5-flash"
_JUDGE_MODEL_NEBIUS        = os.getenv("JUDGE_MODEL_NEBIUS")        # e.g. "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1"
_LEXICOGRAPHER_MODEL_NEBIUS = os.getenv("LEXICOGRAPHER_MODEL_NEBIUS") # e.g. "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1"
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


def _get_llm(model: str | None = None, temperature: float = 0.7, seed: int | None = None) -> Any:
    """Instantiate an LLM based on the LLM_BACKEND env variable.

    If ``seed`` is None, reads LLM_SEED from the environment (if set).
    Seed is passed to Nebius/OpenRouter (OpenAI-compatible) and Gemini backends
    for reproducible sampling at temperature > 0.

    Backends
    --------
    google_ai_studio  Uses langchain-google-genai + GOOGLE_AI_STUDIO_KEY.
                      Model resolved from DEFAULT_MODEL_GAS env var.
    vertex_ai         Uses langchain-google-genai + VERTEX_AI_KEY (Express mode).
                      Model resolved from DEFAULT_MODEL_VAI env var.
    groq              Uses langchain-groq + GROQ_API_KEY.
                      Model resolved from DEFAULT_MODEL_GROQ env var.
    nebius            Uses langchain-openai pointed at Nebius AI Studio.
                      Model resolved from NEBIUS_MODEL env var.
    openrouter        Uses langchain-openai pointed at OpenRouter.
                      Model resolved from DEFAULT_MODEL_OR env var.

    Pass ``model`` explicitly to override the default for a specific call site
    (e.g. the judge using a stronger reasoning model).
    """
    backend = os.getenv("LLM_BACKEND", "openrouter").lower()

    # Resolve seed: explicit arg → LLM_SEED env var → None
    _env_seed = os.getenv("LLM_SEED")
    if seed is None and _env_seed is not None:
        try:
            seed = int(_env_seed)
        except ValueError:
            pass

    if backend == "google_ai_studio":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_AI_STUDIO_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_AI_STUDIO_KEY is not set in .env")
        resolved_model = model or os.getenv("DEFAULT_MODEL_GAS", _DEFAULT_MODEL_GAS)
        kwargs = {}
        if seed is not None:
            kwargs["seed"] = seed
        return ChatGoogleGenerativeAI(
            model=resolved_model,
            temperature=temperature,
            google_api_key=api_key,
            max_retries=3,
            max_output_tokens=2048,
            **kwargs,
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

    if backend == "nebius":
        api_key = os.getenv("NEBIUS_API_KEY")
        if not api_key:
            raise EnvironmentError("LLM_BACKEND=nebius but NEBIUS_API_KEY is not set in .env.")
        resolved_model = model or os.getenv("NEBIUS_MODEL", _DEFAULT_MODEL_NEBIUS)
        return ChatOpenAI(
            model=resolved_model,
            temperature=temperature,
            api_key=api_key,
            base_url="https://api.studio.nebius.com/v1/",
            max_retries=3,
            **({"seed": seed} if seed is not None else {}),
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
        },
        **({"seed": seed} if seed is not None else {}),
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
    if backend == "google_ai_studio":
        judge_model = _JUDGE_MODEL_GAS
    elif backend == "nebius":
        judge_model = _JUDGE_MODEL_NEBIUS
    else:
        judge_model = _JUDGE_MODEL_OR
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


def _format_oed_quotes_block(ctx: dict) -> str:
    """Format raw OED/Wiktionary quotes into a neutral evidence block for team prompts."""
    historical = ctx.get("historical") or []
    modern = ctx.get("modern") or []
    if not historical and not modern:
        return ""

    source = ctx.get("source", "unknown").upper()
    lines = [f"=== OED QUOTATION EVIDENCE (source: {source}) ==="]
    lines.append("Earliest attested quotations (oldest known usages):")

    if historical:
        for year, text in historical:
            lines.append(f"  [{year}] \"{text}\"")

    if modern:
        lines.append("Most recent quotations (latest known usages):")
        for year, text in modern:
            lines.append(f"  [{year}] \"{text}\"")

    lines.append("Use these as supplementary evidence alongside the corpus sentences.")
    lines.append("They show the temporal arc of attested usage but do NOT predetermine the change mechanism.")
    lines.append("=" * 53)
    return "\n".join(lines)


def oed_context_node(state: GraphState) -> dict:
    """OED Context: fetches dated quotations from OED/Wiktionary and injects them
    as neutral supplementary evidence into team prompts — no LLM synthesis step."""
    word = state["word"]
    try:
        ctx = fetch_etymology_context(word)
        block = _format_oed_quotes_block(ctx)
        if block:
            print(f"[OED] Evidence fetched for '{word}' (source: {ctx['source']}, "
                  f"historical: {len(ctx.get('historical') or [])}, "
                  f"modern: {len(ctx.get('modern') or [])})")
        else:
            print(f"[OED] No quotes found for '{word}' (source: {ctx['source']})")
        return {"oed_quotes_block": block}
    except Exception as exc:
        print(f"[OED] Skipped for '{word}': {exc}")
        return {"oed_quotes_block": ""}


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
        _backend = os.getenv("LLM_BACKEND", "openrouter").lower()
        _lex_model = _LEXICOGRAPHER_MODEL_NEBIUS if _backend == "nebius" else None
        llm = _get_llm(model=_lex_model, temperature=0.1).with_structured_output(EtymologyResult)
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

STEP 1 — Establish the OLD core sense
  Identify the dominant, prototypical meaning from OLD sentences only (2–3 key features).

STEP 2 — Identify the incompatible NEW sense
  Find NEW sentences whose semantics cannot be explained by the OLD core sense.
  These are your strongest evidence for change.

STEP 3 — Classify the mechanism (CRITICAL — work through this before writing)
  Ask the following questions IN ORDER:

  Q1: Is the new referent/domain qualitatively DIFFERENT from the old one?
      (i.e. not simply a wider or narrower version of the same thing)
      → If YES: the mechanism is a TRANSFER type — continue to Q2.
      → If NO:  the scope shifted within the same domain — continue to Q4.

  Q2 (Transfer): What is the link between the old and new referent?
      • Perceptual or structural RESEMBLANCE across domains  → **Metaphor**
        e.g. "mouse" (rodent) → (computer device): shape/cable resembles tail.
      • CONTIGUITY or ASSOCIATION within the same event frame → **Metonymy**
        e.g. "bead" (prayer) → (small sphere): the physical object used in prayer.
        e.g. "shall" (obligation) → (future): modal shift within futurity domain.
      • Abstract/functional PARALLEL (not perceptual)        → **Analogy**
        e.g. "fast" (firmly fixed) → (moving quickly): "no yielding" applied to speed.
      • Word was SHORTENED from a historically attested compound → **Ellipsis**
        e.g. "canine tooth" → "canine".  Always check for the source compound first.
      • Meaning reversed through IRONY or euphemism           → **Antiphrasis**
      • Meaning reversed WITHOUT irony (genuine polarity flip) → **Auto-Antonym**
      • PART-FOR-WHOLE or whole-for-part within same domain   → **Synecdoche**

  Q3 (Generalization vs. Transfer — most common mistake):
      Do NOT choose Generalization just because the word gained new uses.
      Generalization requires the new referents to be a STRICTLY WIDER SET of the
      same kind of thing as the old referents.
        ✗ "mouse" gaining a computer sense is NOT Generalization — computer devices
          are not a broader category of rodents. → Metaphor.
        ✗ "holiday" shifting from holy day to leisure break is NOT Generalization —
          leisure breaks are not a broader category of holy days. → Metonymy.
        ✓ "dog" expanding from male canine to any canine IS Generalization.

  Q4 (Scope shift within same domain):
      • New meaning covers MORE referents of the same kind → **Generalization**
      • New meaning covers FEWER referents of the same kind → **Specialization**

STEP 4 — Hypothesize EXACTLY ONE Causal Driver:
  - **Cultural Shift**: driven by broad societal, technological, or cultural changes.
  - **Linguistic Drift**: driven by internal linguistic processes (metaphor, metonymy, analogy)."""

_SUPPORT_USER = """Analyze the word "{word}" (used as a {word_type}) for diachronic semantic change.

SENTENCES FROM {t_old} (OLD period):
{sentences_old}

SENTENCES FROM {t_new} (NEW period):
{sentences_new}

Follow Steps 1–4 from your instructions explicitly:
  Step 1: State the OLD core sense (2–3 features from OLD sentences).
  Step 2: Quote the NEW sentence(s) that are incompatible with the OLD sense.
  Step 3: Work through Q1→Q2/Q4 to determine the mechanism. Show your reasoning.
           In particular: is the new referent qualitatively DIFFERENT (→ Transfer type)
           or just a wider/narrower set of the same thing (→ Generalization/Specialization)?
  Step 4: State the Causal Driver.

Then write your final Arg_change, concluding with:
  Change Type: <exactly one of Metaphor / Metonymy / Analogy / Generalization / Specialization / Ellipsis / Antiphrasis / Auto-Antonym / Synecdoche>
  Causal Driver: <Cultural Shift | Linguistic Drift>"""


def team_support_node(state: GraphState) -> dict:
    """Team Support: constructs Arg_change arguing for semantic shift."""
    llm = _get_llm()

    sentences_old = "\n".join(f"  • {s}" for s in state["sentences_old"])
    sentences_new = "\n".join(f"  • {s}" for s in state["sentences_new"])

    # Append OED quote block after system instructions (supplementary evidence, not ground truth).
    oed_block = state.get("oed_quotes_block", "") or ""
    system_content = f"{_SUPPORT_SYSTEM}\n\n{oed_block}" if oed_block else _SUPPORT_SYSTEM

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

    # Append OED quote block after system instructions (supplementary evidence, not ground truth).
    oed_block = state.get("oed_quotes_block", "") or ""
    system_content = f"{_REFUSE_SYSTEM}\n\n{oed_block}" if oed_block else _REFUSE_SYSTEM

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

{fewshot_block}

Decision rules
--------------
1. If BOTH arguments agree there is no change → lean STABLE.
2. If the new referents are strictly a larger set → BROADENING.
3. If the new referents are strictly a smaller set → NARROWING.
4. If the new referent is qualitatively different (different domain, reversed polarity, \
   ironic inversion, compound shortening) → TRANSFER.
5. Weigh QUALITY of evidence, not volume.

Output ONLY valid JSON with EXACTLY these two fields:
  {{"coarse_category": "STABLE|Transfer|Broadening|Narrowing", "reasoning": "<brief justification>"}}\
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

Metonymy    – New sense connected to old via CONTIGUITY or ASSOCIATION within the
              same conceptual world (part-for-whole, cause-for-effect,
              container-for-content, instrument-for-action).

Analogy     – New sense connected via STRUCTURAL resemblance across semantic domains
              (not perceptual). The mapping is abstract/functional rather than visual.

Ellipsis    – A COMPOUND phrase was shortened; the head noun inherited the full meaning
              of the compound.
              Diagnostic: always check if the word was historically part of a longer phrase.

Antiphrasis – Meaning shifted to its OPPOSITE through IRONIC or EUPHEMISTIC usage.
              The shift happened gradually via sarcastic/ironic context.
              Distinguish from Auto-Antonym: the mechanism is social/ironic, not internal.

Auto-Antonym – Word acquired a sense DIRECTLY OPPOSITE to its original meaning without
               irony — a straightforward polarity reversal driven by slang/register shift.
               Distinguish from Antiphrasis: no irony — speakers genuinely use it positively.

Synecdoche  – PART-FOR-WHOLE or WHOLE-FOR-PART meaning transfer within the same domain.

{fewshot_block}

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
    fewshot = _build_fewshot_block(current_word=word)
    coarse_prompt = _JUDGE_COARSE_SYSTEM.format(fewshot_block=fewshot)
    coarse_system = (
        f"{lexicographer_dossier}\n\n{coarse_prompt}"
        if lexicographer_dossier else coarse_prompt
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
        llm = _get_judge_llm(temperature=0.1).with_structured_output(_CoarseVerdict, method="json_mode")
        result = _robust_invoke(llm, messages)
        if result is not None:
            print(f"[JUDGE-S1] '{word}' → {result.coarse_category}")
            return result
    except Exception as e:
        print(f"[JUDGE-S1] Structured output failed for '{word}': {e}. Trying raw.")

    # Fallback: raw text parse for coarse category
    # Also handles model-specific aliases: "verdict"→coarse_category, "reason"→reasoning,
    # and case variants like "NARROWING"→"Narrowing", "BROADENING"→"Broadening".
    _COARSE_ALIASES = {
        "STABLE": "STABLE", "stable": "STABLE",
        "Transfer": "Transfer", "TRANSFER": "Transfer", "transfer": "Transfer",
        "Broadening": "Broadening", "BROADENING": "Broadening", "broadening": "Broadening",
        "Narrowing": "Narrowing", "NARROWING": "Narrowing", "narrowing": "Narrowing",
    }
    raw_llm = _get_judge_llm(temperature=0.1)
    messages_raw = [
        SystemMessage(content=coarse_system),
        HumanMessage(content=user_prompt + '\n\nRespond with a JSON object using EXACTLY these fields: {"coarse_category": "STABLE|Transfer|Broadening|Narrowing", "reasoning": "..."}'),
    ]
    try:
        resp = _robust_invoke(raw_llm, messages_raw)
        text = _extract_text(resp)
        # Try canonical field name first, then common aliases
        m = re.search(
            r'"(?:coarse_category|verdict|category|label)"\s*:\s*"([^"]+)"', text
        )
        if m:
            raw_val = m.group(1)
            cat = _COARSE_ALIASES.get(raw_val)
            if cat:
                print(f"[JUDGE-S1-fallback] '{word}' → {cat}")
                return _CoarseVerdict(coarse_category=cat, reasoning=text)
            print(f"[JUDGE-S1-fallback] unrecognised value '{raw_val}' for '{word}'")
    except Exception as e:
        print(f"[JUDGE-S1] Raw fallback also failed for '{word}': {e}")
    return None


def _run_transfer_stage(word: str, t_old: str, t_new: str,
                        arg_change: str, arg_stable: str,
                        coarse_reasoning: str,
                        lexicographer_dossier: str = "") -> JudgeVerdict:
    """Stage 2: identify exact Transfer mechanism."""
    fewshot = _build_transfer_fewshot_block(current_word=word)
    transfer_prompt = _JUDGE_TRANSFER_SYSTEM.format(fewshot_block=fewshot)
    transfer_system = (
        f"{lexicographer_dossier}\n\n{transfer_prompt}"
        if lexicographer_dossier else transfer_prompt
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
        llm = _get_judge_llm(temperature=0.2).with_structured_output(JudgeVerdict, method="json_mode")
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
