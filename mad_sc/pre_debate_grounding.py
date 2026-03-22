"""
mad_sc/pre_debate_grounding.py
------------------------------
Pre-debate grounding pipeline for MAD-SC.

Bridges raw historical corpora and the agentic debate layer by computing
quantitative tension metrics (SED, TD) over lexical substitutes and
packaging them as a structured Hypothesis Document ready for agent injection.

Pipeline
--------
    sentences_old, sentences_new
            │
            ▼
    generate_substitutes()           ← WordNet (default) or pluggable LLM stub
            │
            ▼
    compute_sed_profiles()           ← layer-wise BERT cosine distance
            │
            ▼
    build_hypothesis_document()      ← ranked JSON + narrative prompt block
            │
            ▼
    HypothesisDocument.to_prompt_block()  →  injected into agent system prompts

Quick start
-----------
    from mad_sc.pre_debate_grounding import run_grounding_pipeline

    doc = run_grounding_pipeline(
        word="edge",
        sentences_old=sentences_c1,
        sentences_new=sentences_c2,
        t_old="Corpus 1 (1810–1860)",
        t_new="Corpus 2 (1960–2010)",
        top_k=5,
    )
    print(doc.to_prompt_block())   # inject into agent system prompts
    print(doc.to_json())           # full structured output
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional, Protocol, runtime_checkable

import numpy as np
import nltk
from nltk.corpus import wordnet as wn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SubstituteType = Literal["synonym", "antonym", "hypernym", "random"]

# Layer index ranges (1-indexed; BERT-base has 12 transformer layers).
# hidden_states[0] is the embedding layer; we skip it.
LAYER_GROUPS: dict[str, range] = {
    "early":  range(1, 5),    # layers 1–4
    "middle": range(5, 9),    # layers 5–8
    "deep":   range(9, 13),   # layers 9–12
}

_DEFAULT_BERT_MODEL = "bert-base-uncased"
_MAX_SEQ_LENGTH = 512
_NLTK_RESOURCES = [
    ("corpora/wordnet", "wordnet"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("tokenizers/punkt", "punkt"),
    ("corpora/omw-1.4", "omw-1.4"),
]


# ---------------------------------------------------------------------------
# NLTK bootstrap (silent download when not present)
# ---------------------------------------------------------------------------

def _ensure_nltk_resources() -> None:
    for find_path, download_key in _NLTK_RESOURCES:
        try:
            nltk.data.find(find_path)
        except LookupError:
            nltk.download(download_key, quiet=True)


_ensure_nltk_resources()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Substitute:
    """A lexical replacement for the target word.

    Parameters
    ----------
    base_form : str
        Canonical lemma of the replacement (e.g., ``"boundary"``).
    surface_form : str
        Morphologically inflected form fitted to the original sentence context
        (e.g., ``"boundaries"`` when the target appeared as a plural).
    sub_type : SubstituteType
        Relationship to the target: ``"synonym"``, ``"antonym"``,
        ``"hypernym"``, or ``"random"``.
    """

    base_form: str
    surface_form: str
    sub_type: SubstituteType


@dataclass
class LayerSEDScore:
    """SED score at a single transformer layer for one sentence.

    Parameters
    ----------
    layer_idx : int
        Transformer layer index (1-indexed; 1–12 for BERT-base).
    layer_group : str
        ``"early"`` (1–4), ``"middle"`` (5–8), or ``"deep"`` (9–12).
    sed : float
        Cosine distance between the original-word and substitute embeddings
        at this layer, in [0, 2] (practically [0, 1] for contextual encoders).
    """

    layer_idx: int
    layer_group: str
    sed: float


@dataclass
class SentenceSEDResult:
    """Layer-wise SED computed for a single sentence pair.

    Parameters
    ----------
    sentence_idx : int
        Index into the sentence list passed to the pipeline.
    original_sentence : str
        Source sentence as retrieved from the corpus.
    modified_sentence : str
        Sentence with the target word replaced by the substitute's
        surface form.
    layer_scores : list[LayerSEDScore]
        One score per transformer layer (12 for BERT-base, ordered 1→12).
    """

    sentence_idx: int
    original_sentence: str
    modified_sentence: str
    layer_scores: list[LayerSEDScore]


@dataclass
class SubstituteProfile:
    """Full SED/TD analysis for one substitute across both corpora.

    Parameters
    ----------
    substitute : Substitute
        The lexical substitute this profile describes.
    sed_t1 / sed_t2 : list[SentenceSEDResult]
        Per-sentence SED results for Corpus 1 (old) and Corpus 2 (new).
        Sentences where the target word could not be located are skipped.
    avg_sed_t1 / avg_sed_t2 : dict[int, float]
        Mean SED per layer index averaged across all matched sentences.
    td_per_layer : dict[int, float]
        Time Difference per layer: ``|avg_sed_t1[l] − avg_sed_t2[l]|``.
    td_aggregate : float
        Mean TD across all layers — the primary ranking score.
    group_avg_t1 / group_avg_t2 : dict[str, float]
        Mean SED per layer group (``"early"``, ``"middle"``, ``"deep"``).
    group_td : dict[str, float]
        TD per layer group: ``|group_avg_t1[g] − group_avg_t2[g]|``.
    """

    substitute: Substitute
    sed_t1: list[SentenceSEDResult] = field(default_factory=list)
    sed_t2: list[SentenceSEDResult] = field(default_factory=list)
    avg_sed_t1: dict[int, float] = field(default_factory=dict)
    avg_sed_t2: dict[int, float] = field(default_factory=dict)
    td_per_layer: dict[int, float] = field(default_factory=dict)
    td_aggregate: float = 0.0
    group_avg_t1: dict[str, float] = field(default_factory=dict)
    group_avg_t2: dict[str, float] = field(default_factory=dict)
    group_td: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocol: pluggable substitute generators
# ---------------------------------------------------------------------------

@runtime_checkable
class SubstituteGeneratorProtocol(Protocol):
    """Interface for lexical substitute generators.

    Any object with a ``generate()`` method matching this signature qualifies,
    so duck-typing works without explicit subclassing.
    """

    def generate(
        self,
        target_word: str,
        context_sentences: list[str],
        n_per_type: int = 3,
    ) -> list[Substitute]:
        """Generate morphologically matched lexical substitutes.

        Parameters
        ----------
        target_word : str
            Bare lemma of the target (e.g., ``"edge"``, not ``"edge_nn"``).
        context_sentences : list[str]
            Sample sentences from which the dominant surface form is inferred
            for morphological matching.
        n_per_type : int
            Maximum number of candidates per substitute type.

        Returns
        -------
        list[Substitute]
            Flat list of substitutes (≤ 4 × n_per_type entries).
        """
        ...


# ---------------------------------------------------------------------------
# Morphological matching utilities
# ---------------------------------------------------------------------------

def _detect_dominant_surface(word: str, sentences: list[str]) -> str:
    """Detect the most frequent surface form of *word* across *sentences*.

    Searches each sentence for the target word (case-insensitive, whole-token)
    and returns the form that appears most often.  Falls back to *word* itself
    if no match is found.
    """
    pattern = re.compile(rf"\b({re.escape(word)}[a-z']*)\b", re.IGNORECASE)
    counts: dict[str, int] = {}
    for sentence in sentences:
        for m in pattern.finditer(sentence):
            surface = m.group(1).lower()
            counts[surface] = counts.get(surface, 0) + 1
    if not counts:
        return word
    return max(counts, key=counts.__getitem__)


def _detect_morphological_function(base: str, surface: str) -> str | None:
    """Infer the morphological function that maps *base* to *surface*.

    Returns a semantic label (``"gerund"``, ``"past_tense"``, ``"plural"``,
    ``"comparative"``, ``"superlative"``) or ``None`` when no common English
    inflectional relationship is recognised.

    By working with *functions* rather than raw suffix strings, we avoid
    blindly transferring consonant-doubling or -y/-ies rules that belong to
    the original word but not necessarily to the substitute.
    """
    b, s = base.lower(), surface.lower()
    if s == b:
        return None

    # Gerund (-ing), with e-elision (edge→edging), -ie→-ying (die→dying),
    # or consonant doubling (run→running)
    if s.endswith("ing"):
        stem = s[:-3]
        if stem == b:                              # walk → walking
            return "gerund"
        if b.endswith("e") and stem == b[:-1]:    # edge → edging
            return "gerund"
        if b.endswith("ie") and s == b[:-2] + "ying":  # die → dying
            return "gerund"
        # consonant doubling: run → running (stem = "runn", b = "run")
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[:-1] == b:
            return "gerund"

    # Past tense (-ed / -d), with e-elision, -y→-ied, and consonant doubling
    if s.endswith("ed"):
        stem = s[:-2]
        if stem == b:                              # walk → walked
            return "past_tense"
        if b.endswith("e") and stem == b[:-1]:    # edge → edged (stem = "edg")
            return "past_tense"
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[:-1] == b:
            return "past_tense"                   # run → runned (rare but detectable)
    if s == b + "d" and b.endswith("e"):           # love → loved
        return "past_tense"
    # -y → -ied (carry → carried)
    if b.endswith("y") and s == b[:-1] + "ied":
        return "past_tense"

    # Plural / 3rd-person singular
    if s == b + "s":
        return "plural"
    if s == b + "es":
        return "plural"
    if b.endswith("y") and s == b[:-1] + "ies":   # body → bodies
        return "plural"

    # Comparative / superlative (with consonant doubling: big → bigger)
    if s == b + "er":
        return "comparative"
    if s.endswith("er"):
        stem = s[:-2]
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[:-1] == b:
            return "comparative"
    if s == b + "est":
        return "superlative"
    if s.endswith("est"):
        stem = s[:-3]
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[:-1] == b:
            return "superlative"

    return None


def _apply_morphological_function(substitute: str, function: str | None) -> str:
    """Inflect *substitute* according to *function* using English spelling rules.

    Each function applies the correct rule for the *substitute's* own spelling
    (e.g., ``"plural"`` turns ``"boundary"`` → ``"boundaries"``), not a raw
    suffix copied from the target word.
    """
    if function is None:
        return substitute
    s = substitute.lower()

    if function == "gerund":
        if s.endswith("ie"):                        # die → dying
            return s[:-2] + "ying"
        if s.endswith("e") and not s.endswith("ee"):
            return s[:-1] + "ing"                  # edge → edging
        return s + "ing"                            # walk → walking

    if function == "past_tense":
        if s.endswith("y") and len(s) > 1 and s[-2] not in "aeiou":
            return s[:-1] + "ied"                  # carry → carried
        if s.endswith("e"):
            return s + "d"                          # love → loved
        return s + "ed"                             # walk → walked

    if function == "plural":
        if s.endswith("y") and len(s) > 1 and s[-2] not in "aeiou":
            return s[:-1] + "ies"                  # boundary → boundaries
        if s.endswith(("s", "x", "z", "ch", "sh")):
            return s + "es"                         # bus → buses
        return s + "s"                              # edge → edges

    if function == "comparative":
        return s + "er"

    if function == "superlative":
        return s + "est"

    return substitute


def _match_morphology(target_base: str, target_surface: str, substitute_base: str) -> str:
    """Return *substitute_base* inflected to match *target_surface*'s morphology.

    Parameters
    ----------
    target_base : str
        Canonical lemma of the target word (e.g., ``"edge"``).
    target_surface : str
        Inflected surface form found in context (e.g., ``"edges"``).
    substitute_base : str
        Canonical lemma of the substitute (e.g., ``"boundary"``).

    Returns
    -------
    str
        Inflected substitute (e.g., ``"boundaries"``).
    """
    function = _detect_morphological_function(target_base, target_surface)
    return _apply_morphological_function(substitute_base, function)


# ---------------------------------------------------------------------------
# WordNet-based substitute generator (default)
# ---------------------------------------------------------------------------

class WordNetSubstituteGenerator:
    """Generate lexical substitutes using NLTK WordNet.

    Produces:
    - **synonyms** from the target's synset lemmas
    - **antonyms** from WordNet antonym links
    - **hypernyms** from the synset hierarchy (immediate parents)
    - **random** words sampled from WordNet synsets with disjoint lexname

    Surface forms are morphologically matched to the dominant inflection
    detected in *context_sentences*.
    """

    def generate(
        self,
        target_word: str,
        context_sentences: list[str],
        n_per_type: int = 3,
    ) -> list[Substitute]:
        dominant_surface = _detect_dominant_surface(target_word, context_sentences)
        candidates: list[tuple[str, SubstituteType]] = []
        candidates += [(w, "synonym") for w in self._synonyms(target_word, n_per_type)]
        candidates += [(w, "antonym") for w in self._antonyms(target_word, n_per_type)]
        candidates += [(w, "hypernym") for w in self._hypernyms(target_word, n_per_type)]
        candidates += [(w, "random")  for w in self._random_words(target_word, n_per_type)]

        substitutes: list[Substitute] = []
        seen: set[str] = {target_word.lower()}
        for base_form, sub_type in candidates:
            if base_form.lower() in seen:
                continue
            seen.add(base_form.lower())
            surface = _match_morphology(target_word, dominant_surface, base_form)
            substitutes.append(Substitute(
                base_form=base_form,
                surface_form=surface,
                sub_type=sub_type,
            ))
        return substitutes

    # ---- private helpers ----

    def _synonyms(self, word: str, n: int) -> list[str]:
        results: list[str] = []
        for synset in wn.synsets(word):
            for lemma in synset.lemmas():
                candidate = lemma.name().replace("_", " ")
                if candidate.lower() != word.lower():
                    results.append(candidate)
                if len(results) >= n:
                    return results
        return results

    def _antonyms(self, word: str, n: int) -> list[str]:
        results: list[str] = []
        for synset in wn.synsets(word):
            for lemma in synset.lemmas():
                for ant in lemma.antonyms():
                    results.append(ant.name().replace("_", " "))
                    if len(results) >= n:
                        return results
        return results

    def _hypernyms(self, word: str, n: int) -> list[str]:
        results: list[str] = []
        for synset in wn.synsets(word):
            for hypernym in synset.hypernyms():
                for lemma in hypernym.lemmas():
                    candidate = lemma.name().replace("_", " ")
                    if candidate.lower() != word.lower():
                        results.append(candidate)
                if len(results) >= n:
                    return results
        return results

    def _random_words(self, word: str, n: int) -> list[str]:
        """Sample *n* words from WordNet synsets with a different lexname."""
        target_synsets = wn.synsets(word)
        target_lexnames: set[str] = {ss.lexname() for ss in target_synsets}
        target_words: set[str] = {
            lemma.name().lower()
            for ss in target_synsets
            for lemma in ss.lemmas()
        }
        # Collect candidates from disjoint lexicographer categories
        pool: list[str] = []
        for synset in wn.all_synsets():
            if synset.lexname() in target_lexnames:
                continue
            for lemma in synset.lemmas():
                candidate = lemma.name()
                # Skip multi-word expressions, numbers, and known relatives
                if "_" not in candidate and candidate.lower() not in target_words:
                    pool.append(candidate)
            if len(pool) > 500:
                break  # avoid exhausting the full WordNet graph
        if not pool:
            return []
        random.shuffle(pool)
        return pool[:n]


# ---------------------------------------------------------------------------
# LLM substitute generator (pluggable stub)
# ---------------------------------------------------------------------------

class LLMSubstituteGenerator:
    """Stub substitute generator that delegates to an LLM backend.

    The default ``_call_llm()`` implementation returns a mock response so
    the pipeline can run end-to-end without an API key.  Wire a real backend
    by subclassing and overriding ``_call_llm()``.

    Examples
    --------
    **vLLM (local)**::

        class VLLMGenerator(LLMSubstituteGenerator):
            def __init__(self, endpoint: str, model: str):
                self._endpoint = endpoint
                self._model = model

            def _call_llm(self, prompt: str) -> str:
                import requests
                r = requests.post(
                    f"{self._endpoint}/v1/completions",
                    json={"model": self._model, "prompt": prompt, "max_tokens": 300},
                    timeout=30,
                )
                return r.json()["choices"][0]["text"]

    **OpenAI-compatible API**::

        class OpenAIGenerator(LLMSubstituteGenerator):
            def __init__(self, client, model: str = "gpt-4o-mini"):
                self._client = client
                self._model = model

            def _call_llm(self, prompt: str) -> str:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.choices[0].message.content

    **OpenRouter (already used in nodes.py)**::

        class OpenRouterGenerator(LLMSubstituteGenerator):
            def _call_llm(self, prompt: str) -> str:
                from mad_sc.nodes import _get_llm, _robust_invoke
                from langchain_core.messages import HumanMessage
                llm = _get_llm(temperature=0.5)
                return _robust_invoke(llm, [HumanMessage(content=prompt)]).content
    """

    _PROMPT_TEMPLATE = """\
You are a lexical semantics expert. For the English word "{word}", generate lexical \
substitutes in four categories.

Return ONLY a JSON object in this exact format (no commentary):
{{
  "synonym":  [<up to {n} single words>],
  "antonym":  [<up to {n} single words>],
  "hypernym": [<up to {n} single words>],
  "random":   [<up to {n} semantically unrelated single words>]
}}

Context sentences (use these to infer the dominant part-of-speech and sense):
{context}
"""

    def generate(
        self,
        target_word: str,
        context_sentences: list[str],
        n_per_type: int = 3,
    ) -> list[Substitute]:
        context_block = "\n".join(f"  - {s}" for s in context_sentences[:5])
        prompt = self._PROMPT_TEMPLATE.format(
            word=target_word,
            n=n_per_type,
            context=context_block,
        )
        raw = self._call_llm(prompt)
        return self._parse_response(raw, target_word, context_sentences)

    def _call_llm(self, prompt: str) -> str:
        """Override this method to connect to a real LLM backend.

        The default implementation returns a deterministic mock response.
        """
        logger.warning(
            "LLMSubstituteGenerator._call_llm() is returning MOCK data. "
            "Subclass and override _call_llm() to use a real LLM backend."
        )
        # Extract word from prompt for plausible mock
        m = re.search(r'word "(\w+)"', prompt)
        word = m.group(1) if m else "word"
        mock = {
            "synonym":  [f"mock_syn_{word}_1", f"mock_syn_{word}_2"],
            "antonym":  [f"mock_ant_{word}_1"],
            "hypernym": [f"mock_hyp_{word}_1", f"mock_hyp_{word}_2"],
            "random":   [f"mock_rnd_1", f"mock_rnd_2"],
        }
        return json.dumps(mock)

    def _parse_response(
        self,
        raw: str,
        target_word: str,
        context_sentences: list[str],
    ) -> list[Substitute]:
        """Parse a JSON response from the LLM into ``Substitute`` objects."""
        dominant_surface = _detect_dominant_surface(target_word, context_sentences)
        try:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            data: dict[str, list[str]] = json.loads(m.group() if m else raw)
        except (json.JSONDecodeError, AttributeError) as exc:
            logger.error("Failed to parse LLM substitute response: %s", exc)
            return []

        substitutes: list[Substitute] = []
        seen: set[str] = {target_word.lower()}
        for sub_type in ("synonym", "antonym", "hypernym", "random"):
            for base_form in data.get(sub_type, []):
                base_form = base_form.strip().lower().replace(" ", "_")
                if base_form in seen:
                    continue
                seen.add(base_form)
                surface = _match_morphology(target_word, dominant_surface, base_form)
                substitutes.append(Substitute(
                    base_form=base_form,
                    surface_form=surface,
                    sub_type=sub_type,  # type: ignore[arg-type]
                ))
        return substitutes


# ---------------------------------------------------------------------------
# Transformer model management
# ---------------------------------------------------------------------------

def _load_bert_model(model_name: str = _DEFAULT_BERT_MODEL):
    """Load a BERT-family model and tokenizer from HuggingFace.

    Returns
    -------
    (model, tokenizer, device) : tuple
        - model: ``transformers.BertModel`` (or compatible) in eval mode
        - tokenizer: ``transformers.BertTokenizerFast``
        - device: ``"mps"`` | ``"cuda"`` | ``"cpu"``

    Raises
    ------
    ImportError
        If ``transformers`` or ``torch`` are not installed.
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "The 'transformers' and 'torch' packages are required for SED "
            "computation. Install them with:\n"
            "    pip install torch transformers"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = model.to(device)
    logger.info("Loaded model '%s' on device '%s'.", model_name, device)
    return model, tokenizer, device


# ---------------------------------------------------------------------------
# SED computation
# ---------------------------------------------------------------------------

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two 1-D float vectors, in [0, 2]."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-8:
        return 0.0
    return float(1.0 - np.dot(a, b) / denom)


def _token_span_for_word(
    word: str,
    sentence: str,
    offset_mapping: list[tuple[int, int]],
) -> list[int] | None:
    """Return the token indices covering *word*'s first occurrence in *sentence*.

    Uses character-level offset mapping produced by a fast HuggingFace
    tokenizer (``return_offsets_mapping=True``).

    Returns ``None`` if the word cannot be located.
    """
    pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
    m = pattern.search(sentence)
    if not m:
        return None
    char_start, char_end = m.start(), m.end()
    indices = [
        i for i, (tok_start, tok_end) in enumerate(offset_mapping)
        if tok_start < char_end and tok_end > char_start
        and not (tok_start == 0 and tok_end == 0)  # skip [CLS]/[SEP]/padding
    ]
    return indices if indices else None


def _embed_word_in_sentence(
    sentence: str,
    target: str,
    model,
    tokenizer,
    device: str,
) -> np.ndarray | None:
    """Run *sentence* through the model and return layer-wise mean embeddings.

    Parameters
    ----------
    sentence : str
        Full input sentence.
    target : str
        The specific word (surface form) to extract embeddings for.
    model, tokenizer, device :
        As returned by ``_load_bert_model()``.

    Returns
    -------
    np.ndarray of shape ``(num_layers, hidden_size)`` or ``None``.
    ``num_layers`` is 12 for BERT-base (layers 1–12; embedding layer skipped).
    """
    import torch

    enc = tokenizer(
        sentence,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=_MAX_SEQ_LENGTH,
        padding=False,
    )
    offset_mapping: list[tuple[int, int]] = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}

    token_indices = _token_span_for_word(target, sentence, offset_mapping)
    if not token_indices:
        logger.debug("Word '%s' not found in sentence via offset mapping.", target)
        return None

    try:
        with torch.no_grad():
            outputs = model(**enc, output_hidden_states=True)
    except RuntimeError as exc:
        logger.warning("Forward pass failed for sentence: %s", exc)
        return None

    # hidden_states: tuple of len (num_layers + 1), each (batch, seq_len, hidden)
    hidden_states = outputs.hidden_states
    num_transformer_layers = len(hidden_states) - 1  # skip embedding layer at [0]

    layer_embeddings: list[np.ndarray] = []
    for layer_idx in range(1, num_transformer_layers + 1):
        # Average sub-word token embeddings for the target word
        token_embeds = hidden_states[layer_idx][0, token_indices, :]  # (n_toks, hidden)
        mean_embed = token_embeds.mean(dim=0).cpu().float().numpy()    # (hidden,)
        layer_embeddings.append(mean_embed)

    return np.stack(layer_embeddings)  # (num_layers, hidden_size)


def _compute_sentence_sed(
    sentence: str,
    target_word: str,
    substitute: Substitute,
    sentence_idx: int,
    model,
    tokenizer,
    device: str,
) -> SentenceSEDResult | None:
    """Compute layer-wise SED for one sentence/substitute pair.

    Replaces *target_word* (matched by its dominant surface form) with
    ``substitute.surface_form``, embeds both sentences independently, and
    computes cosine distance at each transformer layer.

    Returns ``None`` when the target word cannot be located in the sentence.
    """
    # Locate target's surface form in the sentence
    pattern = re.compile(rf"\b{re.escape(target_word)}\b", re.IGNORECASE)
    if not pattern.search(sentence):
        # Also try the substitute's base form (in case the sentence uses it)
        logger.debug(
            "Skipping sentence %d: target '%s' not found.", sentence_idx, target_word
        )
        return None

    modified = pattern.sub(substitute.surface_form, sentence, count=1)

    orig_embeds = _embed_word_in_sentence(sentence, target_word, model, tokenizer, device)
    if orig_embeds is None:
        return None

    sub_embeds = _embed_word_in_sentence(
        modified, substitute.surface_form, model, tokenizer, device
    )
    if sub_embeds is None:
        return None

    if orig_embeds.shape[0] != sub_embeds.shape[0]:
        logger.warning(
            "Layer count mismatch: original=%d, substitute=%d. Skipping.",
            orig_embeds.shape[0], sub_embeds.shape[0],
        )
        return None

    # Assign each layer index to a group
    layer_scores: list[LayerSEDScore] = []
    for layer_idx in range(1, orig_embeds.shape[0] + 1):
        group = next(
            (g for g, r in LAYER_GROUPS.items() if layer_idx in r),
            "deep",  # fallback for models with >12 layers
        )
        sed = _cosine_distance(orig_embeds[layer_idx - 1], sub_embeds[layer_idx - 1])
        layer_scores.append(LayerSEDScore(
            layer_idx=layer_idx,
            layer_group=group,
            sed=sed,
        ))

    return SentenceSEDResult(
        sentence_idx=sentence_idx,
        original_sentence=sentence,
        modified_sentence=modified,
        layer_scores=layer_scores,
    )


# ---------------------------------------------------------------------------
# SED aggregation and TD computation
# ---------------------------------------------------------------------------

def _aggregate_sed_results(results: list[SentenceSEDResult]) -> dict[int, float]:
    """Average SED per layer index across all sentence results."""
    if not results:
        return {}
    layer_sums: dict[int, list[float]] = {}
    for result in results:
        for score in result.layer_scores:
            layer_sums.setdefault(score.layer_idx, []).append(score.sed)
    return {layer: float(np.mean(vals)) for layer, vals in layer_sums.items()}


def _group_averages(avg_per_layer: dict[int, float]) -> dict[str, float]:
    """Average SED values over each layer group (early/middle/deep)."""
    group_avgs: dict[str, float] = {}
    for group, layer_range in LAYER_GROUPS.items():
        vals = [avg_per_layer[l] for l in layer_range if l in avg_per_layer]
        group_avgs[group] = float(np.mean(vals)) if vals else 0.0
    return group_avgs


def _build_substitute_profile(
    substitute: Substitute,
    sed_t1: list[SentenceSEDResult],
    sed_t2: list[SentenceSEDResult],
) -> SubstituteProfile:
    """Assemble a complete SubstituteProfile from raw SED sentence results."""
    avg_t1 = _aggregate_sed_results(sed_t1)
    avg_t2 = _aggregate_sed_results(sed_t2)

    all_layers = sorted(set(avg_t1) | set(avg_t2))
    td_per_layer = {
        l: abs(avg_t1.get(l, 0.0) - avg_t2.get(l, 0.0))
        for l in all_layers
    }
    td_aggregate = float(np.mean(list(td_per_layer.values()))) if td_per_layer else 0.0

    group_t1 = _group_averages(avg_t1)
    group_t2 = _group_averages(avg_t2)
    group_td = {g: abs(group_t1.get(g, 0.0) - group_t2.get(g, 0.0)) for g in LAYER_GROUPS}

    return SubstituteProfile(
        substitute=substitute,
        sed_t1=sed_t1,
        sed_t2=sed_t2,
        avg_sed_t1=avg_t1,
        avg_sed_t2=avg_t2,
        td_per_layer=td_per_layer,
        td_aggregate=td_aggregate,
        group_avg_t1=group_t1,
        group_avg_t2=group_t2,
        group_td=group_td,
    )


def compute_sed_profiles(
    word: str,
    substitutes: list[Substitute],
    sentences_old: list[str],
    sentences_new: list[str],
    model_name: str = _DEFAULT_BERT_MODEL,
    max_sentences: int = 20,
) -> list[SubstituteProfile]:
    """Compute SED/TD profiles for all substitutes across both corpora.

    Parameters
    ----------
    word : str
        Bare lemma of the target word (used for regex matching in sentences).
    substitutes : list[Substitute]
        Lexical substitutes to evaluate.
    sentences_old / sentences_new : list[str]
        Historical sentences from Corpus 1 and Corpus 2.
    model_name : str
        HuggingFace model identifier (default: ``"bert-base-uncased"``).
    max_sentences : int
        Maximum sentences to process per corpus per substitute (controls cost).

    Returns
    -------
    list[SubstituteProfile]
        One profile per substitute, unsorted.
    """
    model, tokenizer, device = _load_bert_model(model_name)
    profiles: list[SubstituteProfile] = []

    for sub in substitutes:
        logger.info(
            "Computing SED for substitute '%s' (%s)…", sub.base_form, sub.sub_type
        )
        sed_t1 = _collect_sed(
            word, sub, sentences_old[:max_sentences], model, tokenizer, device
        )
        sed_t2 = _collect_sed(
            word, sub, sentences_new[:max_sentences], model, tokenizer, device
        )
        profile = _build_substitute_profile(sub, sed_t1, sed_t2)
        profiles.append(profile)

    return profiles


def _collect_sed(
    word: str,
    substitute: Substitute,
    sentences: list[str],
    model,
    tokenizer,
    device: str,
) -> list[SentenceSEDResult]:
    """Run SED for all *sentences*, skipping those without the target word."""
    results: list[SentenceSEDResult] = []
    for idx, sentence in enumerate(sentences):
        result = _compute_sentence_sed(
            sentence, word, substitute, idx, model, tokenizer, device
        )
        if result is not None:
            results.append(result)
    return results


# ---------------------------------------------------------------------------
# Hypothesis Document
# ---------------------------------------------------------------------------

def _narrative_for_profile(profile: SubstituteProfile, word: str) -> str:
    """Auto-generate a natural language interpretation of a substitute's TD pattern.

    Uses threshold-based heuristics on group-level TD values to produce an
    evidence statement suitable for injection into an agent system prompt.
    """
    sub = profile.substitute
    g_td = profile.group_td
    g_t1 = profile.group_avg_t1
    g_t2 = profile.group_avg_t2

    def _fmt(val: float) -> str:
        return f"{val:.3f}"

    lines: list[str] = []

    # Identify the highest-divergence layer group
    if g_td:
        peak_group = max(g_td, key=g_td.__getitem__)
        peak_td = g_td[peak_group]
        low_td_threshold = 0.05
        high_td_threshold = 0.15

        if peak_td < low_td_threshold:
            lines.append(
                f'"{sub.base_form}" ({sub.sub_type}) showed uniformly low temporal '
                f"divergence across all layer groups (max TD={_fmt(peak_td)}), "
                f'suggesting the semantic relationship between "{word}" and '
                f'"{sub.base_form}" remained stable across time periods.'
            )
        else:
            early_td  = g_td.get("early", 0.0)
            middle_td = g_td.get("middle", 0.0)
            deep_td   = g_td.get("deep", 0.0)

            layer_desc = (
                f'early layers TD={_fmt(early_td)}, '
                f'middle layers TD={_fmt(middle_td)}, '
                f'deep layers TD={_fmt(deep_td)}'
            )
            lines.append(f'"{sub.base_form}" ({sub.sub_type}): {layer_desc}.')

            if deep_td > high_td_threshold and deep_td > early_td + 0.05:
                lines.append(
                    f'The sharp divergence in deep layers (TD={_fmt(deep_td)}) — '
                    f'which encode high-level semantic content — indicates that '
                    f'"{word}" and "{sub.base_form}" were semantically close in '
                    f'the old period (SED={_fmt(g_t1.get("deep", 0.0))}) '
                    f'but diverged significantly in the new period '
                    f'(SED={_fmt(g_t2.get("deep", 0.0))}). '
                    f'This is strong evidence of a genuine diachronic shift.'
                )
            elif early_td > high_td_threshold and early_td > deep_td + 0.05:
                lines.append(
                    f'Divergence is concentrated in early layers (TD={_fmt(early_td)}), '
                    f'suggesting distributional/positional context shifts rather than '
                    f'deep semantic change — consistent with a collocational drift.'
                )
            elif all(g_td.get(g, 0.0) > low_td_threshold for g in ("early", "middle", "deep")):
                lines.append(
                    f'Uniformly high divergence across all layer groups '
                    f'(early={_fmt(early_td)}, middle={_fmt(middle_td)}, '
                    f'deep={_fmt(deep_td)}) suggests broad semantic incompatibility '
                    f'of "{sub.base_form}" as a replacement for "{word}" in the newer period.'
                )
            else:
                lines.append(
                    f'Mixed layer-group divergence; the relationship between '
                    f'"{word}" and "{sub.base_form}" shifted in ways that affect '
                    f'some representational levels more than others.'
                )

    return " ".join(lines)


class HypothesisDocument:
    """Ranked collection of SubstituteProfiles formatted for agent injection.

    Attributes
    ----------
    word : str
        Target word being analyzed.
    t_old / t_new : str
        Period labels for the two corpora.
    ranked_profiles : list[SubstituteProfile]
        Profiles sorted by ``td_aggregate`` descending (highest tension first).
    top_k : int
        Number of top profiles included in formatted outputs.
    model_name : str
        Name of the transformer model used for SED computation.
    generated_at : str
        ISO 8601 UTC timestamp of document creation.
    """

    def __init__(
        self,
        word: str,
        t_old: str,
        t_new: str,
        ranked_profiles: list[SubstituteProfile],
        top_k: int,
        model_name: str,
        generated_at: str,
    ) -> None:
        self.word = word
        self.t_old = t_old
        self.t_new = t_new
        self.ranked_profiles = ranked_profiles
        self.top_k = top_k
        self.model_name = model_name
        self.generated_at = generated_at

    # ------------------------------------------------------------------ #
    # Serialization                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a nested Python dictionary."""
        def _profile_dict(rank: int, p: SubstituteProfile) -> dict[str, Any]:
            return {
                "rank": rank,
                "substitute": {
                    "base_form": p.substitute.base_form,
                    "surface_form": p.substitute.surface_form,
                    "type": p.substitute.sub_type,
                },
                "td_aggregate": round(p.td_aggregate, 5),
                "group_metrics": {
                    group: {
                        "avg_sed_t1": round(p.group_avg_t1.get(group, 0.0), 5),
                        "avg_sed_t2": round(p.group_avg_t2.get(group, 0.0), 5),
                        "td": round(p.group_td.get(group, 0.0), 5),
                        "layer_range": f"layers {list(LAYER_GROUPS[group])[0]}–"
                                       f"{list(LAYER_GROUPS[group])[-1]}",
                    }
                    for group in ("early", "middle", "deep")
                },
                "layer_metrics": {
                    str(l): {
                        "avg_sed_t1": round(p.avg_sed_t1.get(l, 0.0), 5),
                        "avg_sed_t2": round(p.avg_sed_t2.get(l, 0.0), 5),
                        "td": round(p.td_per_layer.get(l, 0.0), 5),
                    }
                    for l in sorted(p.td_per_layer)
                },
                "sentence_coverage": {
                    "t1_sentences_used": len(p.sed_t1),
                    "t2_sentences_used": len(p.sed_t2),
                },
                "narrative": _narrative_for_profile(p, self.word),
            }

        return {
            "hypothesis_document": {
                "word": self.word,
                "t_old": self.t_old,
                "t_new": self.t_new,
                "top_k": self.top_k,
                "model": self.model_name,
                "generated_at": self.generated_at,
                "substitutes": [
                    _profile_dict(rank + 1, p)
                    for rank, p in enumerate(self.ranked_profiles[: self.top_k])
                ],
            }
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    # ------------------------------------------------------------------ #
    # Prompt block for agent injection                                     #
    # ------------------------------------------------------------------ #

    def to_prompt_block(self) -> str:
        """Return a formatted text block ready for injection into system prompts.

        The block presents the tension metrics as data-backed evidence,
        structured to guide the Historical Linguist, Contextual Critic,
        and Consensus Judge agents.
        """
        lines: list[str] = [
            "=" * 72,
            "PRE-DEBATE GROUNDING  —  QUANTITATIVE HYPOTHESIS DOCUMENT",
            "=" * 72,
            f'Target word : "{self.word}"',
            f"Old period  : {self.t_old}",
            f"New period  : {self.t_new}",
            f"Model       : {self.model_name}",
            f"Generated   : {self.generated_at}",
            "",
            "METHODOLOGY",
            "-" * 40,
            "Self-Embedding Distance (SED): cosine distance between the transformer",
            f"  embedding of the original word (\"{self.word}\") and its lexical",
            "  substitute, computed within the same sentence context at each",
            "  transformer layer. SED ∈ [0, 2]; higher = more semantically distant.",
            "Time Difference (TD): |mean_SED(old corpus) − mean_SED(new corpus)|.",
            "  High TD signals that the substitute's relationship to the target",
            "  word changed measurably between historical periods.",
            "",
            f"TOP {min(self.top_k, len(self.ranked_profiles))} SUBSTITUTES "
            f"RANKED BY AGGREGATE TD (highest tension first)",
            "=" * 72,
        ]

        for rank, profile in enumerate(self.ranked_profiles[: self.top_k], start=1):
            sub = profile.substitute
            lines += [
                "",
                f"  RANK {rank}: \"{sub.base_form}\"  [{sub.sub_type}]  "
                f"→ surface form used: \"{sub.surface_form}\"",
                f"  Aggregate TD: {profile.td_aggregate:.4f}  "
                f"(n_old={len(profile.sed_t1)}, n_new={len(profile.sed_t2)} sentences)",
                "",
                "  Layer-group breakdown:",
                "  ┌──────────────┬──────────────┬──────────────┬──────────────┐",
                "  │ Layer group  │ SED (old)    │ SED (new)    │ TD           │",
                "  ├──────────────┼──────────────┼──────────────┼──────────────┤",
            ]
            for group in ("early", "middle", "deep"):
                layer_range = LAYER_GROUPS[group]
                range_str = f"({list(layer_range)[0]}–{list(layer_range)[-1]})"
                sed_t1_val = profile.group_avg_t1.get(group, 0.0)
                sed_t2_val = profile.group_avg_t2.get(group, 0.0)
                td_val = profile.group_td.get(group, 0.0)
                flag = " ◄" if group == max(profile.group_td, key=profile.group_td.__getitem__,
                                            default=group) else ""
                lines.append(
                    f"  │ {group.capitalize():<5} {range_str:<7}"
                    f" │ {sed_t1_val:>10.4f}   │ {sed_t2_val:>10.4f}   "
                    f"│ {td_val:>10.4f}   │{flag}"
                )
            lines.append(
                "  └──────────────┴──────────────┴──────────────┴──────────────┘"
            )

            narrative = _narrative_for_profile(profile, self.word)
            # Word-wrap narrative to 68 chars
            words = narrative.split()
            current = "  Interpretation: "
            for w in words:
                if len(current) + len(w) + 1 > 72:
                    lines.append(current)
                    current = "    " + w
                else:
                    current = current + (" " if not current.endswith(": ") else "") + w
            if current.strip():
                lines.append(current)

        lines += [
            "",
            "=" * 72,
            "AGENT GUIDANCE",
            "-" * 40,
            "• Substitutes with high TD in DEEP layers provide the strongest",
            "  evidence of genuine semantic shift (high-level meaning change).",
            "• High TD in EARLY layers only suggests distributional/positional",
            "  drift, not necessarily core sense change.",
            "• Antonym substitutes with HIGH SED in both periods confirm",
            "  polarity of the original meaning was stable.",
            "• Synonym substitutes with rising SED in the new period indicate",
            "  the core meaning has diverged from that synonym's sense.",
            "• Random substitutes set a baseline: unexpectedly LOW TD for a",
            "  random word may flag distributional confounds in the corpus.",
            "=" * 72,
        ]

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def build_hypothesis_document(
    word: str,
    profiles: list[SubstituteProfile],
    t_old: str,
    t_new: str,
    top_k: int,
    model_name: str,
) -> HypothesisDocument:
    """Sort profiles by TD and wrap them in a HypothesisDocument.

    Parameters
    ----------
    word : str
        Target word being analyzed.
    profiles : list[SubstituteProfile]
        Unsorted substitute profiles from ``compute_sed_profiles()``.
    t_old / t_new : str
        Corpus period labels.
    top_k : int
        Number of top substitutes to highlight in outputs.
    model_name : str
        Name of the transformer model used.

    Returns
    -------
    HypothesisDocument
    """
    ranked = sorted(profiles, key=lambda p: p.td_aggregate, reverse=True)
    return HypothesisDocument(
        word=word,
        t_old=t_old,
        t_new=t_new,
        ranked_profiles=ranked,
        top_k=top_k,
        model_name=model_name,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def run_grounding_pipeline(
    word: str,
    sentences_old: list[str],
    sentences_new: list[str],
    t_old: str = "Corpus 1 (1810–1860)",
    t_new: str = "Corpus 2 (1960–2010)",
    top_k: int = 5,
    n_per_type: int = 3,
    model_name: str = _DEFAULT_BERT_MODEL,
    max_sentences: int = 20,
    generator: Optional[SubstituteGeneratorProtocol] = None,
) -> HypothesisDocument:
    """End-to-end grounding pipeline: substitution → SED → TD → HypothesisDocument.

    Parameters
    ----------
    word : str
        Bare lemma of the target word (e.g., ``"edge"``).
    sentences_old / sentences_new : list[str]
        Historical sentences from the old and new corpora respectively.
    t_old / t_new : str
        Human-readable labels for the two periods (used in output text).
    top_k : int
        Number of top-ranked substitutes to feature in the document.
    n_per_type : int
        Number of candidates to generate per substitute type.
    model_name : str
        HuggingFace model identifier for SED computation.
    max_sentences : int
        Maximum sentences per corpus per substitute to process (cost control).
    generator : SubstituteGeneratorProtocol | None
        Custom substitute generator.  Defaults to ``WordNetSubstituteGenerator``.

    Returns
    -------
    HypothesisDocument
        Call ``.to_prompt_block()`` to get the system-prompt injection string,
        or ``.to_json()`` for the full structured output.

    Example
    -------
    ::

        from mad_sc.pre_debate_grounding import run_grounding_pipeline
        from mad_sc.data_loader import get_semeval_contexts, CORPUS1_LABEL, CORPUS2_LABEL

        sentences_old, sentences_new = get_semeval_contexts("edge_nn", max_samples=30)
        doc = run_grounding_pipeline(
            word="edge",
            sentences_old=sentences_old,
            sentences_new=sentences_new,
            t_old=CORPUS1_LABEL,
            t_new=CORPUS2_LABEL,
            top_k=5,
        )
        prompt_injection = doc.to_prompt_block()
    """
    if not sentences_old and not sentences_new:
        raise ValueError(
            "Both 'sentences_old' and 'sentences_new' are empty. "
            "Load corpus data before calling run_grounding_pipeline()."
        )

    if generator is None:
        generator = WordNetSubstituteGenerator()

    all_sentences = sentences_old + sentences_new
    logger.info("Generating substitutes for '%s'…", word)
    substitutes = generator.generate(word, all_sentences, n_per_type=n_per_type)

    if not substitutes:
        raise RuntimeError(
            f"No substitutes could be generated for '{word}'. "
            "Ensure the word is present in WordNet or configure a custom generator."
        )
    logger.info("Generated %d substitutes. Computing SED…", len(substitutes))

    profiles = compute_sed_profiles(
        word=word,
        substitutes=substitutes,
        sentences_old=sentences_old,
        sentences_new=sentences_new,
        model_name=model_name,
        max_sentences=max_sentences,
    )

    logger.info("Building hypothesis document (top_k=%d)…", top_k)
    return build_hypothesis_document(
        word=word,
        profiles=profiles,
        t_old=t_old,
        t_new=t_new,
        top_k=top_k,
        model_name=model_name,
    )
