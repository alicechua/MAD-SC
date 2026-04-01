"""Graph state and output schema definitions for MAD-SC."""

from typing import List, Literal, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


class EtymologyResult(BaseModel):
    """Structured output of the Lexicographer Agent.

    Captures the factual etymological profile of a target word: its historical
    sense, its modern sense, when the shift occurred, and the mechanism responsible.
    This is passed as a Definition Dossier into the debate team system prompts.

    synthesis_reasoning acts as a chain-of-thought scratchpad: the LLM fills it
    first (quote-by-quote sense analysis) before committing to definitions.
    """

    synthesis_reasoning: str = Field(
        description=(
            "Step-by-step reasoning. "
            "(1) For each historical quote, note the sense being expressed. "
            "(2) For each modern quote, note the sense being expressed. "
            "(3) State what changed between the two periods. "
            "(4) Name the best-fit Blank taxonomy mechanism and explain why."
        )
    )
    target_word: str = Field(description="The word being analysed.")
    old_sense_definition: str = Field(
        description=(
            "ONE concise sentence: the dominant sense of the word as attested in the "
            "HISTORICAL quotes (pre-1900). Must be grounded in the actual quote evidence, "
            "not just general knowledge. Start with the word itself, e.g. 'Bead: a small "
            "ball threaded on a rosary, used to count prayers.'"
        )
    )
    new_sense_definition: str = Field(
        description=(
            "ONE concise sentence: the dominant sense of the word as attested in the "
            "MODERN quotes (post-1900). Must reflect what the modern quotes actually show. "
            "Start with the word itself."
        )
    )
    year_of_shift: Optional[int] = Field(
        default=None,
        description=(
            "Approximate year (CE) when the new sense became dominant, based on quote "
            "evidence. Null if the shift is gradual or cannot be dated from the quotes."
        ),
    )
    mechanism_of_change: Optional[
        Literal[
            "Metaphor",
            "Metonymy",
            "Analogy",
            "Generalization",
            "Specialization",
            "Ellipsis",
            "Antiphrasis",
            "Auto-Antonym",
            "Synecdoche",
            "STABLE",
        ]
    ] = Field(
        default=None,
        description=(
            "The single best-fit mechanism from Blank's taxonomy. "
            "Choose STABLE only if old and new senses are functionally identical. "
            "Never leave null if a shift is detectable."
        ),
    )

    def to_dossier_block(self) -> str:
        """Format as a directive prompt block for injection into team system messages."""
        mech = self.mechanism_of_change or "UNKNOWN"
        lines = [
            "=" * 68,
            "LEXICOGRAPHER'S DEFINITION DOSSIER  [treat as ground-truth]",
            "=" * 68,
            f'Target word : "{self.target_word}"',
            "",
            "HISTORICAL SENSE  (corpus OLD period, pre-1900)",
            f"  {self.old_sense_definition}",
            "",
            "MODERN SENSE  (corpus NEW period, post-1900)",
            f"  {self.new_sense_definition}",
        ]
        if self.year_of_shift:
            lines += ["", f"  Estimated shift year : {self.year_of_shift}"]
        lines += [
            "",
            f"  Proposed mechanism   : {mech}",
            "",
            "WHAT THIS MEANS FOR YOUR ARGUMENT",
            "-" * 34,
        ]
        if mech == "STABLE":
            lines += [
                "The Lexicographer finds NO genuine diachronic shift. Both teams should",
                "argue about whether the corpus sentences confirm stability or reveal",
                "subtle variation that the Lexicographer may have missed.",
            ]
        else:
            lines += [
                f"A {mech} shift has been identified. The senses above are the",
                "established historical and modern meanings. Your task is NOT to",
                "re-identify the shift — accept it as given.",
                "",
                "Team Support : find corpus sentences where the word is used in the",
                "  MODERN sense (above). Quote year + text. Show the shift is real.",
                "Team Refuse  : argue the corpus sentences are too ambiguous to confirm",
                "  the shift, OR that the word is used in BOTH senses throughout,",
                "  indicating stable polysemy rather than replacement.",
            ]
        lines.append("=" * 68)
        return "\n".join(lines)


class JudgeVerdict(BaseModel):
    """Structured output schema for the LLM Judge node.

    Enforces constrained decoding over the strict taxonomy defined in the
    MAD-SC paper: Change Types × Causal Drivers × verdict labels.
    """

    word: str = Field(description="The target word being analyzed.")
    verdict: Literal["CHANGE DETECTED", "STABLE"] = Field(
        description="Final verdict on whether genuine diachronic semantic change has occurred."
    )
    change_type: Optional[
        Literal[
            "Metaphor",
            "Metonymy",
            "Analogy",
            "Generalization",
            "Specialization",
            "Ellipsis",
            "Antiphrasis",
            "Auto-Antonym",
            "Synecdoche",
        ]
    ] = Field(
        default=None,
        description=(
            "Fine-grained type of semantic change. Null when verdict is STABLE.\n"
            "  - Metaphor: meaning extended via conceptual similarity (resemblance across domains).\n"
            "  - Metonymy: meaning shifted via contiguity or association (part-for-whole, cause-for-effect).\n"
            "  - Analogy: meaning extended via structural resemblance across semantic domains.\n"
            "  - Generalization: scope broadened to cover a wider range of referents.\n"
            "  - Specialization: scope narrowed to a specific domain or subset.\n"
            "  - Ellipsis: compound phrase shortened; meaning transferred to the head noun alone.\n"
            "  - Antiphrasis: meaning shifted to its opposite through ironic/euphemistic usage.\n"
            "  - Auto-Antonym: word acquired a sense directly opposite to its original meaning.\n"
            "  - Synecdoche: part-for-whole or whole-for-part meaning transfer."
        ),
    )
    causal_driver: Optional[Literal["Cultural Shift", "Linguistic Drift"]] = Field(
        default=None,
        description=(
            "Primary driver of change. Null when verdict is STABLE.\n"
            "  - Cultural Shift: broad societal, technological, or cultural changes.\n"
            "  - Linguistic Drift: internal linguistic processes (metaphor, metonymy, analogy)."
        ),
    )
    break_point_year: Optional[int] = Field(
        default=None,
        description=(
            "Estimated year the semantic shift became the dominant usage. "
            "Null when verdict is STABLE."
        ),
    )
    reasoning: str = Field(
        description=(
            "Detailed reasoning for the verdict, citing evidence from both "
            "Arg_change and Arg_stable."
        )
    )


class GraphState(TypedDict):
    """LangGraph state passed between all nodes in the MAD-SC pipeline."""

    # --- Inputs ---
    word: str
    word_type: str         # Part of speech, e.g. "noun" or "verb"
    t_old: str             # Label for the old period, e.g. "Corpus 1 (1810–1860)"
    t_new: str             # Label for the new period, e.g. "Corpus 2 (1960–2010)"
    sentences_old: List[str]   # SemEval corpus1 sentences (1810–1860)
    sentences_new: List[str]   # SemEval corpus2 sentences (1960–2010)

    # --- Pre-debate grounding (populated by grounding_node) ---
    grounding_block: Optional[str]       # HypothesisDocument.to_prompt_block() or "" if unavailable

    # --- Lexicographer dossier (populated by lexicographer_node, kept for trace compat) ---
    lexicographer_dossier: Optional[str] # EtymologyResult.to_dossier_block() or "" if unavailable

    # --- OED quote block (populated by oed_context_node) ---
    oed_quotes_block: Optional[str]      # Raw dated OED quotes formatted for team prompts

    # --- Debate outputs (populated by parallel Team nodes) ---
    arg_change: str            # Latest argument for semantic change (Team Support)
    arg_stable: str            # Latest argument for semantic stability (Team Refuse)
    tool_calls_support: Optional[List[dict]]  # Tool calls made by Team Support [{tool, args, result}, ...]
    tool_calls_refuse: Optional[List[dict]]   # Tool calls made by Team Refuse [{tool, args, result}, ...]

    # --- Multi-round rebuttal fields (unused in single-round mode) ---
    num_rounds: int            # Total number of rebuttal rounds requested (default 1)
    current_round: int         # Round counter, incremented after each Support→Refuse pass
    debate_history: List[dict] # Full transcript: [{"round": int, "arg_change": str, "arg_stable": str}, ...]

    # --- Final verdict (populated by Judge node) ---
    verdict: Optional[dict]    # JudgeVerdict serialised to dict
