"""Graph state and output schema definitions for MAD-SC."""

from typing import List, Literal, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


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
        Literal["Generalization", "Specialization", "Co-hyponymous Transfer"]
    ] = Field(
        default=None,
        description=(
            "Type of semantic change. Null when verdict is STABLE.\n"
            "  - Generalization: meaning broadened to cover more concepts.\n"
            "  - Specialization: meaning narrowed to a more specific domain.\n"
            "  - Co-hyponymous Transfer: shifted to a semantically related but "
            "    distinct concept at the same level of abstraction."
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
    t_old: str             # Label for the old period, e.g. "Corpus 1 (1810–1860)"
    t_new: str             # Label for the new period, e.g. "Corpus 2 (1960–2010)"
    sentences_old: List[str]   # SemEval corpus1 sentences (1810–1860)
    sentences_new: List[str]   # SemEval corpus2 sentences (1960–2010)

    # --- Debate outputs (populated by parallel Team nodes) ---
    arg_change: str            # Argument for semantic change (Team Support)
    arg_stable: str            # Argument for semantic stability (Team Refuse)

    # --- Final verdict (populated by Judge node) ---
    verdict: Optional[dict]    # JudgeVerdict serialised to dict
