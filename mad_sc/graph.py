"""LangGraph graph construction for the MAD-SC debate pipeline.

Topologies
----------

With grounding (use_grounding=True):

    START ──► grounding ──► team_support ──┐
                       │                    ▼
                       └──► team_refuse ──► judge ──► END

Without grounding (use_grounding=False):

    START ──► team_support ──┐
          │                   ▼
          └──► team_refuse ──► judge ──► END

Both team nodes run in parallel (fan-out).  They converge at the judge (fan-in).
The grounding node computes SED/TD metrics via BERT and injects a
HypothesisDocument prompt block into the team agent system prompts.

Alternative paradigms
---------------------
If the debate architecture yields suboptimal convergence, swap the team nodes for
a single ReAct or Reflexion node without changing the graph topology — the judge
remains agnostic to how the arguments were produced.
"""

import os

from langgraph.graph import END, START, StateGraph

from mad_sc.nodes import (
    grounding_node,
    judge_node,
    oed_context_node,
    self_consistency_judge_node,
    team_refuse_node,
    team_support_node,
)
from mad_sc.state import GraphState

# Default controlled by env var so scripts and the Streamlit UI share one setting.
_GROUNDING_DEFAULT = os.getenv("USE_GROUNDING", "false").lower() not in ("0", "false", "no")
_LEXICOGRAPHER_DEFAULT = os.getenv("USE_LEXICOGRAPHER", "true").lower() not in ("0", "false", "no")
_SC_DEFAULT = os.getenv("USE_SELF_CONSISTENCY", "false").lower() not in ("0", "false", "no")


def compile_graph(
    use_grounding: bool = _GROUNDING_DEFAULT,
    use_lexicographer: bool = _LEXICOGRAPHER_DEFAULT,
    use_self_consistency: bool = _SC_DEFAULT,
):
    """Build and compile the MAD-SC StateGraph.

    Parameters
    ----------
    use_grounding:
        When True, a grounding node runs before the debate teams, injecting
        BERT-based SED/TD evidence into their system prompts.
    use_lexicographer:
        When True, an OED context node runs before the debate teams,
        injecting raw dated quotations from OED/Wiktionary as supplementary
        evidence into team prompts (no LLM synthesis step).
    use_self_consistency:
        When True, replaces the single judge with a self-consistency panel of
        3 judges (Lexicographer / CognitiveSemantist / Sociolinguist) that run
        in parallel and resolve disagreements via majority vote.

    Topology (sequential pre-processing steps, teams always in parallel):

        No flags    : START → (team_support ∥ team_refuse) → judge → END
        grounding   : START → grounding → (teams) → judge → END
        lexicographer: START → lexicographer → (teams) → judge → END
        both        : START → grounding → lexicographer → (teams) → judge → END

    Returns
    -------
    CompiledGraph
        A LangGraph-compiled graph ready to call with ``graph.invoke(state)``.
    """
    builder = StateGraph(GraphState)

    judge_fn = self_consistency_judge_node if use_self_consistency else judge_node

    builder.add_node("team_support", team_support_node)
    builder.add_node("team_refuse", team_refuse_node)
    builder.add_node("judge", judge_fn)

    # Build the upstream chain: START → [grounding →] [lexicographer →] teams
    upstream_tail = START  # the node that feeds into teams (updated below)

    if use_grounding:
        builder.add_node("grounding", grounding_node)
        builder.add_edge(upstream_tail, "grounding")
        upstream_tail = "grounding"

    if use_lexicographer:
        builder.add_node("oed_context", oed_context_node)
        builder.add_edge(upstream_tail, "oed_context")
        upstream_tail = "oed_context"

    # Fan-out to parallel debate teams from whichever node is last upstream.
    builder.add_edge(upstream_tail, "team_support")
    builder.add_edge(upstream_tail, "team_refuse")

    # Fan-in at the judge.
    builder.add_edge("team_support", "judge")
    builder.add_edge("team_refuse", "judge")
    builder.add_edge("judge", END)

    return builder.compile()
