"""LangGraph graph construction for the MAD-SC multi-round rebuttal debate pipeline.

Topology
--------
Round 0 (opening):   START → opening_support → opening_refuse_record
                                                       │
                                             should_continue
                                           ┌─────────────────┐
                                    more rounds?         no rounds?
                                           │                  │
                              Round 1..N (rebuttal):  closing_support → judge → END
                         rebuttal_support → rebuttal_refuse
                                   └──── should_continue ────┘
                                              │ (exhausted)
                                       closing_support → judge → END

Round 0 uses the opening-statement prompts so each team makes its best
independent case.  Each rebuttal round quotes the opponent's PREVIOUS argument.
After all rebuttals, a closing_support step lets Team Support respond one final
time to Team Refuse's last argument — neutralising the last-word advantage.
The judge receives the FULL debate transcript, not just the final round pair.

Usage
-----
    from mad_sc.graph_multi import compile_multi_round_graph
    graph = compile_multi_round_graph(num_rounds=3)
    result = graph.invoke(initial_state)
"""

from langgraph.graph import END, START, StateGraph

from mad_sc.graph import _GROUNDING_DEFAULT, _LEXICOGRAPHER_DEFAULT
from mad_sc.nodes import (
    closing_refuse_node,
    closing_support_node,
    grounding_node,
    judge_node,
    lexicographer_node,
    rebuttal_refuse_node,
    rebuttal_support_node,
    should_continue,
    team_refuse_node,
    team_support_node,
)
from mad_sc.state import GraphState


def _opening_refuse_record_node(state: GraphState) -> dict:
    """Run Team Refuse's opening statement and record round 0 in debate_history."""
    result = team_refuse_node(state)
    arg_stable = result["arg_stable"]
    history = list(state.get("debate_history", []))
    history.append({
        "round": 0,
        "arg_change": state.get("arg_change", ""),
        "arg_stable": arg_stable,
    })
    return {"arg_stable": arg_stable, "debate_history": history}


def compile_multi_round_graph(
    num_rounds: int = 3,
    use_grounding: bool = _GROUNDING_DEFAULT,
    use_lexicographer: bool = _LEXICOGRAPHER_DEFAULT,
):
    """Build and compile the multi-round MAD-SC StateGraph.

    Parameters
    ----------
    num_rounds:
        Number of rebuttal rounds AFTER the opening exchange.
        - ``0``: opening statements only (Support + Refuse, no rebuttals) → closing → judge.
        - ``1``: opening + 1 rebuttal round each → closing → judge.
        - ``N``: opening + N rebuttal rounds each → closing → judge.
    use_grounding:
        When True, runs the BERT-based grounding node before the opening round.
        The ``grounding_block`` is then available to all team nodes (opening + rebuttal).
    use_lexicographer:
        When True, runs the Lexicographer Agent before the opening round to produce
        a Definition Dossier. The dossier is injected into all team and judge prompts.

    Returns
    -------
    CompiledGraph
        A LangGraph-compiled graph ready to call with ``graph.invoke(state)``.
    """
    if num_rounds < 0:
        raise ValueError(f"num_rounds must be >= 0, got {num_rounds}")

    builder = StateGraph(GraphState)

    # --- Pre-debate nodes (optional) ------------------------------------
    # Build the upstream chain: START → [grounding →] [lexicographer →] opening_support
    upstream_tail = START

    if use_grounding:
        builder.add_node("grounding", grounding_node)
        builder.add_edge(upstream_tail, "grounding")
        upstream_tail = "grounding"

    if use_lexicographer:
        builder.add_node("lexicographer", lexicographer_node)
        builder.add_edge(upstream_tail, "lexicographer")
        upstream_tail = "lexicographer"

    # --- Opening round (round 0) ----------------------------------------
    builder.add_node("opening_support", team_support_node)
    builder.add_node("opening_refuse", _opening_refuse_record_node)

    # --- Rebuttal round nodes (rounds 1..N) -----------------------------
    builder.add_node("rebuttal_support", rebuttal_support_node)
    builder.add_node("rebuttal_refuse", rebuttal_refuse_node)

    # --- Closing rounds -------------------------------------------------
    # Both teams get a closing statement to summarize their arguments.
    # Refuse goes first in closing, then Support gets the final word.
    builder.add_node("closing_refuse", closing_refuse_node)
    builder.add_node("closing_support", closing_support_node)

    # --- Judge (terminal) -----------------------------------------------
    builder.add_node("judge", judge_node)

    # --- Edges ----------------------------------------------------------
    # Opening pass: sequential (Support writes first, then Refuse responds).
    builder.add_edge(upstream_tail, "opening_support")
    builder.add_edge("opening_support", "opening_refuse")

    # After opening: enter rebuttal loop or go straight to closing.
    # should_continue returns "judge" when exhausted; we remap that to
    # "closing_refuse" so the judge always sees both closings first.
    builder.add_conditional_edges(
        "opening_refuse",
        should_continue,
        {
            "rebuttal_support": "rebuttal_support",
            "judge": "closing_refuse",
        },
    )

    # Rebuttal loop: Support → Refuse → conditional (loop or closing).
    builder.add_edge("rebuttal_support", "rebuttal_refuse")
    builder.add_conditional_edges(
        "rebuttal_refuse",
        should_continue,
        {
            "rebuttal_support": "rebuttal_support",
            "judge": "closing_refuse",
        },
    )

    builder.add_edge("closing_refuse", "closing_support")
    builder.add_edge("closing_support", "judge")
    builder.add_edge("judge", END)

    return builder.compile()
