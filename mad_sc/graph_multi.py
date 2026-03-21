"""LangGraph graph construction for the MAD-SC multi-round rebuttal debate pipeline.

Topology
--------
Round 0 (opening):   START → opening_support → opening_refuse_record
                                                       │
                                             should_continue
                                           ┌─────────────────┐
                                    more rounds?         no rounds?
                                           │                  │
                              Round 1..N (rebuttal):       judge → END
                         rebuttal_support → rebuttal_refuse
                                   └──── should_continue ───┘

Round 0 uses the original opening-statement prompts so each team makes its
best independent case. Each rebuttal round uses prompts that quote the
opponent's PREVIOUS argument and ask for a direct counter.

Usage
-----
    from mad_sc.graph_multi import compile_multi_round_graph
    graph = compile_multi_round_graph(num_rounds=3)
    result = graph.invoke(initial_state)
"""

from langgraph.graph import END, START, StateGraph

from mad_sc.nodes import (
    judge_node,
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
    # Record the opening exchange as round 0 in the history.
    history = list(state.get("debate_history", []))
    history.append({
        "round": 0,
        "arg_change": state.get("arg_change", ""),
        "arg_stable": arg_stable,
    })
    return {"arg_stable": arg_stable, "debate_history": history}


def compile_multi_round_graph(num_rounds: int = 3):
    """Build and compile the multi-round MAD-SC StateGraph.

    Parameters
    ----------
    num_rounds:
        Number of rebuttal rounds AFTER the opening exchange.
        - ``0``: opening statements only (Support + Refuse, no rebuttals) → judge.
        - ``1``: opening + 1 rebuttal round each → judge.
        - ``N``: opening + N rebuttal rounds each → judge.

    Returns
    -------
    CompiledGraph
        A LangGraph-compiled graph ready to call with ``graph.invoke(state)``.
    """
    if num_rounds < 0:
        raise ValueError(f"num_rounds must be >= 0, got {num_rounds}")

    builder = StateGraph(GraphState)

    # --- Opening round (round 0) ----------------------------------------
    # Uses original opening-statement prompts; teams don't see each other yet.
    builder.add_node("opening_support", team_support_node)
    builder.add_node("opening_refuse", _opening_refuse_record_node)

    # --- Rebuttal round nodes (rounds 1..N) -----------------------------
    # Each team reads the opponent's last argument and writes a direct counter.
    builder.add_node("rebuttal_support", rebuttal_support_node)
    builder.add_node("rebuttal_refuse", rebuttal_refuse_node)

    # --- Judge (terminal) -----------------------------------------------
    builder.add_node("judge", judge_node)

    # --- Edges ----------------------------------------------------------
    # Opening pass: sequential (Support writes first, then Refuse responds).
    builder.add_edge(START, "opening_support")
    builder.add_edge("opening_support", "opening_refuse")

    # After opening: enter rebuttal loop if num_rounds >= 1, else go to judge.
    # current_round starts at 1; should_continue routes to rebuttal if
    # current_round <= num_rounds.
    builder.add_conditional_edges(
        "opening_refuse",
        should_continue,
        {
            "rebuttal_support": "rebuttal_support",
            "judge": "judge",
        },
    )

    # Rebuttal loop: Support → Refuse → conditional (loop or exit).
    builder.add_edge("rebuttal_support", "rebuttal_refuse")
    builder.add_conditional_edges(
        "rebuttal_refuse",
        should_continue,
        {
            "rebuttal_support": "rebuttal_support",
            "judge": "judge",
        },
    )

    builder.add_edge("judge", END)

    return builder.compile()
