"""LangGraph graph construction for the MAD-SC debate pipeline.

Topology
--------

    START ──► team_support ──┐
          │                   ▼
          └──► team_refuse ──► judge ──► END

Both team nodes run in parallel (fan-out from START) and converge at the judge
(fan-in).  LangGraph executes fan-out branches concurrently and fires the judge
only after both predecessors have completed.

Alternative paradigms
---------------------
If the debate architecture yields suboptimal convergence, swap the team nodes for
a single ReAct or Reflexion node without changing the graph topology — the judge
remains agnostic to how the arguments were produced.
"""

from langgraph.graph import END, START, StateGraph

from mad_sc.nodes import judge_node, team_refuse_node, team_support_node
from mad_sc.state import GraphState


def compile_graph():
    """Build and compile the MAD-SC StateGraph.

    Returns
    -------
    CompiledGraph
        A LangGraph-compiled graph ready to call with `graph.invoke(state)`.
    """
    builder = StateGraph(GraphState)

    # Register the three agent nodes.
    builder.add_node("team_support", team_support_node)
    builder.add_node("team_refuse", team_refuse_node)
    builder.add_node("judge", judge_node)

    # Fan-out: START fires both debate teams in parallel.
    builder.add_edge(START, "team_support")
    builder.add_edge(START, "team_refuse")

    # Fan-in: both teams must complete before the judge runs.
    builder.add_edge("team_support", "judge")
    builder.add_edge("team_refuse", "judge")

    # Judge is the terminal node.
    builder.add_edge("judge", END)

    return builder.compile()
