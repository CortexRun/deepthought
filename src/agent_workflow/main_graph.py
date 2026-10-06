from agent_workflow.preprocessing_step.preprocessing_graph import (
    build_preprocessing_graph,
)
from agent_workflow.execution_step.execution_graph import build_execution_graph
from agent_workflow.state import AgentState
from langgraph.graph import StateGraph, START, END
from context.deepthought_context import DeepthoughtExecContext
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()


def build_main_graph(ctx: DeepthoughtExecContext):
    """Builds the main execution workflow"""
    graph = StateGraph(AgentState)

    preprocessing_graph = build_preprocessing_graph(ctx=ctx)
    execution_graph = build_execution_graph(ctx=ctx)

    graph.add_node("Preprocessing", preprocessing_graph)
    graph.add_node("Execution", execution_graph)

    graph.add_edge(START, "Preprocessing")
    graph.add_edge("Preprocessing", "Execution")
    graph.add_edge("Execution", END)

    graph = graph.compile(checkpointer=checkpointer)  # type: ignore

    return graph
