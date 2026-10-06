from langgraph.graph import START, END, StateGraph
from agent_workflow.execution_step.execution_tools import (
    search_rag,
    llm_call,
    reasoner,
    write_answer,
)
from context.deepthought_context import DeepthoughtExecContext
from agent_workflow.state import AgentState
from agent_workflow.execution_step.models import IntentResult
import logging


logger = logging.getLogger("Logger")
logger.setLevel(logging.DEBUG)


def build_execution_graph(ctx: DeepthoughtExecContext):
    """Builds the execution graph"""
    llm = ctx.create_llm()
    llm_struct = ctx.create_llm(structured_output_schema=IntentResult)
    
    graph = StateGraph(AgentState)
    graph.add_node("reasoner_d", lambda state: reasoner(state, llm_struct))
    graph.add_node("llm_call_d", lambda state: llm_call(state, llm))
    graph.add_node("search_rag_d", lambda state: search_rag(state, ctx=ctx))
    graph.add_node("write_answer", lambda state: write_answer(state, llm))

    graph.add_edge(START, "reasoner_d")
    graph.add_conditional_edges(
        "reasoner_d",
        lambda state: state["used_tool"],
        {"llm_call": "llm_call_d", "search_rag": "search_rag_d"},
    )
    graph.add_edge("llm_call_d", "write_answer")
    graph.add_edge("search_rag_d", "write_answer")
    graph.add_edge("write_answer", END)

    execution_graph = graph.compile()

    return execution_graph
