from langgraph.graph import StateGraph, START, END
from agent_workflow.preprocessing_step.preprocessing_nodes import (
    translate_question,
    embed_enhanced_question,
)
from agent_workflow.state import AgentState
from context.deepthought_context import DeepthoughtExecContext


def build_preprocessing_graph(ctx: DeepthoughtExecContext):
    """Builds the preprocessing graph (Translate, enhance, embed)"""

    llm = ctx.create_llm()
    graph = StateGraph(AgentState)

    graph.add_node("Translate", lambda state: translate_question(state, llm=llm))
    graph.add_node("Embed", lambda state: embed_enhanced_question(state, ctx=ctx))
    graph.add_edge(START, "Translate")
    graph.add_edge("Translate", "Embed")
    graph.add_edge("Embed", END)

    subgraph = graph.compile()

    return subgraph
