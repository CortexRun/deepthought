from typing import TypedDict, Optional, Annotated
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from db.db_models import DocumentChunk


class AgentState(TypedDict):
    user_question: HumanMessage
    translated_question: Optional[AIMessage]
    embedded_translated_question: Optional[list[list[float]]]
    messages: Annotated[
        list[BaseMessage], add_messages
    ]  # add_messages reducer: upserts by message id, so returning the full history does not duplicate
    final_answer: Optional[AIMessage]
    retrieved_chunk: list[DocumentChunk]
    used_tool: Optional[str]