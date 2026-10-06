from langchain_core.runnables import Runnable
from agent_workflow.state import AgentState
from agent_workflow.execution_step.models import Intent
from context.deepthought_context import DeepthoughtExecContext
from db.db_procedures import query_similar_chunks, get_db_credentials
from db.db_models import DocumentChunk
from agent_workflow.execution_step.execution_prompts import (
    intent_classification,
    general_instructions,
    rephrase_answer,
)
import logging


logger = logging.getLogger("Logger")
logger.setLevel(logging.DEBUG)


def search_rag(state: AgentState, ctx: DeepthoughtExecContext) -> AgentState:
    """
    Use this tool to answer questions specifically about operating systems.
    Examples include Windows, Linux, macOS, Android, iOS, kernels, or file systems.

    Input:
        query (str): The user's question or topic related to operating systems.

    Output:
        A short factual summary or retrieved information about the operating system(s).
    """
    host, port, db, user, password = get_db_credentials()
    conn = ctx.establish_db_connection(
        host=host, port=port, dbname=db, user=user, password=password
    )
    try:
        results = query_similar_chunks(conn, state["embedded_translated_question"], top_k=2)
    finally:
        conn.close()

    chunks = []
    for r in results:
        chunk = DocumentChunk(
            id=str(r[0]),
            doc_title=r[1],
            page_number=r[2],
            chunk_id=r[3],
            text=r[4],
            embedding=None
        )
        chunks.append(chunk)

    state["retrieved_chunk"] = chunks

    logger.debug(f"State after search_rag state: {state}")
    return state


def write_answer(state: AgentState, llm: Runnable) -> AgentState:
    prompt = rephrase_answer(state=state)

    state["final_answer"] = llm.invoke(prompt)
    logger.debug(f"State after write_answer: {state}")
    return state


def llm_call(state: AgentState, llm: Runnable) -> AgentState:
    history = state.get("messages", [])
    q_msg = state.get("translated_question") or state.get("user_question")

    prompt = [general_instructions(), *history]
    if q_msg:
        prompt.append(q_msg)

    ai = llm.invoke(prompt)

    updated_history = history[:]
    if q_msg:
        updated_history.append(q_msg)  
    updated_history.append(ai)

    logger.debug(f"State after llm_call: {state}")

    return {**state, "messages": updated_history}


def reasoner(state: AgentState, llm_struct):
    state["retrieved_chunk"] = []

    msg = state.get("translated_question") or state.get("user_question")
    system_message = intent_classification()

    prompt = f"{system_message.content.strip()}\n\nText: {msg.content.strip()}"  # type: ignore
    result = llm_struct.invoke(prompt)

    if result.intent == Intent.GENERAL:
        state["used_tool"] = "llm_call"
    elif result.intent == Intent.OPERATING_SYSTEM:
        state["used_tool"] = "search_rag"
    else:
        raise ValueError(f"Unhandled intent: {result.intent!r}")
    logger.debug(f"State after reasoner: {state}")
    return state
