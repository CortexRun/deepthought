from agent_workflow.preprocessing_step.preprocessing_prompts import translate_prompt
from agent_workflow.state import AgentState
from context.deepthought_context import DeepthoughtExecContext
from langchain_core.language_models import BaseChatModel

import logging


logger = logging.getLogger("Logger")
logger.setLevel(logging.DEBUG)


def translate_question(state: AgentState, llm: BaseChatModel) -> AgentState:
    """Translate the question of the user in english."""
    
    state["translated_question"] = llm.invoke(
        input=translate_prompt(user_question=state["user_question"])  # type: ignore
    )
    logger.debug(f"State after translate_question: {state}")
    return state


def embed_enhanced_question(state: AgentState, ctx: DeepthoughtExecContext) -> AgentState:
    """Embed the enhanced user question."""
    question = state["translated_question"]
    state["embedded_translated_question"] = ctx.create_multiple_embeddings(  # type: ignore
        question.content  # type: ignore
    )[0]
    logger.debug(f"State after embed_enhanced_question {state}")
    logger.debug(f"question_content: {question.content}")  # type: ignore
    return state
