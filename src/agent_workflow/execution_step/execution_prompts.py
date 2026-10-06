from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from agent_workflow.state import AgentState


def general_instructions() -> SystemMessage:
    return SystemMessage(
        content="""
        You are a helpful and professional assistant.

        Keep answers very short.
    """
    )


def intent_classification() -> SystemMessage:
    return SystemMessage(
        content="""
        You are an intent classifier.

        Determine whether the following text is related to operating systems.

        Set intent = "operating_system" if **any** of the following is true:

        - The text explicitly mentions operating system names:
          Linux, Ubuntu, Debian, Arch, Windows, macOS, Android, iOS.

        - The text references core OS concepts, including but not limited to:
          kernel, syscalls, processes, threads, scheduling, synchronization,
          locks, race conditions, memory management, paging, virtual memory,
          filesystem, drivers, boot process, permissions, user/kernel space.

        - The text discusses topics that are primarily associated with OS-level functionality:
          process or thread models, inter-process communication (IPC),
          concurrency mechanisms, context switches, resource management,
          hardware abstraction, system startup.

        If none of these conditions apply, set:
        intent = "general".
        """
    )


def rephrase_answer(state: AgentState) -> list[BaseMessage]:
    translated = state.get("translated_question")
    translated_text = (
        translated.content if translated and hasattr(translated, "content") else ""
    )

    retrieved = state.get("retrieved_chunk")
    used_tool = state.get("used_tool")

    if retrieved and not isinstance(retrieved, list):
        retrieved = [retrieved]

    retrieved_text = ""
    if retrieved:
        retrieved_text = " | ".join(
            f"[Title: {c.doc_title}, Page: {c.page_number}] {c.text}"
            for c in retrieved
        )

    messages = state.get("messages", [])
    last_messages = messages[-4:] 

    chat_history_msgs: list[BaseMessage] = []

    for m in last_messages:
        if isinstance(m, BaseMessage):
            chat_history_msgs.append(m)
        else:

            chat_history_msgs.append(
                HumanMessage(content=str(getattr(m, "content", m)))
            )

    if used_tool == "search_rag":
        system_prompt = SystemMessage(
            content=(
                "Write a concise and strictly accurate answer in English. "
                "Use ONLY the provided reference text. "
                "Cite page numbers like (Page X). "
                f"Reference text: {retrieved_text}"
            )
        )
    else:
        system_prompt = SystemMessage(
            content=(
                "Give a short and natural answer in English. "
                "Respond directly to the user's latest message."
            )
        )

    prompt_messages: list[BaseMessage] = [
        system_prompt,
        *chat_history_msgs,
        HumanMessage(content=translated_text),
    ]

    return prompt_messages