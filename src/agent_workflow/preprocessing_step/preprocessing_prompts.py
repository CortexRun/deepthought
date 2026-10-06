def translate_prompt(user_question: str) -> str:
    """
    Return a strict system prompt that asks the model to translate the user question to English,
    fix minor spelling or grammar errors, and return only the translated text — no explanation.
    """
    prompt = f"""
    You are a precise translation module.

    Task:
    Translate the following user question into natural, grammatically correct English.
    If the text is already in English, simply return it unchanged.
    Fix minor spelling or grammar mistakes silently.

    Return only the translated text — without any explanation, notes, or formatting.

    Text:
    {user_question}
    """
    return prompt.strip()
