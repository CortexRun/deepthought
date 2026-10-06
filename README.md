# deepthought

A local RAG assistant that runs entirely on a Jetson Orin Nano (8 GB). No cloud, no API keys: Ollama for the LLM and embeddings, PostgreSQL + pgvector for retrieval, LangGraph for the control flow, FastAPI in front.

Built as a university team project (FOM, 2025–2026); I implemented the whole stack. The interesting part is not the RAG itself but what 8 GB of shared memory forces you to do differently.

<!-- TODO Andi: ein Satz, was der Use Case war (Fragen zum OS-Lehrbuch beantworten, Zielgruppe?) -->

## What it does

1. **Preprocessing** – translates the question to English (small models are noticeably better at English) and embeds it (`nomic-embed-text`).
2. **Routing** – a structured-output call classifies the intent: `operating_system` → retrieval, `general` → plain chat.
3. **Execution** – either a pgvector similarity search over the chunked textbook, or a short chat completion over the last four messages.
4. **Answer** – one final rephrasing call. Retrieval answers must cite page numbers `(Page X)` and may only use the retrieved text.

![graph](docs/graph.png)

## Why it looks like this

**Routing instead of ReAct.** The first version was a ReAct agent with the retrieval as a tool. On a 3B model that was not stable: <!-- TODO Andi: was genau ist passiert? Tool-Calls kaputt formatiert / Endlosschleifen / Halluzinierte Tool-Namen? Welches Modell? --> Replacing the loop with one structured-output classification (`IntentResult`, Pydantic schema enforced by Ollama) made the behaviour deterministic enough to ship: the model decides *once*, the graph does the rest.

**Translate first.** <!-- TODO Andi: warum der Übersetzungsschritt — deutsche Fragen von Nutzern, aber Korpus und kleine Modelle englisch? Hat es messbar geholfen? -->

**Bounding memory and latency.** Everything on the Jetson shares the same 8 GB with the OS. The knobs that matter:
- chat history is truncated to the last 4 messages before the final call,
- `top_k = 2` chunks of 500 characters (100 overlap) per retrieval,
- `num_predict = 2048`, `temperature = 0.2`.

<!-- TODO Andi: eine Zahl wäre gut — Antwortlatenz p50 auf dem Jetson, oder RAM-Verbrauch mit granite4:3b + nomic-embed geladen -->

**Models.** `LlmApiModel` in `src/context/deepthought_context.py` is the list of what I tried. `granite4:3b` is the one that runs reliably on the device. <!-- TODO Andi: 1–2 Sätze, warum die anderen rausgefallen sind (zu langsam / zu viel RAM / Structured Output nicht stabil) -->

**Chunking by characters, per page.** Pages are chunked individually so every chunk carries its page number, which is what makes the `(Page X)` citations possible. Character-based rather than token-based chunking keeps the ingestion dependency-free.

## Stack

LangGraph 1.x (two compiled subgraphs inside a main graph, `InMemorySaver` checkpointer keyed by `thread_id`) · LangChain-Ollama · PostgreSQL + pgvector (768-dim, cosine) · FastAPI/uvicorn · pypdf · uv

## Running it

```bash
# 1. Ollama with the models
ollama pull granite4:3b
ollama pull nomic-embed-text

# 2. PostgreSQL with pgvector, then
cp .env.example .env   # fill in PG_* (or point PGPASS_PATH at a .pgpass file)

# 3. Install and ingest the corpus (see demo/db_insert.ipynb)
uv sync

# 4. Serve
uv run uvicorn run:app --app-dir src --host 0.0.0.0 --port 8000
```

```bash
curl -X POST localhost:8000/start \
  -H 'content-type: application/json' \
  -d '{"thread_id": "demo", "user_question": "Was ist ein Context Switch?"}'
```

The corpus used during the project is Max Hailperin's *Operating Systems and Middleware* (<!-- TODO Andi: Lizenz prüfen; meines Wissens CC BY-SA 3.0, dann hier verlinken und das PDF aus dem Repo nehmen -->).

## What I would do differently

- Connection pooling instead of one Postgres connection per request.
- Return partial state updates from the LangGraph nodes instead of mutating the `TypedDict` in place; it works, but it is not idiomatic.
- Token-based chunking and a proper eval set: right now "does it cite the right page" is checked by hand.
- <!-- TODO Andi: dein eigener Punkt -->

## Layout

```
src/
  run.py                      FastAPI entrypoint
  settings.py                 configuration (env / .env / .pgpass)
  agent_workflow/
    main_graph.py             Preprocessing -> Execution
    preprocessing_step/       translate + embed
    execution_step/           intent routing, retrieval, chat, final answer
    state.py                  AgentState
  context/                    Ollama client + model registry
  db/                         pgvector schema, ingestion, similarity search
demo/                         ingestion and client notebooks
exploration/                  scratch notebooks from development
```
