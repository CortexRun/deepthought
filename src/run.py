from fastapi import FastAPI
from pydantic import BaseModel
from settings import settings
from agent_workflow.main_graph import build_main_graph
from context.deepthought_context import DeepthoughtExecContext
from langchain_core.runnables import RunnableConfig

app = FastAPI(title=settings.app_name)


class StartRequest(BaseModel):
    thread_id: str
    user_question: str


@app.post("/start")
def root(request: StartRequest):
    """Main execution Workflow"""
    ctx = DeepthoughtExecContext()
    graph = build_main_graph(ctx=ctx)
    config: RunnableConfig = {"configurable": {"thread_id": request.thread_id}}
    result = graph.invoke({"user_question": request.user_question}, config=config)
    return {"result": result}