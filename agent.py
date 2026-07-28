from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.agents import (
    AgentExecutor,
    create_tool_calling_agent,
)

from tools.tools import tools
from prompts.system_prompt import system_prompt

from graph.graph import build_graph


from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
import argparse
from utils.get_mcp_tools import get_mcp_tools_with_retry

from uuid import UUID

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from schemas.schemas import ChatRequest

load_dotenv()
DB_URI = os.environ["DATABASE_URL"]


router_llm = ChatOpenAI(
    model="google/gemma-3-27b-it",
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
    temperature=0.0,
    max_tokens=20,
)

llm = ChatOpenAI(
    model="Qwen/Qwen3-235B-A22B-Instruct-2507",
    base_url="https://api.tokenfactory.us-central1.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY")
)


app = FastAPI()


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    user_input = request.message
    user_id = request.user_id

    if not user_input or not user_id:
        raise HTTPException(
            status_code=400, detail="Missing 'input' or 'user_id' in request body")

    all_tools = list(tools)

    if not all_tools:
        raise ValueError("No tools available")

    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.setup()
        all_tools = list(tools)
        agent = create_tool_calling_agent(llm, all_tools, system_prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=all_tools,
            max_iterations=10,
            verbose=True
        )
        app_graph = build_graph(router_llm, executor, checkpointer)

        config = {"configurable": {"thread_id": user_id}}
        result = await app_graph.ainvoke({"input": user_input}, config=config)
        answer = result["output"]

        async def stream_response():
            yield answer

        return StreamingResponse(stream_response(), media_type="text/plain")
