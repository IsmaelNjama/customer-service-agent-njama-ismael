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
    model="Qwen/Qwen3.5-397B-A17B-fast",
    base_url="https://api.tokenfactory.us-central1.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY")
)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=str, required=True)
    parser.add_argument("--no-mcp", action="store_true",
                        help="Run without MCP server")
    args = parser.parse_args()

    all_tools = list(tools)

    if not args.no_mcp:
        client = MultiServerMCPClient({
            "server": {
                "transport": "streamable_http",
                "url": "http://localhost:8000/mcp"
            }
        })
        mcp_tools = await get_mcp_tools_with_retry(client)

        if mcp_tools:
            all_tools += mcp_tools
        else:
            print("Failed to load MCP tools. Continuing with base tools only")
    else:
        print("Running without MCP tools as per --no-mcp flag")

    if not all_tools:
        raise ValueError("No tools available")

    agent = create_tool_calling_agent(llm, all_tools, system_prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=all_tools,
        max_iterations=10,
        verbose=True
    )

    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.setup()
        app = build_graph(router_llm, executor, checkpointer)

        print(f"\nSession: {args.session}")

        while True:
            user_input = input("\nUser: ")
            if user_input.lower() == "exit":
                break

            config = {"configurable": {"thread_id": args.session}}
            result = await app.ainvoke({"input": user_input}, config=config)
            print("\n── Final Answer ──────────────────────────────────")
            print(result["output"])
            print("──────────────────────────────────────────────────\n")


if __name__ == "__main__":
    asyncio.run(main())
