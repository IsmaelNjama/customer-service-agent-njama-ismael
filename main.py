import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.agents import (
    AgentExecutor,
    create_tool_calling_agent,
)

from tools.tools import tools
from utils.system_prompt import system_prompt

from graph.graph import build_graph


load_dotenv()

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


agent = create_tool_calling_agent(llm, tools, system_prompt)

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=10,
    verbose=True
)

# executor.invoke({"input": "Summarize the FEEDBACK category."})

app = build_graph(router_llm, executor)


def run(query: str) -> str:
    """Run a query through the router → agent pipeline."""
    result = app.invoke({"input": query, "messages": []})
    print("\n── Final Answer ──────────────────────────────────")
    print(result["output"])
    print("──────────────────────────────────────────────────\n")
    return result["output"]


if __name__ == "__main__":
    run("What is the distribution of intents in the ACCOUNT category?")
