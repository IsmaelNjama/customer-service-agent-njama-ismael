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

from langgraph.checkpoint.postgres import PostgresSaver
DB_URI = os.environ["DATABASE_URL"]


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


with PostgresSaver.from_conn_string(DB_URI) as checkpointer:

    checkpointer.setup()

    app = build_graph(
        router_llm,
        executor,
        checkpointer
    )

    def run(query: str, session_id: str):

        config = {
            "configurable": {
                "thread_id": session_id
            }
        }

        result = app.invoke(
            {
                "input": query,
            },
            config=config,
        )

        print("\n── Final Answer ──────────────────────────────────")
        print(result["output"])
        print("──────────────────────────────────────────────────\n")

        return result["output"]

    if __name__ == "__main__":
        import argparse

        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--session",
            type=str,
            required=True,
        )

        args = parser.parse_args()

        print(f"\nSession: {args.session}")

        while True:

            user_input = input("\nYou: ")

            if user_input.lower() == "exit":
                break

            run(user_input, args.session)
