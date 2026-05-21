import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.agents import (
    AgentExecutor,
    create_tool_calling_agent,
)

from tools.tools import tools
from utils.system_prompt import system_prompt


load_dotenv()

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

executor.invoke({"input": "Summarize the FEEDBACK category."})
