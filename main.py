from tools.tools import tools
from utils.prompt import prompt
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI


from langchain.agents import create_agent

load_dotenv()

llm = ChatOpenAI(
    model="Qwen/Qwen3.5-397B-A17B-fast",
    base_url="https://api.tokenfactory.us-central1.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY")
)
prompt = prompt
agent = create_agent(
    model=llm,
    tools=[tools],
    system_prompt="""
You are a customer service data analyst assistant.
You have access to a customer service dataset and can query it using the tools provided.
Answer the user's question as accurately as possible using the tools.
"""
)
print(agent.invoke({"input": "What categories exist in the dataset"}))
# agent = create_tool_calling_agent(llm, tools=tools, prompt=prompt)
# executor = AgentExecutor(agent=agent, tools=tools,
#                          max_iterations=10, verbose=True)
# executor.invoke({"input": "What categories exist in the dataset"})
