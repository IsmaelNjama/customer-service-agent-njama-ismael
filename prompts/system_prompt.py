from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# system instructions
system_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful customer service data analyst assistant. 
You have access to a dataset and various query tools. 
Always answer the user's question accurately based on the tool outputs.

If you are unsure of the answer, state that you cannot find the information."""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
