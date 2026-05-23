from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate


# ROUTER_SYSTEM = """
# You are a strict query classifier.

# Classify the user's query into EXACTLY ONE label:

# - structured
# - unstructured
# - out_of_scope

# Rules:
# - Return ONLY the label.
# - Do not explain.
# - Do not add punctuation.
# - Do not think step-by-step.
# - Do not output anything else.
# """

from langchain_core.prompts import ChatPromptTemplate


ROUTER_SYSTEM = """\
You are a strict query classifier for a data-analysis agent.

You MUST consider BOTH:
- the current user query
- the previous conversation history

A follow-up question may depend on previous context.

Examples:
- "What about the FEEDBACK category?"
- "And the last two?"
- "What is the total count of both?"
- "Can you summarize that one?"

These should still be classified IN DOMAIN if the conversation history is about business/customer-service data.

STEP 1 – Domain check:
Is the CURRENT query OR its conversational context related to:
- customer service data
- feedback categories
- intents
- business metrics
- reports
- aggregations
- structured datasets

If YES:
  continue classification.

If NO:
  classify as out_of_scope.

STEP 2 – Classification:
• structured
  specific retrievals, counts, filters, aggregations

• unstructured
  reasoning, trends, insights, recommendations

• out_of_scope
  unrelated requests

Be strict, but understand conversational follow-ups.
"""


router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ROUTER_SYSTEM),
        ("human", "Conversation history:\n{history}\n\nUser query:\n{input}"),
    ]
)
