"""
LangGraph pipeline with PostgreSQL checkpoint persistence
"""

from __future__ import annotations

from typing import TypedDict, Annotated
import operator
import os

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END


from router.router import build_router
from schemas.schemas import Route


# Shared graph state
class AgentState(TypedDict):
    input: str
    messages: Annotated[list[BaseMessage], operator.add]
    query_type: str
    output: str


# Router node
def make_router_node(router_chain):

    def router_node(state: AgentState) -> dict:

        history = "\n".join(
            [msg.content for msg in state.get("messages", [])]
        )

        result: Route = router_chain.invoke(
            {
                "input": state["input"],
                "history": history,
            }
        )

        print(f"\n[Router] type={result.query_type!r}")

        return {
            "query_type": result.query_type
        }

    return router_node


# Agent node
def make_agent_node(executor):

    async def agent_node(state: AgentState) -> dict:

        result = await executor.ainvoke(
            {
                "input": state["input"],
                "chat_history": state["messages"],
            }
        )

        answer = result.get("output", "")

        return {
            "output": answer,
            "messages": [HumanMessage(content=state["input"]), AIMessage(content=answer)],
        }

    return agent_node

# Decline node


def decline_node(state: AgentState) -> dict:

    message = (
        "I'm sorry, but that question is outside my area of expertise. "
        "I'm designed to help with structured business data queries and analysis."
    )

    return {
        "output": message,
        "messages": [AIMessage(content=message)],
    }


# Routing logic
def route_decision(state: AgentState) -> str:

    query_type = state.get("query_type", "out_of_scope")

    if query_type in ("structured", "unstructured"):
        return "agent"

    return "decline"


# Build graph
def build_graph(router_llm, executor, checkpointer):

    router_chain = build_router(router_llm)

    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("router", make_router_node(router_chain))
    graph.add_node("agent", make_agent_node(executor))
    graph.add_node("decline", decline_node)

    # Edges
    graph.add_edge(START, "router")

    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "agent": "agent",
            "decline": "decline",
        },
    )

    graph.add_edge("agent", END)
    graph.add_edge("decline", END)

    return graph.compile(
        checkpointer=checkpointer
    )
