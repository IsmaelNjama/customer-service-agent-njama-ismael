"""
LangGraph pipeline
------------------
Graph layout:

  [START] ──► router_node ──► (conditional)
                                  ├─ "structured"   ──► agent_node ──► [END]
                                  ├─ "unstructured" ──► agent_node ──► [END]
                                  └─ "out_of_scope" ──► decline_node ──► [END]

The router runs BEFORE the agent touches any tool, so out-of-scope
queries never reach tool selection.
"""

from __future__ import annotations

from typing import TypedDict, Annotated
import operator

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END

from router.router import build_router
from schemas.schemas import Route


# Shared graph state
class AgentState(TypedDict):
    input: str                                          # raw user query
    messages: Annotated[list[BaseMessage],
                        operator.add]                   # conversation history
    query_type: str                                     # set by router node
    output: str                                         # final answer


# Node builders
def make_router_node(router_chain):
    """Returns the router node function."""

    def router_node(state: AgentState) -> dict:
        result: Route = router_chain.invoke({"input": state["input"]})
        print(
            f"\n[Router] type={result.query_type!r}  "
            # f"reason={result.reasoning!r}\n"
        )
        return {"query_type": result.query_type}

    return router_node


def make_agent_node(executor):
    """Wraps AgentExecutor as a LangGraph node."""

    def agent_node(state: AgentState) -> dict:
        result = executor.invoke({"input": state["input"]})
        answer = result.get("output", "")
        return {
            "output": answer,
            "messages": [AIMessage(content=answer)],
        }

    return agent_node


def decline_node(state: AgentState) -> dict:
    """Politely declines out-of-scope queries without using the LLM."""
    message = (
        "I'm sorry, but that question is outside my area of expertise. "
        "I'm designed to help with structured business data queries and analysis. "
        "Please ask me something related to your data, categories, reports, or metrics."
    )
    return {
        "output": message,
        "messages": [AIMessage(content=message)],
    }


# Routing condition
def route_decision(state: AgentState) -> str:
    """Maps query_type to the next node name."""
    query_type = state.get("query_type", "out_of_scope")
    if query_type in ("structured", "unstructured"):
        return "agent"
    return "decline"


# Graph factory
def build_graph(router_llm, executor):
    """
    Assembles and compiles the full LangGraph pipeline.

    Parameters
    ----------
    llm      : the LLM instance (must support .with_structured_output)
    executor : a LangChain AgentExecutor

    Returns
    -------
    A compiled LangGraph app; call  app.invoke({"input": "..."})
    """
    router_chain = build_router(router_llm)

    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("router",  make_router_node(router_chain))
    graph.add_node("agent",   make_agent_node(executor))
    graph.add_node("decline", decline_node)

    # Edges
    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "agent":   "agent",
            "decline": "decline",
        },
    )
    graph.add_edge("agent",   END)
    graph.add_edge("decline", END)

    return graph.compile()
