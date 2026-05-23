from schemas.schemas import Route
from prompts.router_prompt import router_prompt


def build_router(llm):
    """
    Returns a callable router chain.

    Usage:
        router_chain = build_router(llm)
        result: Route = router_chain.invoke({"input": "Summarize FEEDBACK category."})
    """
    llm_router = llm.with_structured_output(Route)
    return router_prompt | llm_router
