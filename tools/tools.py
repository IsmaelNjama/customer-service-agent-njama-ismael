from langchain_core.tools import tool
from data.dataset import load_bitext
from schemas.schemas import (ListCategoriesInput, ListCategoriesResponse, CountByIntentInput, CountByIntentResponse, SampleByCategoryInput, SampleByCategoryResponse, SummarizeIntentResponsesInput,
                             SummarizeIntentResponsesResponse, SemanticSearchInput, SemanticSearchResponse, IntentDistributionInput, IntentDistributionResponse, FilterByIntentInput, FilterByIntentResponse)

df = load_bitext()


# @tool(args_schema=ListCategoriesInput)
# def list_categories() -> ListCategoriesResponse:
#     """Lists all unique categories that exist in the customer service dataset."""
#     categories = df['category'].unique().tolist()
#     return ListCategoriesResponse(categories=categories)


# @tool(args_schema=CountByIntentInput)
# def count_by_intent(intent_keyword: str) -> CountByIntentResponse:
#     """Counts how many entries match an intent name or keyword (e.g. 'refund', 'shipping')."""
#     mask = df['intent'].str.contains(intent_keyword, case=False, na=False)
#     count = int(mask.sum())
#     matched_intents = df[mask]['intent'].unique().tolist()
#     return CountByIntentResponse(count=count, matched_intents=matched_intents)


# @tool(args_schema=SampleByCategoryInput)
# def sample_by_category(category: str, n: int = 5) -> SampleByCategoryResponse:
#     """Returns n example instruction/response pairs from a given category (e.g. 'SHIPPING', 'ACCOUNT')."""
#     filtered = df[df['category'].str.upper() == category.upper()]
#     if filtered.empty:
#         return SampleByCategoryResponse(category=category, total_available=0, examples=[])
#     sample = filtered[['intent', 'instruction', 'response']].sample(
#         min(n, len(filtered)))
#     examples = sample.to_dict(orient='records')
#     return SampleByCategoryResponse(
#         category=category,
#         total_available=len(filtered),
#         examples=examples
#     )


@tool(args_schema=SummarizeIntentResponsesInput)
def summarize_intent_responses(intent_keyword: str) -> SummarizeIntentResponsesResponse:
    """Retrieves agent responses for a given intent keyword so they can be summarized (e.g. 'complaint')."""
    mask = df['intent'].str.contains(intent_keyword, case=False, na=False)
    filtered = df[mask]['response']
    if filtered.empty:
        return SummarizeIntentResponsesResponse(
            intent_keyword=intent_keyword,
            total_found=0,
            sampled_responses=[]
        )
    sampled = filtered.sample(min(10, len(filtered))).tolist()
    return SummarizeIntentResponsesResponse(
        intent_keyword=intent_keyword,
        total_found=int(mask.sum()),
        sampled_responses=sampled
    )


@tool(args_schema=SemanticSearchInput)
def semantic_search_instructions(query: str, n: int = 5) -> SemanticSearchResponse:
    """Finds dataset entries where the user instruction semantically matches the query (e.g. 'wanting money back')."""
    mask = df['instruction'].str.contains(query, case=False, na=False)
    results = df[mask][['intent', 'instruction']].head(
        n).to_dict(orient='records')
    return SemanticSearchResponse(query=query, results=results)


@tool(args_schema=IntentDistributionInput)
def intent_distribution_by_category(category: str) -> IntentDistributionResponse:
    """Shows the count of each intent within a given category (e.g. 'ACCOUNT')."""
    filtered = df[df['category'].str.upper() == category.upper()]
    if filtered.empty:
        return IntentDistributionResponse(category=category, total_entries=0, distribution={})
    distribution = filtered['intent'].value_counts().to_dict()
    return IntentDistributionResponse(
        category=category,
        total_entries=len(filtered),
        distribution=distribution
    )


@tool(args_schema=FilterByIntentInput)
def filter_by_intent(intent: str, n: int = 5) -> FilterByIntentResponse:
    """
    Returns n instruction/response pairs for an exact intent name.
    Use when the user specifies a precise intent (e.g. 'get_refund', 'complaint').
    Falls back to fuzzy suggestions if no exact match is found.
    """
    normalized = intent.lower().replace(" ", "_")
    exact = df[df['intent'].str.lower() == normalized]

    if not exact.empty:
        sample = exact[['intent', 'instruction', 'response']].head(n)
        return FilterByIntentResponse(
            intent=normalized,
            total_available=len(exact),
            examples=sample.to_dict(orient='records'),
            suggestions=[]
        )

    # Fuzzy fallback — find intents that contain the keyword
    close_matches = df[
        df['intent'].str.contains(normalized, case=False, na=False)
    ]['intent'].unique().tolist()

    return FilterByIntentResponse(
        intent=normalized,
        total_available=0,
        examples=[],
        suggestions=close_matches
    )


tools = [
    # list_categories,
    # count_by_intent,
    # sample_by_category,
    summarize_intent_responses,
    semantic_search_instructions,
    intent_distribution_by_category,
    filter_by_intent,
]
