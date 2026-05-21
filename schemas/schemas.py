from pydantic import BaseModel, Field


# --- list_categories ---
class ListCategoriesInput(BaseModel):
    pass


class ListCategoriesResponse(BaseModel):
    categories: list[str]


# --- count_by_intent ---
class CountByIntentInput(BaseModel):
    intent_keyword: str = Field(
        description="The intent keyword to search for, e.g. 'refund', 'shipping', 'billing'")


class CountByIntentResponse(BaseModel):
    count: int
    matched_intents: list[str]


# --- sample_by_category ---
class SampleByCategoryInput(BaseModel):
    category: str = Field(
        description="The category name to filter by, e.g. 'SHIPPING', 'ACCOUNT', 'REFUNDS'")
    n: int = Field(
        default=5,
        description="Number of examples to return")


class SampleByCategoryResponse(BaseModel):
    category: str
    total_available: int
    examples: list[dict]  # each dict has keys: intent, instruction, response


# --- summarize_intent_responses ---
class SummarizeIntentResponsesInput(BaseModel):
    intent_keyword: str = Field(
        description="The intent keyword to retrieve responses for, e.g. 'complaint', 'refund'")


class SummarizeIntentResponsesResponse(BaseModel):
    intent_keyword: str
    total_found: int
    sampled_responses: list[str]


# --- semantic_search_instructions ---
class SemanticSearchInput(BaseModel):
    query: str = Field(
        description="Natural language query to match against user instructions, e.g. 'wanting money back'")
    n: int = Field(
        default=5,
        description="Number of results to return")


class SemanticSearchResponse(BaseModel):
    query: str
    results: list[dict]  # each dict has keys: intent, instruction


# --- intent_distribution_by_category ---
class IntentDistributionInput(BaseModel):
    category: str = Field(
        description="The category to get intent distribution for, e.g. 'ACCOUNT', 'SHIPPING'")


class IntentDistributionResponse(BaseModel):
    category: str
    total_entries: int
    distribution: dict[str, int]  # intent -> count


# --- filter_by_intent ---
class FilterByIntentInput(BaseModel):
    intent: str = Field(
        description="Exact intent name to filter by, e.g. 'get_refund', 'complaint', 'cancel_order'")
    n: int = Field(
        default=5,
        description="Number of examples to return")


class FilterByIntentResponse(BaseModel):
    intent: str
    total_available: int
    # each dict has keys: intent, instruction, response
    examples: list[dict]
    # populated if no exact match found, empty otherwise
    suggestions: list[str]
