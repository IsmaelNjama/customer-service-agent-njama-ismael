import signal
import sys
from fastmcp import FastMCP
from datasets import load_dataset
import pandas as pd

mcp = FastMCP("server")


def shutdown_handler(signum, frame):
    print("\nShutting down MCP server ...")
    sys.exit(0)


signal.signal(signal.SIGINT, shutdown_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, shutdown_handler)  # docker/system stop


def load_bitext():
    ds = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
    )
    df = pd.DataFrame(ds["train"])
    return df


try:
    df = load_bitext()
    print(f"Dataset loaded: {len(df)} rows")
except Exception as e:
    print(f"Failed to load dataset: {e}")
    df = None


@mcp.tool
def list_categories() -> list:
    """Lists all unique categories that exist in the customer service dataset."""
    if df is None:
        raise RuntimeError("Dataset not loaded")

    return df["category"].unique().tolist()


@mcp.tool()
def count_by_intent(intent_keyword: str) -> dict:
    """Counts how many entries match an intent name or keyword (e.g. 'refund', 'shipping')."""
    mask = df['intent'].str.contains(intent_keyword, case=False, na=False)
    count = int(mask.sum())
    matched_intents = df[mask]['intent'].unique().tolist()
    return {"count": count, "matched_intents": matched_intents}


@mcp.tool()
def sample_by_category(category: str, n: int = 5) -> dict:
    """Returns n example instruction/response pairs from a given category (e.g. 'SHIPPING', 'ACCOUNT')."""
    filtered = df[df['category'].str.upper() == category.upper()]
    if filtered.empty:
        return {"category": category, "total_available": 0, "examples": []}
    sample = filtered[['intent', 'instruction', 'response']].sample(
        min(n, len(filtered)))
    examples = sample.to_dict(orient='records')
    return {"category": category,
            "total_available": len(filtered),
            "examples": examples}


if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8000)
