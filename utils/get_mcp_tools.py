import asyncio

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


async def get_mcp_tools_with_retry(client, retries=MAX_RETRIES, delay=RETRY_DELAY):
    """Attempt to connect to MCP server with retries."""
    for attempt in range(1, retries + 1):
        try:
            mcp_tools = await client.get_tools()
            if mcp_tools:
                print(
                    f"✓ MCP server connected ({len(mcp_tools)} tools loaded)")
                return mcp_tools
            print(
                f"  Attempt {attempt}/{retries}: No tools returned, retrying...")
        except Exception as e:
            print(
                f"  Attempt {attempt}/{retries}: MCP connection failed — {e}")

        if attempt < retries:
            await asyncio.sleep(delay)

    return None  # All attempts failed
