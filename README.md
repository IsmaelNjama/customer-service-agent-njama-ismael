# Customer Service Agent

A LangGraph-powered customer service agent with PostgreSQL checkpoint persistence, a router LLM for query classification, and optional MCP tool integration.

---

## Prerequisites

- Python 3.10+
- Git
- Docker
- [Nebius](https://tokenfactory.nebius.com/) Token Factory API key

---

## Installation

```bash
# Clone the repository
git clone https://github.com/IsmaelNjama/customer-service-agent-njama-ismael.git
cd customer-service-agent-njama-ismael

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Create `.env` and fill in your credentials:

```bash
touch .env
```

```env
NEBIUS_API_KEY=your_nebius_api_key
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/langgraph
```

---

## Database

Make sure the Docker engine is running, then start a PostgreSQL container:

```bash
docker run --name langgraph-postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=langgraph \
  -p 5432:5432 \
  -d postgres:16
```

---

## Running the Server

### Without MCP

```bash
python main.py --session my_session --no-mcp
```

```
Running without MCP tools as per --no-mcp flag

Session: my_session

User:
```

### With MCP

Start the MCP server in the background first, then connect the agent to it.

**1. Start the MCP server:**

```bash
cd mcp_server
python server.py &
cd ..
```

The server starts on `http://localhost:8000/mcp`. Wait a moment for the dataset to load before starting the agent.

**2. Start the agent:**

```bash
python main.py --session my_session
```

```
✓ MCP server connected (3 tools loaded)

Session: my_session

User:
```

---

### CLI flags

- `--session` — a unique session/thread ID used to persist conversation history in PostgreSQL.
- `--no-mcp` — skips connecting to the MCP server and runs with the base tools only. Remove this flag if you have the MCP server running at `http://localhost:8000/mcp`.

Type `exit` to quit the session.

---

## Model Selection

Two models are used, both served via the [Nebius](https://tokenfactory.nebius.com/) inference API:

| Role       | Model                         | Reason                                                                                                                                                                                                                |
| ---------- | ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Router** | `google/gemma-3-27b-it`       | Lightweight and fast. Optimized for high quality instruction following. Used only to classify each query as `structured`, `unstructured`, or `out_of_scope`. Low token limit (`max_tokens=20`) keeps latency minimal. |
| **Agent**  | `Qwen/Qwen3.5-397B-A17B-fast` | A large mixture-of-experts model with strong reasoning and tool-calling capabilities. Handles the actual customer service responses and tool invocations.                                                             |

---

## Project Structure

```
customer-service-agent-njama-ismael/
├── main.py                  # Entry point — argument parsing, agent setup, chat loop
├── README.md                # Project documentation
├── requirements.in          # Top-level dependencies
├── requirements.txt         # Pinned dependencies
├── .env                     # Environment variables (API keys, DB URL)
│
├── graph/
│   └── graph.py             # LangGraph state definition
│
├── router/
│   └── router.py            # Router chain — classifies queries before the agent
│
├── prompts/
│   ├── system_prompt.py     # System prompt for the agent
│   └── router_prompt.py     # Prompt for the router LLM
│
├── tools/
│   └── tools.py             # LangChain tools available to the agent
│
├── schemas/
│   └── schemas.py           # Pydantic schemas (e.g. Route)
│
├── mcp_server/
│   └── server.py            # Optional FastMCP server
│
├── utils/
│   └── get_mcp_tools.py     # Helper to load MCP tools with retry logic
│
└── data/
    └── dataset.py           # Dataset loading utilities
```
