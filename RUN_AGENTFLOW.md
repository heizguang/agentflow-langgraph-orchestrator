# AgentFlow Local Run Guide

## Project Location

Use your local clone path, for example:
`<your-local-path>/AgentFlow`

## Configuration Notes

This project uses OpenAI-compatible environment variables and supports local document knowledge storage.

Key changes:

- LLM access now uses `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `OPENAI_MODEL`
- PDF knowledge storage no longer depends on Pinecone
- Uploaded PDF chunks are stored in a local knowledge base under `data/local_kb/`
- The project no longer requires `GROQ_API_KEY`, `GOOGLE_API_KEY`, or `PINECONE_API_KEY`

## Environment Setup

File:

`<your-local-path>/AgentFlow/.env`

- `OPENAI_BASE_URL`
- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `TAVILY_API_KEY`

Optional variables you may still fill later:

- `SERPER_API_KEY`
- `WEATHER_API_KEY`
- `STOCK_FINANCE_API_KEY`
- `LANGSMITH_API_KEY`

Optional per-agent model overrides:

- `AGENTFLOW_MODEL`
- `AGENTFLOW_ROUTER_MODEL`
- `AGENTFLOW_RAG_MODEL`
- `AGENTFLOW_WEB_MODEL`
- `AGENTFLOW_ANSWER_MODEL`
- `AGENTFLOW_RESEARCH_MODEL`

If these are empty, the app falls back to `OPENAI_MODEL`.

## Required Minimum To Run

At minimum, these must be present in `.env`:

- `OPENAI_BASE_URL`
- `OPENAI_API_KEY`
- `OPENAI_MODEL`

Recommended:

- `TAVILY_API_KEY`

Without `TAVILY_API_KEY`, web search related capabilities may be limited or fail.

## Install Status

Recommended baseline setup:

- create a virtual environment
- install dependencies from `requirements.txt`
- ensure `data/` exists for local knowledge storage

## Start Command

Run in PowerShell:

```powershell
cd <your-local-path>\AgentFlow
.\.venv\Scripts\python -m streamlit run app.py --server.headless true --server.port 8501
```

Open:

`http://localhost:8501`

## How This Version Works

This project is still a multi-agent workflow, not an `NL2SQL` app.

Main flow:

- `Router Agent`: decides which path to use
- `RAG Agent`: checks uploaded local document knowledge
- `Web Agent`: uses search and utility tools
- `Research Agent`: performs deeper research-style answering
- `Answer Agent`: synthesizes the final response

## PDF / RAG Behavior

When you upload a PDF:

1. the PDF is parsed
2. a summary and topic are generated with the configured LLM
3. the document is chunked
4. chunks are saved locally into `data/local_kb/<topic>.json`
5. later RAG retrieval uses local keyword-based recall over those chunks

This means:

- no Pinecone account is needed
- no separate embedding provider is required
- local setup is much simpler

## Notes

- Python `3.11+` is recommended
- Search/news/weather/stock tools still depend on their own external APIs if you want those features
- Free-tier API endpoints may rate-limit or timeout

## Suggested Next Step

1. start the app
2. test a normal question first
3. upload one PDF
4. ask questions related to that PDF
5. add optional tool keys only if you need those features
