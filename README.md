# ReAct Agent (Assignment 2)

A CLI-based ReAct agent that answers factual questions by reasoning through a `Thought → Action → Observation` loop, backed by Tavily web search and an OpenAI LLM.

## Architecture

```
ReAct-agent/
├── agent.py          # ReActAgent class: prompt, LLM call, loop logic
├── tools.py          # search_web(): Tavily API wrapper
├── main.py           # CLI entry point (interactive + benchmark modes)
├── report.pdf        # Assignment report
├── .env.example      # Environment variable template
└── requirements.txt  # Python dependencies
```

### How the loop works

```
User Question
     │
     ▼
┌─────────────────────────────────────┐
│  LLM generates:                     │
│  Thought: <reasoning>               │
│  Action: Search["<query>"]   ──────►│  Tavily API
│                 or                  │◄──────────── Observation: <results>
│  Final Answer: <answer>             │
└─────────────────────────────────────┘
     │
     ▼ (repeat up to max_steps)
  Final Answer returned
```

**Key design decisions:**

- Stop sequence `"Observation:"` in the LLM call prevents the model from hallucinating observations.
- Conversation history is accumulated across steps so the model has full context at each turn.
- Benchmark mode uses a **single shared agent instance** across all 3 questions (as required by the assignment).
- Parser falls back gracefully when the model output does not match either expected format.

## Requirements

- Python 3.10+
- OpenAI API key
- Tavily API key

## Setup

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env         # then fill in your API keys
```

`.env` format:

```
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
MODEL_NAME=gpt-4o-mini
```

## Usage

**Interactive mode** — ask any question:

```bash
python main.py
python main.py --debug           # show step-by-step trace
python main.py --max_steps 8     # increase step limit (default: 5)
```

**Benchmark mode** — runs the 3 assignment questions with a shared agent:

```bash
python main.py --benchmark
python main.py --benchmark --debug
```

Benchmark questions:

1. What fraction of Japan's population is Taiwan's population as of 2025?
2. Compare the main display specs of iPhone 15 and Samsung S24.
3. Who is the CEO of the startup "Morphic" AI search?

## CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--benchmark` | off | Run the 3 benchmark questions |
| `--debug` | off | Print each Thought / Action / Observation |
| `--max_steps` | `5` | Max ReAct steps in interactive mode |
