# CloudDash — Multi-Agent Customer Support System

Multi-agent support system for CloudDash (fictional cloud monitoring SaaS). Built for the AI Engineering Intern take-home.

Instead of one LLM handling everything, there's a triage agent that reads the message first and routes it to Technical, Billing, or Escalation. Each of those has its own KB access and its own prompt. If a conversation needs to move between agents mid-chat, there's an actual handover step that carries the history and extracted entities over, so the user isn't asked the same thing twice.

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-FF4B4B?style=for-the-badge)](https://cloudbase-envp6xbcxfvjnybalsb33h.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

## Setup

```bash
git clone <repo-url>
cd clouddash
pip install -r requirements.txt

cp .env.example .env
# add GROQ_API_KEY=your_key_here

python cli/chat.py          # CLI
python api/main.py          # or run as API, docs at localhost:8000/docs
python tests/test_system.py # tests
```

## Flow

```
User Input
    │
    ▼
┌─────────────────────────────────────────────┐
│              INPUT GUARDRAILS               │
│  • Prompt injection detection               │
│  • Off-topic filtering                      │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│            ORCHESTRATOR                     │
│  • Loads YAML config (agents, prompts)      │
│  • Manages conversation state               │
│  • Controls routing loop (max 4 hops)       │
└─────────────────────────────────────────────┘
    │
    ▼
┌──────────────┐
│ TRIAGE AGENT │ ← classifies intent via LLM JSON output
│              │   extracts entities (plan, issue_type)
└──────┬───────┘
       │ routes to one of:
  ┌────┴──────────────────┬──────────────────┐
  ▼                       ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  TECHNICAL   │  │   BILLING    │  │  ESCALATION  │
│  SUPPORT     │  │   AGENT      │  │  AGENT       │
│              │  │              │  │              │
│ RAG pipeline │  │ RAG pipeline │  │ Packages     │
│ KB retrieval │  │ KB retrieval │  │ context for  │
│ Citations    │  │ Plan changes │  │ human handoff│
└──────┬───────┘  └──────┬───────┘  └──────────────┘
       │                 │
       │ [[ROUTE_TO_*]]  │ [[ROUTE_TO_ESCALATION]]
       └────────┬────────┘
                ▼
        HANDOVER PROTOCOL
        • Preserves full history
        • Transfers entities
        • Audit log entry
        • Failure fallback
                │
                ▼
┌─────────────────────────────────────────────┐
│             OUTPUT GUARDRAILS               │
│  • PII redaction (email, card, phone)       │
│  • Hallucination check vs KB context        │
└─────────────────────────────────────────────┘
    │
    ▼
Final Response to Customer
```

## Structure

```
clouddash/
├── agents/
│   ├── billing.py
│   ├── escalation.py
│   ├── orchestrator.py
│   ├── technical.py
│   └── triage.py
├── config/
│   ├── prompts.yaml
│   └── settings.py
├── handover/
│   ├── protocol.py
│   └── protocol.readme
├── knowledge_base/
│   └── documents
├── retrieval/
│   ├── __init__.py
│   ├── query_rewrite.py
│   └── search_engine.py
├── tests/
│   ├── test_agents.py
│   └── test_orchestration.py
├── app.py
├── requirements.txt
├── .env
└── README.md
```

## RAG

Chunking: 4-sentence windows, step of 2, so context doesn't get cut off at a boundary. Anything under 4 sentences stays as one chunk. Ends up as 57 chunks from 20 articles.

Retrieval is TF-IDF (60%) + BM25 (40%, via `rank-bm25`), scores combined with a weighted sum. BM25 pulls its weight on stuff like exact error codes and product names that TF-IDF alone tends to miss.

Query gets the last 300 chars of conversation history prepended before retrieval runs — otherwise "that issue" or "it" means nothing to the retriever on a follow-up message.

Responses carry `cited_sources: [KB-XXX — Article Title]`, and Technical/Billing also put `[Source: KB-XXX]` directly in the reply text.

## Handover

Agent emits `[[ROUTE_TO_BILLING]]` or similar in its output when it decides the conversation belongs elsewhere. Orchestrator picks that up, calls `execute_handover()`:

- checks the target agent exists
- builds a `HandoverPayload` (history + entity snapshot + summary)
- stores it in `state.metadata["last_handover"]`
- logs to `state.handover_logs`
- updates `state.current_agent`

Next agent reads `last_handover` on its turn so it already has context. If a handover fails it falls back to triage, or escalation if that was already the target.

## Guardrails

Input: prompt injection detection (regex, 12 patterns), off-topic filtering (keyword allow/blocklist).

Output: PII redaction (email, card, phone, SSN via regex), hallucination check on numeric claims against retrieved context.

Global rule: zero KB results = escalate. Never guess.

## Adding a new agent

1. Add entry to `config/agents.yaml` — `model`, `description`, `system_prompt`
2. Write the function in `agents/agent_impl.py`
3. Add name to `VALID_AGENTS` in `handover/protocol.py`
4. Add routing branch in `orchestrator.py`

Orchestrator loop itself doesn't need touching.

## Test cases

Single agent:
```
"My CloudDash alerts stopped firing after I updated my AWS integration credentials yesterday."
→ Triage → Technical → KB-002 → fix with citation
```

Handover mid-conversation:
```
"I want to upgrade from Pro to Enterprise, but first check my SSO issue."
→ Triage → Technical (SSO) → [[ROUTE_TO_BILLING]] → Billing already has context
```

Escalation:
```
"I've been charged twice for April. I need an immediate refund and want to speak to a manager."
→ Triage → Billing → [[ROUTE_TO_ESCALATION]]
```

KB miss:
```
"Does CloudDash support Datadog integration?"
→ Technical, 0 results → says so → offers escalation
```

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | health check |
| POST | `/conversations` | new session |
| POST | `/conversations/{id}/messages` | send message |
| GET | `/conversations/{id}` | history + handover logs |

Docs: `localhost:8000/docs`

## Notes on decisions

TF-IDF + BM25 over sentence-transformers because no heavy embedding model to load, works offline. Misses semantic paraphrases sometimes, TF-IDF just isn't that smart.

Groq + Llama3-8B for speed and the free tier. Reasoning is weaker than GPT-4 class, noticeable on trickier billing edge cases.

Sessions are in-memory. Fine for a demo, gone on restart. Would need Redis for anything real.

`[[FLAG]]` tokens for routing instead of proper tool calls — faster to build, but it's a string match so it's a bit fragile if the model phrases things weird.

Prompts in YAML so I can tweak wording without redeploying, but a bad indent breaks the whole thing on startup (happened to me twice).

Capped routing at 4 hops to stop loops. A genuinely weird 5-hop case just force-escalates.

## Not done yet

- no persistence, in-memory only
- TF-IDF misses paraphrased questions sometimes
- single process, no scaling
- refunds/plan changes are simulated, not real
- no streaming
- no rate limiting

## If I keep working on this

- dense embeddings behind a flag so I can compare against TF-IDF directly
- Redis for sessions
- real tool calling instead of `[[FLAG]]`
- streaming
- rate limiting

## About

Shakshi Soni. Into agentic AI — systems where the model does things, not just talks.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/shakshi-soni-961048411/)
