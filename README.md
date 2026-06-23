# CloudDash Multi-Agent Support System

A production-grade prototype multi-agent customer support system for **CloudDash** — a fictional cloud infrastructure monitoring SaaS platform. Built for the AI Engineering Intern take-home assessment.

---

## Quick Start

```bash
# 1. Clone and enter the project
cd clouddash

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
cp .env.example .env
# Edit .env and add: GROQ_API_KEY=your_key_here

# 4a. Run the CLI chat interface
python cli/chat.py

# 4b. OR run the REST API
python api/main.py
# API docs at: http://localhost:8000/docs

# 5. Run the test suite
python tests/test_system.py
```

---

## Architecture Overview

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

---

## Project Structure

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

---

## RAG Pipeline Design

### Chunking Strategy
Articles are split into overlapping sentence windows (4 sentences, step 2) to preserve context across chunk boundaries. Short articles (≤4 sentences) are kept as single chunks. This produced 57 chunks from 20 articles.

### Hybrid Retrieval
- **Vector (60%)**: TF-IDF cosine similarity — lightweight, no external embedding service needed
- **BM25 (40%)**: Keyword lexical matching via `rank-bm25` — strong on exact term matches like error codes and product names
- Scores are fused with a weighted sum before ranking

### Query Rewriting
Before retrieval, the user's query is prepended with the last 300 characters of conversation context. This allows the retriever to resolve pronoun references like "that issue" or "it".

### Citation
Every agent response includes `cited_sources: [KB-XXX — Article Title]`. The Technical and Billing agents are instructed to append `[Source: KB-XXX]` inline in their responses.

---

## Agent Handover Protocol

1. Agent detects a domain boundary (via `[[ROUTE_TO_BILLING]]` or `[[ROUTE_TO_ESCALATION]]` signals in LLM output)
2. Orchestrator calls `execute_handover()`:
   - Validates target agent exists
   - Builds `HandoverPayload` (full history + entity snapshot + conversation summary)
   - Stores payload in `state.metadata["last_handover"]` for the receiving agent
   - Appends `HandoverLog` to `state.handover_logs`
   - Updates `state.current_agent`
3. Receiving agent reads `state.metadata["last_handover"]` so it has full prior context without the customer repeating themselves
4. On failure: falls back to `triage_agent` (or `escalation` if that was the intended target)

---

## Guardrails

| Type | Guardrail | Mechanism |
|------|-----------|-----------|
| Input | Prompt injection detection | Regex pattern matching (12 patterns) |
| Input | Off-topic filtering | Keyword allowlist/blocklist |
| Output | PII redaction | Regex: email, card numbers, phone, SSN |
| Output | Hallucination check | Specific numerical claims validated against KB context |
| Logic | No-KB fabrication | If retrieval returns 0 results, agent escalates — never invents |

---

## Configuration

All agent behaviour is defined in `config/agents.yaml`. Adding a new agent type:

1. Add a new key under `config/agents.yaml` with `model`, `description`, and `system_prompt`
2. Create the agent function in `agents/agent_impl.py`
3. Add its name to `VALID_AGENTS` in `handover/protocol.py`
4. Add a routing branch in `orchestrator.py`

The core orchestration loop does not need to change for new agents.

---

## Test Scenarios

### Scenario 1 — Technical (Single Agent)
```
"My CloudDash alerts stopped firing after I updated my AWS integration credentials yesterday."
→ Triage → Technical Support → KB-002 retrieved → step-by-step resolution with citation
```

### Scenario 2 — Cross-Agent Handover
```
"I want to upgrade from Pro to Enterprise, but first check my SSO issue."
→ Triage → Technical (SSO) → [[ROUTE_TO_BILLING]] → Billing has full prior context
```

### Scenario 3 — Escalation
```
"I've been charged twice for April. I need an immediate refund and want to speak to a manager."
→ Triage → Billing → [[ROUTE_TO_ESCALATION]] → Escalation packages manifest
```

### Scenario 4 — KB Miss
```
"Does CloudDash support Datadog integration?"
→ Technical retrieves 0 KB results → honestly acknowledges → offers escalation
```

---

## REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/conversations` | Start a new session |
| POST | `/conversations/{id}/messages` | Send a message |
| GET | `/conversations/{id}` | Get history + handover logs |

Interactive docs: `http://localhost:8000/docs`

---

## Design Decisions & Trade-offs

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| TF-IDF + BM25 instead of sentence-transformers | Zero dependency on large models, works offline, fast | Lower semantic similarity quality vs dense embeddings |
| Groq + Llama3-8b | Fast inference, free tier generous, low latency | Weaker reasoning than GPT-4 class models |
| In-memory session store | Simple, no DB setup needed for prototype | Sessions lost on restart; production would use Redis |
| Routing signals in LLM output (`[[FLAG]]`) | Easy to implement, no tool-calling API needed | Slightly fragile; production would use structured tool calls |
| YAML config for all prompts | Prompts editable without code changes | YAML indentation errors can break startup |
| Max 4 routing hops | Prevents infinite loops | Edge cases with 5+ agent transitions will hard-escalate |

---

## Known Limitations

- **No persistent storage**: sessions are in-memory and lost on restart
- **TF-IDF embeddings**: lower recall on semantic paraphrases vs dense vector models
- **Single-process**: no horizontal scaling; would need a session store (Redis) for multi-instance
- **Simulation only**: plan changes and refunds are simulated responses, not real transactions
- **No streaming**: responses are returned all at once, not streamed token-by-token
- **Rate limiting**: no per-user rate limiting on the API endpoints
