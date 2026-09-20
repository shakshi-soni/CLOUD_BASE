

# CloudDash

An AI engineering project — a multi-agent customer support system for CloudDash, a fictional cloud infrastructure monitoring SaaS. Built for an AI Engineering Intern take-home.

I wanted to understand how multiple LLM agents can work together in a support workflow, specifically: routing a request to the right agent, grounding answers in a knowledge base instead of letting the model make things up, and handling cases where the system genuinely doesn't have enough information to answer.

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-FF4B4B?style=for-the-badge)](https://cloudbase-envp6xbcxfvjnybalsb33h.streamlit.app/)

## What it does

A triage agent reads the incoming message, classifies intent, and pulls out entities (plan type, issue category). It routes to one of three agents — Technical, Billing, or Escalation. Technical and Billing both pull from a knowledge base and cite what they used. If a conversation needs to move between agents mid-chat, there's a handover step that carries the history and entities along, so the user doesn't repeat themselves. If retrieval comes back empty, the system escalates instead of guessing.

## Architecture

```
User
  ↓
Input guardrails (injection check, off-topic filter)
  ↓
Triage Agent
  ↓
Technical / Billing / Escalation
  ↓
Retriever → Knowledge Base
  ↓
Output guardrails (PII redaction, hallucination check)
  ↓
Response
```

The triage agent decides where the request should go. Technical and billing agents retrieve from the knowledge base and cite the source. If the system can't find enough information, it escalates rather than inventing an answer.

## RAG

Chunking: 4-sentence windows, step of 2, so I don't lose context at a chunk boundary. Anything shorter than 4 sentences stays as one chunk. With the current knowledge base — 20 articles — that comes out to 57 chunks.

Retrieval is TF-IDF combined with BM25, weighted 60/40. I considered using embeddings but kept it lightweight for this version so I could actually understand what the retrieval pipeline was doing instead of relying on a managed vector search service. The 60/40 split isn't the result of tuning — it's just what I used to build this version.

It works reasonably well when the query uses the same terminology as the article (error codes, product names especially, where BM25 helps a lot). It's noticeably weaker when a user describes the same problem in different words — TF-IDF/BM25 don't do much for paraphrasing, since neither is actually semantic.

Before retrieval runs, I prepend the last 300 characters of conversation history to the query, otherwise something like "that issue" or "it" means nothing to the retriever on a follow-up message.

## Handover

An agent emits a signal like `[[ROUTE_TO_BILLING]]` in its output when it decides the conversation belongs elsewhere. The orchestrator catches that and calls `execute_handover()`, which checks the target agent exists, builds a payload (history, entity snapshot, summary), stores it for the next agent, and logs it. The next agent reads that payload on its turn. If a handover fails, it falls back to triage, or escalation if that was already the target.

I used string-matched routing signals instead of proper structured tool calls because it was faster to get working for this project. It's a bit fragile if the model phrases the flag differently than expected.

## Guardrails

Input: regex-based prompt injection detection (12 patterns), keyword allow/blocklist for off-topic messages.

Output: regex-based PII redaction (email, card numbers, phone, SSN), and a check that any specific numeric claim in a response actually matches the retrieved context.

## Project structure

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

## Running locally

```bash
git clone <repo-url>
cd clouddash
pip install -r requirements.txt

cp .env.example .env
# add GROQ_API_KEY=your_key_here

python cli/chat.py          # CLI
python api/main.py          # or as an API, docs at localhost:8000/docs
```

REST endpoints: `GET /health`, `POST /conversations`, `POST /conversations/{id}/messages`, `GET /conversations/{id}`.

## Tests

`python tests/test_system.py` runs the suite. Manual scenarios I used while building this:

- Single agent: an AWS-credentials/alerts question resolves entirely within Technical
- Handover: a message that starts as a technical question but pivots to a billing request gets routed to Technical first, then handed to Billing with context intact
- Escalation: a double-charge complaint goes Triage → Billing → Escalation
- KB miss: a question about an integration not in the knowledge base gets an honest "I don't have that" instead of a made-up answer

## Limitations

This isn't a production support platform, it's a take-home project. Specifically:

- Retrieval is TF-IDF + BM25, not semantic embeddings — paraphrased questions don't retrieve as well as ones using the article's own terminology
- The knowledge base is small (20 articles), so recall hasn't been tested at any real scale
- Triage's routing is LLM-classified and can misclassify ambiguous messages
- Sessions are in-memory only — nothing persists across a restart, and it won't scale past one process
- The 60/40 retrieval weighting hasn't been systematically tuned, just chosen
- No load testing, no latency/cost measurement
- Refunds and plan changes are simulated, not real transactions
- No response streaming, no API rate limiting

## What I would improve next

1. Compare TF-IDF/BM25 against embedding-based retrieval directly instead of assuming
2. Build a small eval set for routing and retrieval instead of testing by hand
3. Measure retrieval precision/recall properly
4. Try different chunk sizes and see what actually changes
5. Add latency and cost tracking
6. Test with adversarial and deliberately ambiguous queries
7. Replace `[[FLAG]]` routing with real structured tool calls
8. Redis-backed sessions for multi-instance deployment

## Why I built this

<!-- Write this part yourself, in your own words — what you wanted to learn, what was hard, what you changed while building it, what surprised you. This is the section that should sound the most like you, not me. -->

## About

Shakshi Soni.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/shakshi-soni-961048411/)
