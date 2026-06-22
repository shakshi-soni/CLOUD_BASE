# app.py

import os
import pathlib
import yaml
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any
import streamlit as st
import chromadb
from pydantic import BaseModel, Field
from groq import Groq

# ========== CONFIGURATION & SYSTEM INITIALIZATION ==========
# 1. Fetch parameters from Environment or Streamlit Production Secrets securely
GROQ_API_KEY = os.getenv('GROQ_API_KEY') or st.secrets.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY missing from environment variables or Streamlit secrets. Please configure it to continue.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

st.set_page_config(
    page_title="CloudDash Core Intelligence Console",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme UI Styling Engine Layout
st.markdown("""
    <style>
    .reportview-container { background: #0e1117; }
    div.stButton > button:first-child {
        background-color: #262730;
        color: #f0f2f6;
        border: 1px solid #464855;
        border-radius: 6px;
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        border-color: #ff4b4b;
        color: #ff4b4b;
    }
    .agent-tag {
        font-weight: bold;
        color: #00ffd0;
        background-color: #0c2b26;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid #145e52;
    }
    </style>
""", unsafe_allow_html=True)

# ========== DATA STRUCTURE SCHEMA MODELS ==========
class StructuredLogger:
    @staticmethod
    def log(event_type: str, trace_id: str, payload: dict):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "trace_id": trace_id,
            **payload
        }
        if "logs" not in st.session_state:
            st.session_state.logs = []
        st.session_state.logs.append(log_entry)
        print(f"[{event_type}] {yaml.dump(log_entry, default_flow_style=False)}")

class ConversationState(BaseModel):
    trace_id: str = Field(default_factory=lambda: f"trace_{int(datetime.now(timezone.utc).timestamp())}")
    current_agent: str = "triage_agent"
    history: List[Dict[str, str]] = []
    customer_id: Optional[str] = None
    issue_type: Optional[str] = None
    product_references: List[str] = []
    handover_logs: List[Dict[str, Any]] = []

# ========== CONFIG LOADING INTERCEPTOR ==========
# Tries to import modular prompts config asset layer; falls back gracefully to default matrix if local module structure not built yet.
try:
    CONFIG_PATH = pathlib.Path(__file__).parent / "config" / "prompts.yaml"
    with open(CONFIG_PATH, "r") as f:
        AGENTS_CONFIG = yaml.safe_load(f)
except Exception:
    AGENTS_CONFIG = {
        "triage_agent": {
            "model": "llama-3.1-8b-instant",
            "system_prompt": "Classify customer intent and route to: technical_support, billing, or escalation. Extract customer_id, issue_type, product_references. You are the router."
        },
        "technical_support": {
            "model": "llama-3.1-8b-instant",
            "system_prompt": "Resolve technical platform metrics, dashboards, or alerting updates using the provided KB context chunk. ALWAYS cite sources explicitly using the format [KB-XXX]. If payment or billing issues are brought up, output can include [[ROUTE_TO_BILLING]]"
        },
        "billing": {
            "model": "llama-3.1-8b-instant",
            "system_prompt": "Handle billing plans, disputes, or subscription tiers. If a direct manager override or direct cash refund authority is needed, output can include [[ROUTE_TO_ESCALATION]]"
        },
        "escalation": {
            "model": "llama-3.1-8b-instant",
            "system_prompt": "Package the full conversation context, extracted variables, and architectural state flags into a neat payload summarization ready for human support operator handoff."
        }
    }

# ========== CORE RETRIEVAL & INGESTION IMPORTS ==========
# Integrates cleanly with your modular Knowledge Base if files exist; executes locally otherwise.
try:
    from knowledge_base.ingestion import get_vector_db
    kb_collection = get_vector_db()
except ImportError:
    from chromadb.utils import embedding_functions
    @st.cache_resource
    def get_vector_db_fallback():
        chroma_client = chromadb.EphemeralClient()
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        coll = chroma_client.create_collection(name="clouddash_kb", embedding_function=embedding_fn, get_or_create=True)
        if coll.count() == 0:
            from knowledge_base.documents import KB_ARTICLES
            for doc in KB_ARTICLES:
                coll.add(
                    documents=[f"Title: {doc['title']}\nContent: {doc['content']}"],
                    metadatas=[{"id": doc["id"], "category": doc["category"]}],
                    ids=[doc["id"]]
                )
        return coll
    try:
        kb_collection = get_vector_db_fallback()
    except Exception:
        kb_collection = None

# ========== INLINE CHUNK Retrospective Functions ==========
def context_aware_query_rewrite(state: ConversationState, query: str) -> str:
    if len(state.history) < 2:
        return query
    history_snippet = "\n".join([f"{m['role']}: {m['content']}" for m in state.history[-3:]])
    try:
        rewritten = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Extract a standalone search query string combining user context history and the current raw question. Output ONLY the clear phrase."},
                {"role": "user", "content": f"History:\n{history_snippet}\n\nQuery: {query}"}
            ],
            temperature=0.1
        ).choices[0].message.content
        return rewritten.strip()
    except Exception:
        return query

def execute_rag_lookup(state: ConversationState, query: str, category: Optional[str] = None) -> tuple:
    if not kb_collection:
        return "", ""
    optimized = context_aware_query_rewrite(state, query)
    where_filter = {"category": category} if category else None
    results = kb_collection.query(query_texts=[optimized], n_results=1, where=where_filter)
    
    if results['documents'] and results['documents'][0]:
        return results['documents'][0][0], results['metadatas'][0][0]['id']
    return "", ""

def call_llm(state: ConversationState, agent: str, messages: list) -> str:
    StructuredLogger.log("AGENT_INVOCATION", state.trace_id, {"agent": agent})
    try:
        return client.chat.completions.create(
            model=AGENTS_CONFIG[agent]["model"],
            messages=messages,
            temperature=0.1
        ).choices[0].message.content
    except Exception as e:
        StructuredLogger.log("AGENT_ERROR", state.trace_id, {"agent": agent, "error": str(e)})
        return '{"next_agent": "escalation"}'

# ========== ORCHESTRATION PIPELINE PASSAGES ==========
try:
    from agents.orchestrator import process_customer_turn
except ImportError:
    # Embedded runtime orchestration loop fallback to keep app zero-configuration functional
    from agents.triage import run_triage_agent
    from agents.technical import run_technical_agent
    from agents.billing import run_billing_agent
    from agents.escalation import run_escalation_agent

    AGENT_REGISTRY = {
        "technical_support": run_technical_agent,
        "billing": run_billing_agent,
        "escalation": run_escalation_agent
    }

    def process_customer_turn(state: ConversationState, message: str):
        if any(x in message.lower() for x in ["ignore previous", "bypass", "system override"]):
            state.history.append({"role": "user", "content": message})
            state.history.append({"role": "assistant", "content": "🛑 **Security Boundary Violation Alert**: Request rejected."})
            return

        if state.current_agent == "triage_agent":
            status = run_triage_agent(state, message)
        else:
            state.history.append({"role": "user", "content": message})
            status = "HANDOVER"
        
        handovers = 0
        while status == "HANDOVER" and handovers < 5:
            agent = state.current_agent
            status = AGENT_REGISTRY.get(agent, run_escalation_agent)(state)
            handovers += 1

# ========== STREAMLIT MEMORY INITIALIZATION ==========
if "core_state" not in st.session_state:
    st.session_state.core_state = ConversationState()
if "logs" not in st.session_state:
    st.session_state.logs = []

current_state = st.session_state.core_state

# ========== STREAMLIT SCREEN LAYOUT RUNTIME UI ==========
st.title("⚡ CloudDash Customer Support AI Engine")
st.caption("Adaptive Production Workspace | Multi-Agent RAG Router & State Handover Console")

# Diagnostics Sidebar Layout Config
with st.sidebar:
    st.header("⚙️ Diagnostics Panel")
    st.markdown(f"**Session Trace ID:** `{current_state.trace_id}`")
    st.markdown(f"**Responsible Agent:** <span class='agent-tag'>{current_state.current_agent}</span>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"**Customer ID Key:** `{current_state.customer_id or 'None'}`")
    st.markdown(f"**Issue Classification:** `{current_state.issue_type or 'None'}`")
    st.markdown(f"**Product References:** `{current_state.product_references}`")
    st.markdown("---")
    if st.button("🔄 Clear State Workspace Memory", key="clear_state_btn"):
        st.session_state.core_state = ConversationState()
        st.session_state.logs = []
        st.rerun()

# 4 Interactive Evaluation Scenario Buttons
st.markdown("### 🎯 Quick Evaluation Scenarios")
col1, col2, col3, col4 = st.columns(4)

faq_query = None
with col1:
    if st.button("📋 1. AWS Alert Failures", key="faq_btn_1"):
        faq_query = "My customer ID is CUST-901. My CloudDash alerts stopped firing after I updated my AWS integration credentials yesterday. Help!"
with col2:
    if st.button("🔄 2. Cross Domain Handover", key="faq_btn_2"):
        faq_query = "I am currently using your Pro tier, but I need to upgrade to Enterprise to check out the new SSO & SAML support features."
with col3:
    if st.button("🚨 3. Refund Escalation Chain", key="faq_btn_3"):
        faq_query = "You charged my card twice for the monthly sub. I need an immediate refund and demand to talk to a manager."
with col4:
    if st.button("🔍 4. Rate Limit Verification", key="faq_btn_4"):
        faq_query = "What are the standard Rate limits for making queries against the Developer API keys tokens?"

if faq_query:
    process_customer_turn(current_state, faq_query)
    st.rerun()

st.markdown("---")

# Main Screen Chat vs Telemetry Grid Split Layout
chat_layout, telemetry_layout = st.columns([3, 2])

with chat_layout:
    st.subheader("💬 Active Chat Interaction")
    for msg in current_state.history:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        else:
            st.chat_message("assistant").markdown(msg["content"])
            
    if user_prompt := st.chat_input("Enter account inquiries here..."):
        st.chat_message("user").markdown(user_prompt)
        process_customer_turn(current_state, user_prompt)
        st.rerun()

with telemetry_layout:
    st.subheader("📊 Execution Traces")
    with st.expander("🔗 Handover Logs Matrix", expanded=True):
        if current_state.handover_logs:
            for item in current_state.handover_logs:
                st.info(f"🔄 **{item.get('source')}** ➔ **{item.get('target')}**\n\n*Reason:* {item.get('reason')}")
        else:
            st.text("No handovers registered yet.")

    with st.expander("🛠️ Live JSON Telemetry Pipeline", expanded=True):
        if st.session_state.logs:
            st.json(st.session_state.logs[-3:])
        else:
            st.text("Waiting for logs...")
