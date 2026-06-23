import os
import json
import yaml
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from pydantic import BaseModel, Field
from groq import Groq
from dotenv import load_dotenv

# ========== CONFIGURATION & SYSTEM INITIALIZATION ==========
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY missing from environment variables. Please check your .env file.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

st.set_page_config(
    page_title="CloudDash Core Intelligence Console",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme Stylings
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

# ========== LOGGING & SCHEMAS ==========
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
        print(f"[{event_type}] {json.dumps(log_entry)}")

class ConversationState(BaseModel):
    trace_id: str = Field(default_factory=lambda: f"trace_{int(datetime.now(timezone.utc).timestamp())}")
    current_agent: str = "triage_agent"
    history: List[Dict[str, str]] = []
    customer_id: Optional[str] = None
    issue_type: Optional[str] = None
    product_references: List[str] = []
    handover_logs: List[Dict[str, Any]] = []

# ========== CONFIGURATIONS ==========
AGENTS_CONFIG = yaml.safe_load("""
triage_agent:
  model: "llama-3.1-8b-instant"
  system_prompt: "Classify customer intent and route to: technical_support, billing, or escalation. Extract customer_id, issue_type, product_references. You are the router."
technical_support:
  model: "llama-3.1-8b-instant"
  system_prompt: "Resolve technical platform metrics, dashboards, or alerting updates using the provided KB context chunk. ALWAYS cite sources explicitly using the format [KB-XXX]. If payment or billing issues are brought up, output can include [[ROUTE_TO_BILLING]]"
billing:
  model: "llama-3.1-8b-instant"
  system_prompt: "Handle billing plans, disputes, or subscription tiers. If a direct manager override or direct cash refund authority is needed, output can include [[ROUTE_TO_ESCALATION]]"
escalation:
  model: "llama-3.1-8b-instant"
  system_prompt: "Package the full conversation context, extracted variables, and architectural state flags into a neat payload summarization ready for human support operator handoff."
""")

# FULL 15-ARTICLE REQUIRED SPECIFICATION
KB_ARTICLES = [
    {"id": "KB-001", "title": "Configure Alert Thresholds", "category": "troubleshooting", "content": "Navigate Dashboards → Alert Configurations. Verify AWS integration status and re-link notification endpoints."},
    {"id": "KB-002", "title": "Dashboard Performance Issues", "category": "troubleshooting", "content": "Slow rendering? Adjust query spans from 30 days to 1 hour in Settings."},
    {"id": "KB-003", "title": "AWS Integration Key Failures", "category": "troubleshooting", "content": "Verify IAM role has CloudWatchReadOnlyAccess. Update keys in Settings → Integrations."},
    {"id": "KB-004", "title": "GCP Metrics Out of Sync", "category": "troubleshooting", "content": "Check service account JSON expiration. Re-upload fresh credentials in Settings → Integrations → GCP."},
    {"id": "KB-005", "title": "Rotating API Tokens", "category": "faqs", "content": "Generate new tokens in Settings → Developer API Keys. Revoke legacy items for compliance."},
    {"id": "KB-006", "title": "Supported Platforms", "category": "faqs", "content": "CloudDash supports AWS, GCP, and Azure natively."},
    {"id": "KB-007", "title": "Inviting Team Members", "category": "faqs", "content": "Settings → Team Management → Members. Assign Admin, Editor, or Viewer roles."},
    {"id": "KB-008", "title": "Subscription Tiers", "category": "billing", "content": "Free, Pro ($49/mo), Enterprise. Enterprise includes SSO and 12-month log retention."},
    {"id": "KB-009", "title": "Refund Policy", "category": "billing", "content": "Subscriptions non-refundable. Confirmed double-billing can be overridden by Billing Manager."},
    {"id": "KB-010", "title": "Payment Methods", "category": "billing", "content": "Accept Visa, Mastercard, Amex. Failed payments trigger 14-day grace period."},
    {"id": "KB-011", "title": "API Authentication", "category": "api_documentation", "content": "All requests require bearer token: Authorization: Bearer <TOKEN>"},
    {"id": "KB-012", "title": "Rate Limits", "category": "api_documentation", "content": "1,000 queries/minute per token. Returns HTTP 429 if exceeded."},
    {"id": "KB-013", "title": "Webhooks", "category": "api_documentation", "content": "Deploy alerts to custom endpoints via Settings → Webhooks. Uses HMAC-SHA256 signatures."},
    {"id": "KB-014", "title": "SSO & SAML", "category": "account", "content": "SAML 2.0 (Okta, Azure AD) restricted to Enterprise tier. Configure in Settings → SSO."},
    {"id": "KB-015", "title": "RBAC", "category": "account", "content": "Admins: read-write billing. Editors: connection control. Viewers: read-only."}
]

# ========== VECTOR STORAGE ENGINE (SAFE APHEMERAL ENGINE) ==========
@st.cache_resource
def get_vector_db():
    chroma_client = chromadb.EphemeralClient() # Prevents in-memory drop errors across hot-reloads
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    kb_collection = chroma_client.create_collection(name="clouddash_kb", embedding_function=embedding_fn, get_or_create=True)
    
    if kb_collection.count() == 0:
        for doc in KB_ARTICLES:
            kb_collection.add(
                documents=[f"Title: {doc['title']}\nContent: {doc['content']}"],
                metadatas=[{"id": doc["id"], "category": doc["category"]}],
                ids=[doc["id"]]
            )
    return kb_collection

kb_collection = get_vector_db()

# ========== AGENT CORES & HANDOVER EXECUTION ENGINE ==========
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
    except:
        return query

def execute_rag_lookup(state: ConversationState, query: str, category: Optional[str] = None) -> tuple:
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

def run_triage_agent(state: ConversationState, user_input: str) -> str:
    state.history.append({"role": "user", "content": user_input})
    messages = [
        {"role": "system", "content": AGENTS_CONFIG["triage_agent"]["system_prompt"]},
        {"role": "user", "content": f"Input: {user_input}\nRespond ONLY as JSON payload matching exactly: {{\"next_agent\": \"technical_support\"|\"billing\"|\"escalation\", \"customer_id\": \"string or null\", \"issue_type\": \"string\", \"product_references\": [\"strings\"], \"reason\": \"string\"}}"}
    ]
    output = call_llm(state, "triage_agent", messages)
    try:
        data = json.loads(output)
        state.customer_id = data.get("customer_id") or state.customer_id
        state.issue_type = data.get("issue_type") or state.issue_type
        if data.get("product_references"):
            state.product_references = list(set(state.product_references + data.get("product_references", [])))
        
        target = data.get("next_agent", "escalation")
        if target != "triage_agent":
            state.handover_logs.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "triage_agent",
                "target": target,
                "reason": data.get("reason"),
                "context_snapshot": {"customer_id": state.customer_id, "issue_type": state.issue_type}
            })
            StructuredLogger.log("AGENT_HANDOVER", state.trace_id, {"source": "triage_agent", "target": target})
            state.current_agent = target
        return "HANDOVER"
    except:
        state.current_agent = "escalation"
        return "HANDOVER"

def run_technical_agent(state: ConversationState) -> str:
    last_query = [m["content"] for m in state.history if m["role"] == "user"][-1]
    kb_doc, kb_id = execute_rag_lookup(state, last_query, category="troubleshooting")
    
    if not kb_doc:
        kb_doc, kb_id = execute_rag_lookup(state, last_query, category="faqs")

    if not kb_doc:
        state.handover_logs.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "technical_support",
            "target": "escalation",
            "reason": "Knowledge gap fallback."
        })
        state.current_agent = "escalation"
        return "HANDOVER"
    
    system = f"{AGENTS_CONFIG['technical_support']['system_prompt']}\nContext Grounding:\n[{kb_id}]: {kb_doc}"
    messages = [{"role": "system", "content": system}] + [{"role": m["role"], "content": m["content"]} for m in state.history[-4:]]
    messages.append({"role": "user", "content": "Note: If pricing or plan alterations are explicitly stated, append [[ROUTE_TO_BILLING]]"})
    
    res = call_llm(state, "technical_support", messages)
    if "[[ROUTE_TO_BILLING]]" in res:
        state.handover_logs.append({"timestamp": datetime.now(timezone.utc).isoformat(), "source": "technical_support", "target": "billing", "reason": "Cross domain threshold hit"})
        state.current_agent = "billing"
        return "HANDOVER"
    
    state.history.append({"role": "assistant", "content": res})
    return res

def run_billing_agent(state: ConversationState) -> str:
    last_query = [m["content"] for m in state.history if m["role"] == "user"][-1]
    kb_doc, kb_id = execute_rag_lookup(state, last_query, category="billing")
    
    system = AGENTS_CONFIG['billing']['system_prompt']
    if kb_doc:
        system += f"\nContext Grounding:\n[{kb_id}]: {kb_doc}"
    
    messages = [{"role": "system", "content": system}] + [{"role": m["role"], "content": m["content"]} for m in state.history[-4:]]
    messages.append({"role": "user", "content": "Note: If manager escalations or immediate credits are forced, append [[ROUTE_TO_ESCALATION]]"})
    
    res = call_llm(state, "billing", messages)
    if "[[ROUTE_TO_ESCALATION]]" in res:
        state.current_agent = "escalation"
        return "HANDOVER"
    
    state.history.append({"role": "assistant", "content": res})
    return res

def run_escalation_agent(state: ConversationState) -> str:
    history_string = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in state.history])
    messages = [
        {"role": "system", "content": AGENTS_CONFIG["escalation"]["system_prompt"]},
        {"role": "user", "content": f"Context: Customer {state.customer_id}, Focus issue: {state.issue_type}\nTrace:\n{history_string}"}
    ]
    res = call_llm(state, "escalation", messages)
    final_output = f"⚠️ **[SYSTEM ESCALATION - HUMAN HANDOVER INTERACTION TIER]**\n\n{res}"
    state.history.append({"role": "assistant", "content": final_output})
    return final_output

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

# ========== STREAMLIT MEMORY TIE-IN ==========
if "core_state" not in st.session_state:
    st.session_state.core_state = ConversationState()
if "logs" not in st.session_state:
    st.session_state.logs = []

current_state = st.session_state.core_state

# ========== STREAMLIT SCREEN LAYOUT ==========
st.title("⚡ CloudDash Customer Support AI Engine")
st.caption("Adaptive Production Workspace | Multi-Agent RAG Router & State Handover Console")

# Sidebar
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

# 4 Unique FAQ Interactive Buttons
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

# Layout Split
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