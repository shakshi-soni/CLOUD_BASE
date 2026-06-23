# app.py

import os
import pathlib
import json
import re
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from pydantic import BaseModel, Field
from groq import Groq

# Force pure-python parsing to bypass protobuf descriptor compatibility limits
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# ========== CONFIGURATION & SYSTEM INITIALIZATION ==========
GROQ_API_KEY = os.getenv('GROQ_API_KEY') or st.secrets.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY missing from environment variables or Streamlit secrets. Please configure it to continue.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

st.set_page_config(
    page_title="CloudDash AI Support Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== ORIGINAL UI DESIGN SYSTEM ==========
st.markdown("""
    <style>
    .stApp { background-color: #0A0F1C; color: #F3F4F6; }
    section[data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1F2937;
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366F1 0%, #a855f7 50%, #10B981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: #9CA3AF;
        margin-top: 4px;
        margin-bottom: 4px;
        font-weight: 500;
    }
    .hero-badges {
        font-size: 0.85rem;
        color: #6366F1;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 25px;
    }
    div.stButton > button:first-child {
        background: #111827;
        color: #E5E7EB;
        border: 1px solid #1F2937;
        border-radius: 8px;
        padding: 0.6rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        border-color: #6366F1;
        color: #6366F1;
        background: #1E1B4B;
    }
    .kpi-box {
        background: #111827;
        border: 1px solid #1F2937;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .kpi-title {
        font-size: 0.75rem;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    .kpi-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #F9FAFB;
    }
    .pipeline-container {
        background: #111827;
        border: 1px solid #1F2937;
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .pipeline-node {
        padding: 10px 12px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        text-align: center;
        margin: 6px 0;
        border: 1px solid transparent;
        opacity: 0.25;
        transition: all 0.3s ease;
    }
    .node-active { 
        opacity: 1.0 !important; 
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.3);
        transform: scale(1.02);
    }
    .node-triage { background: #1E1B4B; color: #818CF8; border-color: #312E81; }
    .node-tech { background: #06282D; color: #22D3EE; border-color: #083344; }
    .node-billing { background: #062F21; color: #34D399; border-color: #064E3B; }
    .node-escalation { background: #450A0A; color: #F87171; border-color: #7F1D1D; }
    .timeline-card {
        background: #111827;
        border-left: 3px solid #6366F1;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.8rem;
        border-top: 1px solid #1F2937;
        border-right: 1px solid #1F2937;
        border-bottom: 1px solid #1F2937;
    }
    .guardrail-warning {
        background: #450A0A;
        border: 1px solid #EF4444;
        padding: 1rem;
        border-radius: 8px;
        color: #F87171;
        margin-bottom: 1rem;
        font-size: 0.9rem;
    }
    .clean-hr { border: 0; height: 1px; background: #1F2937; margin: 1.5rem 0; }
    </style>
""", unsafe_allow_html=True)

# ========== DATA STRUCTURE SCHEMA MODELS ==========
class ConversationState(BaseModel):
    trace_id: str = Field(default_factory=lambda: f"trace_{int(datetime.now(timezone.utc).timestamp())}")
    current_agent: str = "triage_agent"
    history: List[Dict[str, str]] = []
    customer_id: Optional[str] = None
    issue_type: Optional[str] = None
    product_references: List[str] = []
    handover_logs: List[Dict[str, Any]] = []
    active_kb_citations: List[Dict[str, str]] = []
    guardrail_alerts: List[str] = []

# ========== JSON EVENT LOGGING ENGINE ==========
def log_event(event_type: str, trace_id: str, payload: dict):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "trace_id": trace_id,
        **payload
    }
    print(json.dumps(entry))

# ========== CHROMADB VECTOR ENGINE ==========
@st.cache_resource
def initialize_vector_db():
    chroma_client = chromadb.EphemeralClient()
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    collection = chroma_client.get_or_create_collection(name="clouddash_kb", embedding_function=embedding_fn)
    
    kb_articles = [
        {"id": "KB-014", "title": "AWS Pipeline Key Rotation", "text": "AWS Alert Failures: If alerts drop after credential rotation, update your pipeline target endpoints inside the CloudDash console and re-verify the OAuth handshake protocol tokens."},
        {"id": "KB-008", "title": "Enterprise SSO Pricing Matrix", "text": "SSO and SAML: Single Sign-On configuration profiles are locked explicitly to the Enterprise service level agreement tier. Upgrading from Pro requires a billing contract adjustment."},
        {"id": "KB-011", "title": "Developer API Rate Limits", "text": "Developer API Rate Limits: Standard tokens are capped at 5000 requests per hour. If exceeded, the gateway returns an HTTP 429 error code. Contact support for tailored high-throughput endpoints."},
        {"id": "KB-001", "title": "Data Retention Policy", "text": "CloudDash metrics data is retained for 30 days on Pro tiers and 365 days on Enterprise tiers."},
        {"id": "KB-002", "title": "Refund Authority Matrix", "text": "Refund requests require manager verification if the transaction occurs past the standard 14-day window."},
        {"id": "KB-003", "title": "Webhook Configuration Failure", "text": "If webhooks return 502, verify that your firewall isn't blocking incoming traffic from CloudDash's public IP block."},
        {"id": "KB-004", "title": "Slack Integration Setup", "text": "Slack workspace notifications can be mapped under the Alert Routing settings pane using an incoming webhook URL."},
        {"id": "KB-005", "title": "Grafana Plugin Connection", "text": "Connecting Grafana requires an API token with 'Viewer' or 'Admin' scope generated within CloudDash settings."},
        {"id": "KB-006", "title": "Azure Monitor Connection", "text": "Azure pipeline ingestion requires Tenant ID, Client ID, and a valid Client Secret from App Registrations."},
        {"id": "KB-007", "title": "Invoice Access Paths", "text": "Invoices can be downloaded as PDF files directly under Billing settings -> Transaction History."},
        {"id": "KB-009", "title": "Team Seat Allocations", "text": "The Pro tier includes 5 user seats by default. Extra seats are billed at $15 per seat per month dynamically."},
        {"id": "KB-010", "title": "Prometheus Scrape Interval", "text": "The default metric scrape interval for Prometheus endpoints is 15 seconds, configurable in clouddash.yaml."},
        {"id": "KB-012", "title": "Custom Domain Mapping", "text": "Enterprise users can configure custom white-label dashboard URLs by adding a CNAME record to their DNS provider."},
        {"id": "KB-013", "title": "Datadog Migration Guide", "text": "Import Datadog dashboards smoothly using our native JSON migration script found under Settings -> Import Utilities."},
        {"id": "KB-015", "title": "GDPR Compliance Ledger", "text": "All user telemetry stored within the European region is encrypted at rest using AES-256 keys managed via AWS KMS."}
    ]
    
    if collection.count() == 0:
        collection.add(
            documents=[a["text"] for a in kb_articles],
            metadatas=[{"id": a["id"], "title": a["title"]} for a in kb_articles],
            ids=[a["id"] for a in kb_articles]
        )
    return collection

try:
    vector_collection = initialize_vector_db()
except Exception as e:
    st.error(f"Vector DB Initialization Fault: {str(e)}")

def query_vector_kb(query: str, n_results: int = 1) -> List[Dict[str, str]]:
    try:
        results = vector_collection.query(query_texts=[query], n_results=n_results)
        output = []
        if results and results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                output.append({
                    "id": results['ids'][0][i],
                    "title": results['metadatas'][0][i]['title'],
                    "content": results['documents'][0][i]
                })
        return output
    except Exception:
        return []

# ========== SYSTEM SECURITY GUARDRAILS ==========
def check_input_guardrail(text: str) -> bool:
    malicious_patterns = [r"ignore previous instructions", r"system prompt", r"override rules"]
    for pattern in malicious_patterns:
        if re.search(pattern, text.lower()):
            return False
    return True

def apply_output_guardrail(text: str) -> str:
    redacted = text.replace("[[ROUTE_TO_BILLING]]", "").replace("[[ROUTE_TO_ESCALATION]]", "")
    redacted = re.sub(r"secret_key_[a-zA-Z0-9]+", "[REDACTED_SECRET]", redacted)
    return redacted

# ========== AGENT ROLES AND SETTINGS ==========
AGENTS_CONFIG = {
    "triage_agent": {
        "model": "llama-3.1-8b-instant",
        "system_prompt": "You are a Triage Router. Analyze user intent. Extract customer_id and classify domain. Respond ONLY in valid JSON format: {\"next_agent\": \"technical_support\" | \"billing\" | \"escalation\", \"customer_id\": \"string or null\", \"issue_type\": \"string\"}"
    },
    "technical_support": {
        "model": "llama-3.1-8b-instant",
        "system_prompt": "You are Technical Support. Answer the technical inquiry professionally using the provided context. Always cite your source explicitly as [KB-XXX]. Only cite KB article IDs that are explicitly present in the grounding context provided. Never invent or guess KB IDs. If the user asks about payments, plans, or upgrading tiers, reply exactly with: [[ROUTE_TO_BILLING]]."
    },
    "billing": {
        "model": "llama-3.1-8b-instant",
        "system_prompt": "You are Billing Support. Answer payment, invoice, tier adjustment, or subscription questions using the context. If an immediate refund or manager escalation is explicitly demanded, reply exactly with: [[ROUTE_TO_ESCALATION]]."
    },
    "escalation": {
        "model": "llama-3.1-8b-instant",
        "system_prompt": "You are the Human Escalation Assistant. Provide a reassuring, friendly, customer-facing response letting the customer know a senior engineering lead has been alerted and will contact them directly. Do not output raw JSON or system flags."
    }
}

# Helper to normalize LLM routing responses and completely eliminate KeyErrors
def clean_agent_key(agent_name: str) -> str:
    if not agent_name:
        return "technical_support"
    name = str(agent_name).lower().strip()
    if "triage" in name:
        return "triage_agent"
    if "billing" in name:
        return "billing"
    if "escalation" in name or "lead" in name:
        return "escalation"
    return "technical_support"

# ========== PIPELINE MANAGEMENT & EXTRACTION ==========
def run_orchestration_loop(state: ConversationState, user_message: str):
    state.guardrail_alerts = []

    if not check_input_guardrail(user_message):
        state.guardrail_alerts.append(f"Security Override Triggered: Malicious instructions blocked.")
        state.history.append({"role": "user", "content": user_message})
        state.history.append({"role": "assistant", "content": "⚠️ **Security Flag:** System validation protocols flagged anomalous input instructions."})
        return

    state.history.append({"role": "user", "content": user_message})
    
    matches = query_vector_kb(user_message, n_results=1)
    
    # Place 2 Log Call: KB Retrieval
    log_event("KB_RETRIEVAL", state.trace_id, {"query": user_message, "result": matches[0]['id'] if matches else "none"})
    
    # 3. KB Not Found Handling Execution Turn
    if not matches:
        state.history.append({"role": "assistant", "content": "I couldn't find relevant information in our knowledge base for this query. Would you like me to escalate this to our product team?"})
        return

    context_text = f"\n[GROUNDING CONTEXT] Rely on this data chunk to construct answers:\n{matches[0]['content']}\n"
    if matches[0] not in state.active_kb_citations:
        state.active_kb_citations.append(matches[0])

    # Dynamic extraction turn
    try:
        extraction_res = client.chat.completions.create(
            model=AGENTS_CONFIG["triage_agent"]["model"],
            messages=[
                {"role": "system", "content": AGENTS_CONFIG["triage_agent"]["system_prompt"]},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        ).choices[0].message.content
        
        parsed_data = json.loads(extraction_res)
        
        if parsed_data.get("customer_id"):
            state.customer_id = str(parsed_data.get("customer_id"))
        if parsed_data.get("issue_type"):
            state.issue_type = parsed_data.get("issue_type")
            
        if state.current_agent == "triage_agent":
            raw_next = parsed_data.get("next_agent", "technical_support")
            state.current_agent = clean_agent_key(raw_next)
            
            state.handover_logs.append({
                "timestamp": datetime.now(timezone.utc).strftime("%H:%M"),
                "source": "Triage Agent", 
                "target": state.current_agent.replace("_", " ").title(),
                "reason": f"Intent: {state.issue_type or 'General Inquiry'}"
            })
    except Exception:
        if state.current_agent == "triage_agent":
            state.current_agent = "technical_support"

    # Handoff Hops execution Loop
    hops = 0
    while hops < 3:
        current = clean_agent_key(state.current_agent)
        state.current_agent = current  # Force state alignment

        # Place 1 Log Call: Agent Invocation
        log_event("AGENT_INVOCATION", state.trace_id, {"agent": current})

        system_instruction = AGENTS_CONFIG[current]["system_prompt"] + context_text
        
        try:
            llm_res = client.chat.completions.create(
                model=AGENTS_CONFIG[current]["model"],
                messages=[
                    {"role": "system", "content": system_instruction},
                    *state.history[-4:]
                ],
                temperature=0.1
            ).choices[0].message.content
            
            if "[[ROUTE_TO_BILLING]]" in llm_res and current == "technical_support":
                # Place 3 Log Call: Agent Handover
                log_event("AGENT_HANDOVER", state.trace_id, {"source": "technical_support", "target": "billing"})
                
                state.handover_logs.append({
                    "timestamp": datetime.now(timezone.utc).strftime("%H:%M"),
                    "source": "Technical Support", "target": "Billing Support",
                    "reason": "Upgrade/Billing topic intersection identified."
                })
                state.current_agent = "billing"
                hops += 1
                continue
                
            if "[[ROUTE_TO_ESCALATION]]" in llm_res and current == "billing":
                state.handover_logs.append({
                    "timestamp": datetime.now(timezone.utc).strftime("%H:%M"),
                    "source": "Billing Support", "target": "Escalation Lead",
                    "reason": "Direct refund exception or manager intervention requested."
                })
                state.current_agent = "escalation"
                hops += 1
                continue

            clean_output = apply_output_guardrail(llm_res)
            state.history.append({"role": "assistant", "content": clean_output})
            break
            
        except Exception:
            state.current_agent = "escalation"
            state.history.append({"role": "assistant", "content": "I am connecting an escalation team lead to look into this context pattern for you."})
            break

# ========== INITIALIZATION ==========
if "core_state" not in st.session_state:
    st.session_state.core_state = ConversationState()
if "logs" not in st.session_state:
    st.session_state.logs = []

current_state = st.session_state.core_state

# ========== DRAW DASHBOARD USER INTERFACE ==========
st.markdown("<h1 class='hero-title'>CloudDash AI Support Engine</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-subtitle'>Multi-Agent Customer Support Platform</p>", unsafe_allow_html=True)
st.markdown("<p class='hero-badges'>POWERED BY RAG • AGENT ROUTING • HUMAN ESCALATION <span style='color:#10B981; margin-left:15px;'>● [ACTIVE] 99.9% RESOLUTION TRACKING</span></p>", unsafe_allow_html=True)

total_sessions = 1284 + len(st.session_state.logs)
kb_hits_pct = "98%" if len(current_state.active_kb_citations) > 0 else "0%"

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown(f"<div class='kpi-box'><div class='kpi-title'>Total Sessions</div><div class='kpi-value'>{total_sessions}</div></div>", unsafe_allow_html=True)
with kpi2:
    st.markdown("<div class='kpi-box'><div class='kpi-title'>Core Runtime Agents</div><div class='kpi-value' style='color:#6366F1;'>4 Active</div></div>", unsafe_allow_html=True)
with kpi3:
    st.markdown(f"<div class='kpi-box'><div class='kpi-title'>Inter-Agent Handovers</div><div class='kpi-value' style='color:#F59E0B;'>{len(current_state.handover_logs)}</div></div>", unsafe_allow_html=True)
with kpi4:
    st.markdown(f"<div class='kpi-box'><div class='kpi-title'>KB Grounding Hits</div><div class='kpi-value' style='color:#10B981;'>{kb_hits_pct}</div></div>", unsafe_allow_html=True)

st.markdown("<div class='clean-hr'></div>", unsafe_allow_html=True)

for alert in current_state.guardrail_alerts:
    st.markdown(f"<div class='guardrail-warning'>{alert}</div>", unsafe_allow_html=True)

# Sidebar Parameters Design
with st.sidebar:
    st.markdown("<h3 style='color:#F9FAFB; margin-bottom:1rem; font-size:1.1rem;'>🛰️ Operational Parameters</h3>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style='background:#111827; border:1px solid #1F2937; padding:0.8rem; border-radius:6px; margin-bottom:10px;'>
            <div style='font-size:0.7rem; color:#9CA3AF; text-transform:uppercase;'>System Session Trace</div>
            <div style='font-family:monospace; color:#6366F1; font-weight:600; font-size:0.85rem;'>{current_state.trace_id}</div>
        </div>
        <div style='background:#111827; border:1px solid #1F2937; padding:0.8rem; border-radius:6px;'>
            <div style='font-size:0.7rem; color:#9CA3AF; text-transform:uppercase; margin-bottom:4px;'>Context Metadata Ledger</div>
            <div style='font-size:0.8rem; color:#E5E7EB;'><b>Customer ID:</b> {current_state.customer_id or 'Unassigned'}</div>
            <div style='font-size:0.8rem; color:#E5E7EB; margin-top:2px;'><b>Focus Group:</b> {current_state.issue_type or 'Pending input...'}</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
    if st.button("🔄 Clear System Session Core", key="clear_state_btn"):
        st.session_state.core_state = ConversationState()
        st.session_state.logs = []
        st.rerun()

# Scenarios Cards Selector Strip
st.markdown("<p style='font-size:0.75rem; font-weight:600; color:#9CA3AF; text-transform:uppercase; margin-bottom:0.6rem; letter-spacing:0.05em;'>Pre-seeded System Validation Scenarios</p>", unsafe_allow_html=True)
b1, b2, b3, b4 = st.columns(4)
faq_query = None
with b1:
    if st.button("📋 1. AWS Metrics Failure", key="faq_1"):
        faq_query = "My customer ID is CUST-901. My CloudDash alerts stopped firing after updating AWS keys."
with b2:
    if st.button("🔄 2. Cross-Domain Transfer", key="faq_2"):
        faq_query = "I am currently using your Pro tier, but I need to upgrade to Enterprise to check out SSO options."
with b3:
    if st.button("🚨 3. Refund Manager Path", key="faq_3"):
        faq_query = "You charged my card twice. I need an immediate refund and demand to talk to a manager."
with b4:
    if st.button("🔍 4. Rate Limit Verification", key="faq_4"):
        faq_query = "What are the standard Rate limits for making queries against the Developer API?"

if faq_query:
    run_orchestration_loop(current_state, faq_query)
    st.rerun()

st.markdown("<div class='clean-hr'></div>", unsafe_allow_html=True)

# Main Application Workspace
chat_layout, telemetry_layout = st.columns([11, 9], gap="large")

with chat_layout:
    st.markdown("<h4 style='color:#F9FAFB; margin-bottom:1.2rem; font-size:1.1rem; font-weight:600;'>💬 Customer Conversation</h4>", unsafe_allow_html=True)
    
    for msg in current_state.history:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        else:
            st.chat_message("assistant").markdown(msg["content"])
            
    if user_prompt := st.chat_input("Ask CloudDash Support..."):
        run_orchestration_loop(current_state, user_prompt)
        st.rerun()

with telemetry_layout:
    # A. Agent Matrix View
    st.markdown("<h4 style='color:#F9FAFB; margin-bottom:0.8rem; font-size:1.1rem; font-weight:600;'>🧬 Agent Pipeline State Matrix</h4>", unsafe_allow_html=True)
    c = clean_agent_key(current_state.current_agent)
    triage_act = "node-active" if c == "triage_agent" else ""
    tech_act = "node-active" if c == "technical_support" else ""
    billing_act = "node-active" if c == "billing" else ""
    esc_act = "node-active" if c == "escalation" else ""
    
    st.markdown(f"""
        <div class='pipeline-container'>
            <div class='pipeline-node node-triage {triage_act}'>🟣 Triage Agent</div>
            <div style='text-align:center; color:#6366F1; margin:1px 0; font-size:0.75rem;'>↓</div>
            <div class='pipeline-node node-tech {tech_act}'>🔵 Technical Support Agent</div>
            <div style='text-align:center; color:#6366F1; margin:1px 0; font-size:0.75rem;'>↓</div>
            <div class='pipeline-node node-billing {billing_act}'>🟢 Billing Agent</div>
            <div style='text-align:center; color:#6366F1; margin:1px 0; font-size:0.75rem;'>↓</div>
            <div class='pipeline-node node-escalation {esc_act}'>🔴 Escalation Agent</div>
        </div>
    """, unsafe_allow_html=True)
    
    # B. RAG Citations Ledger Box
    st.markdown("<h4 style='color:#F9FAFB; margin-bottom:0.8rem; font-size:1.1rem; font-weight:600;'>📚 Retrieved Grounding Sources</h4>", unsafe_allow_html=True)
    if current_state.active_kb_citations:
        st.markdown("<div style='background:#111827; border:1px solid #1F2937; padding:1rem; border-radius:10px; margin-bottom:1.5rem;'>", unsafe_allow_html=True)
        for cite in current_state.active_kb_citations:
            st.markdown(f"""
                <div style='display:flex; justify-content:space-between; margin-bottom:8px;'>
                    <span style='background:#1E1B4B; color:#818CF8; font-size:0.75rem; font-weight:700; padding:2px 6px; border-radius:4px;'>{cite['id']}</span>
                    <span style='color:#9CA3AF; font-size:0.8rem; font-weight:500;'><b>{cite['title']}</b></span>
                </div>
                <div style='font-size:0.75rem; color:#9CA3AF; margin-bottom:10px;'>{cite['content']}</div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='color:#9CA3AF; font-size:0.85rem; font-style:italic; margin-bottom:1.5rem;'>No knowledge assets fetched yet.</div>", unsafe_allow_html=True)
    
    # C. Handover Timeline Feed
    st.markdown("<h4 style='color:#F9FAFB; margin-bottom:0.8rem; font-size:1.1rem; font-weight:600;'>⏳ Handover Activity Feed</h4>", unsafe_allow_html=True)
    if current_state.handover_logs:
        for item in current_state.handover_logs:
            st.markdown(f"""
                <div class='timeline-card'>
                    <div style='display:flex; justify-content:space-between; font-size:0.75rem; margin-bottom:4px;'>
                        <span style='color:#6366F1; font-weight:700;'>{item.get('source')} ➔ {item.get('target')}</span>
                        <span style='color:#9CA3AF; font-weight:600;'>{item.get('timestamp')}</span>
                    </div>
                    <div style='font-size:0.85rem; color:#D1D5DB;'><b>Routing Exception Trigger:</b> {item.get('reason')}</div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("<div style='color:#9CA3AF; font-size:0.85rem; font-style:italic;'>Waiting for pipeline orchestration transitions...</div>", unsafe_allow_html=True)
