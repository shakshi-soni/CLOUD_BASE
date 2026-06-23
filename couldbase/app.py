""", unsafe_allow_html=True)

# ========== DATA STRUCTURE SCHEMA MODELS ==========
class StructuredLogger:
    @staticmethod
    def log(event_type: str, trace_id: str, payload: dict):
        if "logs" not in st.session_state:
            st.session_state.logs = []
        st.session_state.logs.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "trace_id": trace_id,
            **payload
        })

class ConversationState(BaseModel):
trace_id: str = Field(default_factory=lambda: f"trace_{int(datetime.now(timezone.utc).timestamp())}")
current_agent: str = "triage_agent"
@@ -255,7 +243,20 @@ def apply_output_guardrail(text: str) -> str:
}
}

# ========== PIPELINE MANAGEMENT & EXTRACTION FIX ==========
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

@@ -274,7 +275,7 @@ def run_orchestration_loop(state: ConversationState, user_message: str):
if matches[0] not in state.active_kb_citations:
state.active_kb_citations.append(matches[0])

    # FIX: Extract metadata out of the triage conditional to catch updates at any point in the chat
    # Dynamic extraction turn
try:
extraction_res = client.chat.completions.create(
model=AGENTS_CONFIG["triage_agent"]["model"],
@@ -288,19 +289,20 @@ def run_orchestration_loop(state: ConversationState, user_message: str):

parsed_data = json.loads(extraction_res)

        # Pull new entity extractions into state instantly
if parsed_data.get("customer_id"):
            state.customer_id = parsed_data.get("customer_id")
            state.customer_id = str(parsed_data.get("customer_id"))
if parsed_data.get("issue_type"):
state.issue_type = parsed_data.get("issue_type")

        # Only rewrite routing state if we are currently parked at triage
if state.current_agent == "triage_agent":
            state.current_agent = parsed_data.get("next_agent", "technical_support")
            raw_next = parsed_data.get("next_agent", "technical_support")
            state.current_agent = clean_agent_key(raw_next)
            
state.handover_logs.append({
"timestamp": datetime.now(timezone.utc).strftime("%H:%M"),
                "source": "Triage Agent", "target": state.current_agent.replace("_", " ").title(),
                "reason": f"Intent: {state.issue_type}"
                "source": "Triage Agent", 
                "target": state.current_agent.replace("_", " ").title(),
                "reason": f"Intent: {state.issue_type or 'General Inquiry'}"
})
except Exception:
if state.current_agent == "triage_agent":
@@ -309,7 +311,9 @@ def run_orchestration_loop(state: ConversationState, user_message: str):
# Handoff Hops execution Loop
hops = 0
while hops < 3:
        current = state.current_agent
        current = clean_agent_key(state.current_agent)
        state.current_agent = current  # Force state alignment

system_instruction = AGENTS_CONFIG[current]["system_prompt"] + context_text

try:
@@ -445,7 +449,7 @@ def run_orchestration_loop(state: ConversationState, user_message: str):
with telemetry_layout:
# A. Agent Matrix View
st.markdown("<h4 style='color:#F9FAFB; margin-bottom:0.8rem; font-size:1.1rem; font-weight:600;'>🧬 Agent Pipeline State Matrix</h4>", unsafe_allow_html=True)
    c = current_state.current_agent
    c = clean_agent_key(current_state.current_agent)
triage_act = "node-active" if c == "triage_agent" else ""
tech_act = "node-active" if c == "technical_support" else ""
billing_act = "node-active" if c == "billing" else ""
