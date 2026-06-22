# agents/orchestrator.py

from app import ConversationState
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
    # Guardrail Interceptor Input Scan
    if any(x in message.lower() for x in ["ignore previous", "bypass", "system override"]):
        state.history.append({"role": "user", "content": message})
        state.history.append({"role": "assistant", "content": "🛑 **Security Boundary Violation Alert**: Request rejected."})
        return

    if state.current_agent == "triage_agent":
        status = run_triage_agent(state, message)
    else:
        state.history.append({"role": "user", "content": message})
        status = "HANDOVER"
    
    # Loop across agents using state-preserving parameters
    handovers = 0
    while status == "HANDOVER" and handovers < 5:
        agent = state.current_agent
        status = AGENT_REGISTRY.get(agent, run_escalation_agent)(state)
        handovers += 1
