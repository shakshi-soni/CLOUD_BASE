# agents/triage.py

import json
from datetime import datetime, timezone
from app import ConversationState, call_llm, AGENTS_CONFIG
from handover.protocol import HandoverManager

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
            log_trace = HandoverManager.create_handover_trace(
                source_agent="triage_agent",
                target_agent=target,
                reason=data.get("reason", "Initial routing distribution"),
                customer_id=state.customer_id,
                issue_type=state.issue_type
            )
            state.handover_logs.append(log_trace)
            state.current_agent = target
        return "HANDOVER"
    except:
        state.current_agent = "escalation"
        return "HANDOVER"
