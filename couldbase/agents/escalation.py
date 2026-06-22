# agents/escalation.py

from app import ConversationState, call_llm, AGENTS_CONFIG

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
