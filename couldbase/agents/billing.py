# agents/billing.py

from app import ConversationState, call_llm, AGENTS_CONFIG, execute_rag_lookup
from handover.protocol import HandoverManager

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
        log_trace = HandoverManager.create_handover_trace(
            source_agent="billing",
            target_agent="escalation",
            reason="Manager authorization threshold requirements hit.",
            customer_id=state.customer_id,
            issue_type=state.issue_type
        )
        state.handover_logs.append(log_trace)
        state.current_agent = "escalation"
        return "HANDOVER"
    
    state.history.append({"role": "assistant", "content": res})
    return res
