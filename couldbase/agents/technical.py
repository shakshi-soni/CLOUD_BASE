# agents/technical.py

from datetime import datetime, timezone
from app import ConversationState, call_llm, AGENTS_CONFIG, execute_rag_lookup
from handover.protocol import HandoverManager

def run_technical_agent(state: ConversationState) -> str:
    last_query = [m["content"] for m in state.history if m["role"] == "user"][-1]
    kb_doc, kb_id = execute_rag_lookup(state, last_query, category="troubleshooting")
    
    if not kb_doc:
        kb_doc, kb_id = execute_rag_lookup(state, last_query, category="faqs")

    if not kb_doc:
        log_trace = HandoverManager.create_handover_trace(
            source_agent="technical_support",
            target_agent="escalation",
            reason="Knowledge gap fallback; no relevant KB items located.",
            customer_id=state.customer_id,
            issue_type=state.issue_type
        )
        state.handover_logs.append(log_trace)
        state.current_agent = "escalation"
        return "HANDOVER"
    
    system = f"{AGENTS_CONFIG['technical_support']['system_prompt']}\nContext Grounding:\n[{kb_id}]: {kb_doc}"
    messages = [{"role": "system", "content": system}] + [{"role": m["role"], "content": m["content"]} for m in state.history[-4:]]
    messages.append({"role": "user", "content": "Note: If pricing or plan alterations are explicitly stated, append [[ROUTE_TO_BILLING]]"})
    
    res = call_llm(state, "technical_support", messages)
    if "[[ROUTE_TO_BILLING]]" in res:
        log_trace = HandoverManager.create_handover_trace(
            source_agent="technical_support",
            target_agent="billing",
            reason="Cross domain boundary intersection detected (Billing intent captured).",
            customer_id=state.customer_id,
            issue_type=state.issue_type
        )
        state.handover_logs.append(log_trace)
        state.current_agent = "billing"
        return "HANDOVER"
    
    state.history.append({"role": "assistant", "content": res})
    return res
