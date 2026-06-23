# tests/test_orchestration.py

import pytest
from app import ConversationState, process_customer_turn

def test_triage_routing_to_technical():
    """Verify that technical problems are correctly routed to technical_support."""
    state = ConversationState()
    user_query = "My AWS integration keys are failing with a permission error."
    
    process_customer_turn(state, user_query)

    assert state.current_agent == "technical_support"
    assert len(state.handover_logs) > 0
    assert state.handover_logs[0]["source"] == "triage_agent"

def test_state_context_preservation():
    """Verify that trace IDs and customer metrics persist across turns."""
    state = ConversationState()
    initial_trace_id = state.trace_id

    process_customer_turn(state, "My customer ID is CUST-555. Having billing issues.")

    process_customer_turn(state, "I want to speak with a manager.")

    assert state.trace_id == initial_trace_id
    assert state.customer_id == "CUST-555"
