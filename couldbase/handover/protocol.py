# handover/protocol.py

import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional

class HandoverManager:
    @staticmethod
    def create_handover_trace(
        source_agent: str, 
        target_agent: str, 
        reason: str, 
        customer_id: Optional[str], 
        issue_type: Optional[str]
    ) -> Dict[str, Any]:
        """
        Generates a structured audit footprint payload whenever a conversation 
        crosses domain boundaries between specialized agents.
        """
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source_agent,
            "target": target_agent,
            "reason": reason,
            "context_snapshot": {
                "customer_id": customer_id or "Unidentified",
                "issue_type": issue_type or "Unclassified"
            }
        }

    @staticmethod
    def verify_state_integrity(state_dict: dict) -> bool:
        """
        Validates that critical session state components and history 
        tokens are intact before authorizing an agent migration pass.
        """
        required_keys = ["trace_id", "current_agent", "history"]
        return all(key in state_dict for key in required_keys)
