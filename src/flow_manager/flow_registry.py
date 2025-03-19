"""
Flow Registry - Manages registration of flows.
"""

import logging
from typing import Dict, Any

from flows.mapping_flow import MappingFlow
from flows.feature_suggestion_flow import FeatureSuggestionFlow

logger = logging.getLogger(__name__)

class FlowRegistry:
    """
    Registry for flows.
    
    This class manages the registration and retrieval of flows.
    """
    
    def __init__(self):
        """Initialize the Flow Registry."""
        self.flows = {}
        self._register_default_flows()
    
    def register_flow(self, flow_id: str, flow_instance: Any) -> None:
        """
        Register a flow.
        
        Args:
            flow_id: Unique identifier for the flow
            flow_instance: Instance of the flow
        """
        logger.info(f"Registering flow: {flow_id}")
        self.flows[flow_id] = flow_instance
    
    def get_flow(self, flow_id: str) -> Any:
        """
        Get a flow by ID.
        
        Args:
            flow_id: ID of the flow to retrieve
            
        Returns:
            Flow instance, or None if not found
        """
        if flow_id not in self.flows:
            logger.warning(f"Flow not found: {flow_id}")
            return None
        
        return self.flows[flow_id]
    
    def _register_default_flows(self) -> None:
        """Register default flows."""
        self.register_flow("mapping_flow", MappingFlow())
        self.register_flow("feature_suggestion_flow", FeatureSuggestionFlow()) 