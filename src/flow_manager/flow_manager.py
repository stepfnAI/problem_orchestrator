"""
Flow Manager - Manages execution of flows.
"""

import logging
from typing import Dict, Any

from .flow_registry import FlowRegistry

logger = logging.getLogger(__name__)

class FlowManager:
    """
    Manages the execution of flows.
    
    This class is responsible for executing flows and managing their lifecycle.
    """
    
    def __init__(self):
        """Initialize the Flow Manager."""
        self.registry = FlowRegistry()
    
    def register_flow(self, flow_id: str, flow_instance: Any) -> None:
        """
        Register a flow.
        
        Args:
            flow_id: Unique identifier for the flow
            flow_instance: Instance of the flow
        """
        self.registry.register_flow(flow_id, flow_instance)
    
    def execute_flow(self, flow_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a flow.
        
        Args:
            flow_id: ID of the flow to execute
            input_data: Input data for the flow
            
        Returns:
            Flow execution results
        """
        logger.info(f"Executing flow: {flow_id}")
        
        # Get the flow from the registry
        flow = self.registry.get_flow(flow_id)
        if not flow:
            error_msg = f"Flow {flow_id} not found in registry"
            logger.error(error_msg)
            return {
                "status": "failed",
                "error": error_msg
            }
        
        # Execute the flow
        try:
            result = flow.execute(input_data)
            return result
        except Exception as e:
            error_msg = f"Error executing flow {flow_id}: {str(e)}"
            logger.exception(error_msg)
            return {
                "status": "failed",
                "error": error_msg
            } 