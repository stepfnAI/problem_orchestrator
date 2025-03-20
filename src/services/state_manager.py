"""
State Manager - Manages the state of the system.
"""

from typing import Dict, Any, Optional
import logging
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class StateManager:
    """
    Manages the state of the system.
    
    This service maintains the current state of the system, including
    problem context, available tables, flow history, and other metadata.
    """
    
    # Singleton instance
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance of the StateManager."""
        if cls._instance is None:
            cls._instance = StateManager()
        return cls._instance
    
    def __init__(self):
        """Initialize the State Manager."""
        # Only initialize if this is not a duplicate instance
        if StateManager._instance is None:
            self.state = {}
            self.state_file = os.path.join(os.getcwd(), "state.json")
            self._load_state()
            StateManager._instance = self
        else:
            logger.warning("Attempted to create a second instance of StateManager. Using existing instance.")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state.
        
        Returns:
            Current state dictionary
        """
        return self.state
    
    def update_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the state.
        
        Args:
            state: New state dictionary
            
        Returns:
            Updated state dictionary
        """
        self.state = state
        self._save_state()
        return self.state
    
    def initialize_state(self, problem_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize the state for a new problem.
        
        Args:
            problem_context: Context of the problem
            
        Returns:
            Initialized state dictionary
        """
        # Preserve existing tables and available_tables if they exist
        existing_tables = self.state.get("tables", {})
        available_tables = self.state.get("available_tables", [])
        
        # Create new state
        self.state = {
            "problem_context": problem_context,
            "status": "initializing",
            "flow_history": [],
            "tables": existing_tables,
            "available_tables": available_tables,
            "created_at": self._get_timestamp(),
            "updated_at": self._get_timestamp()
        }
        
        self._save_state()
        return self.state
    
    def set_state_status(self, status: str) -> None:
        """
        Set the status of the state.
        
        Args:
            status: New status
        """
        self.state["status"] = status
        self.state["updated_at"] = self._get_timestamp()
        self._save_state()
    
    def add_flow_to_history(self, flow_id: str, result: Dict[str, Any]) -> None:
        """
        Add a flow execution to the history.
        
        Args:
            flow_id: ID of the flow
            result: Result of the flow execution
        """
        if "flow_history" not in self.state:
            self.state["flow_history"] = []
        
        self.state["flow_history"].append({
            "flow_id": flow_id,
            "timestamp": self._get_timestamp(),
            "status": result.get("status", "unknown"),
            "summary": result.get("summary", "")
        })
        
        self.state["updated_at"] = self._get_timestamp()
        self._save_state()
    
    def set_next_flow(self, flow_id: Optional[str]) -> None:
        """
        Set the next flow to execute.
        
        Args:
            flow_id: ID of the next flow, or None if no more flows
        """
        self.state["next_flow"] = flow_id
        self.state["updated_at"] = self._get_timestamp()
        self._save_state()
    
    def get_next_flow(self) -> Optional[str]:
        """
        Get the next flow to execute.
        
        Returns:
            ID of the next flow, or None if no more flows
        """
        return self.state.get("next_flow")
    
    def _get_timestamp(self) -> str:
        """Get the current timestamp."""
        return datetime.now().isoformat()
    
    def _save_state(self) -> None:
        """Save the state to a file."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving state: {str(e)}")
    
    def _load_state(self) -> None:
        """Load the state from a file."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    self.state = json.load(f)
                logger.info("State loaded from file")
            else:
                logger.info("No state file found, starting with empty state")
        except Exception as e:
            logger.error(f"Error loading state: {str(e)}")
            self.state = {} 