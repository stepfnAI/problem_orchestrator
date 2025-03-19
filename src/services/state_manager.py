"""
State Manager - Maintains the overall system state and enables persistence.
"""

from typing import Dict, Any, Optional
import logging
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class StateManager:
    """
    Maintains the overall system state and enables persistence.
    
    This service tracks execution progress, stores intermediate results,
    and supports interruption and resumption of workflows.
    """
    
    def __init__(self, state_dir: str = ".state"):
        """
        Initialize the State Manager.
        
        Args:
            state_dir: Directory to store state files
        """
        self.state_dir = state_dir
        self.state = {
            "status": "initializing",
            "current_flow": None,
            "next_flow": None,
            "available_tables": [],
            "results": {},
            "flow_history": [],
            "created_at": self._get_timestamp(),
            "updated_at": self._get_timestamp()
        }
        
        # Create state directory if it doesn't exist
        os.makedirs(state_dir, exist_ok=True)
        
        logger.info("Initialized State Manager")
    
    def initialize_state(self, problem_context: Dict[str, Any]) -> None:
        """
        Initialize the state for a new problem.
        
        Args:
            problem_context: Context information about the problem
        """
        self.state = {
            "status": "initializing",
            "problem_context": problem_context,
            "current_flow": None,
            "next_flow": None,
            "available_tables": [],
            "results": {},
            "flow_history": [],
            "created_at": self._get_timestamp(),
            "updated_at": self._get_timestamp()
        }
        
        logger.info("Initialized state for new problem")
        self._save_state()
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state.
        
        Returns:
            Current state dictionary
        """
        return self.state
    
    def update_state(self, updates: Dict[str, Any]) -> None:
        """
        Update the state with the given updates.
        
        Args:
            updates: Dictionary of state updates
        """
        self.state.update(updates)
        self.state["updated_at"] = self._get_timestamp()
        
        logger.debug(f"Updated state: {updates.keys()}")
        self._save_state()
    
    def set_state_status(self, status: str) -> None:
        """
        Set the status of the state.
        
        Args:
            status: New status
        """
        self.state["status"] = status
        self.state["updated_at"] = self._get_timestamp()
        
        logger.info(f"Set state status to: {status}")
        self._save_state()
    
    def set_current_flow(self, flow_id: str) -> None:
        """
        Set the current flow.
        
        Args:
            flow_id: ID of the current flow
        """
        self.state["current_flow"] = flow_id
        self.state["updated_at"] = self._get_timestamp()
        
        logger.info(f"Set current flow to: {flow_id}")
        self._save_state()
    
    def set_next_flow(self, flow_id: Optional[str]) -> None:
        """
        Set the next flow.
        
        Args:
            flow_id: ID of the next flow, or None if no next flow
        """
        self.state["next_flow"] = flow_id
        self.state["updated_at"] = self._get_timestamp()
        
        logger.info(f"Set next flow to: {flow_id}")
        self._save_state()
    
    def get_next_flow(self) -> Optional[str]:
        """
        Get the next flow to execute.
        
        Returns:
            ID of the next flow, or None if no next flow
        """
        return self.state.get("next_flow")
    
    def add_available_table(self, table_id: str) -> None:
        """
        Add a table to the list of available tables.
        
        Args:
            table_id: ID of the table
        """
        if table_id not in self.state["available_tables"]:
            self.state["available_tables"].append(table_id)
            self.state["updated_at"] = self._get_timestamp()
            
            logger.info(f"Added available table: {table_id}")
            self._save_state()
    
    def update_state_with_flow_result(self, flow_id: str, result: Dict[str, Any]) -> None:
        """
        Update the state with the result of a flow execution.
        
        Args:
            flow_id: ID of the flow
            result: Result of the flow execution
        """
        # Add flow to history
        self.state["flow_history"].append({
            "flow_id": flow_id,
            "timestamp": self._get_timestamp(),
            "status": result.get("status"),
            "summary": result.get("summary", "")
        })
        
        # Update available tables
        if "output_table_name" in result:
            self.add_available_table(result["output_table_name"])
        
        # Update results
        self.state["results"][flow_id] = result.get("output", {})
        
        # Update timestamp
        self.state["updated_at"] = self._get_timestamp()
        
        logger.info(f"Updated state with result of flow: {flow_id}")
        self._save_state()
    
    def save_state(self) -> None:
        """Save the current state."""
        self._save_state()
    
    def load_state(self, state_id: str) -> bool:
        """
        Load a saved state.
        
        Args:
            state_id: ID of the state to load
            
        Returns:
            True if the state was loaded successfully, False otherwise
        """
        state_file = os.path.join(self.state_dir, f"{state_id}.json")
        
        if not os.path.exists(state_file):
            logger.warning(f"State file not found: {state_file}")
            return False
        
        try:
            with open(state_file, "r") as f:
                self.state = json.load(f)
            
            logger.info(f"Loaded state from: {state_file}")
            return True
        except Exception as e:
            logger.exception(f"Error loading state: {e}")
            return False
    
    def _save_state(self) -> None:
        """Save the current state to a file."""
        state_id = self.state.get("state_id", "current")
        state_file = os.path.join(self.state_dir, f"{state_id}.json")
        
        try:
            with open(state_file, "w") as f:
                json.dump(self.state, f, indent=2)
            
            logger.debug(f"Saved state to: {state_file}")
        except Exception as e:
            logger.exception(f"Error saving state: {e}")
    
    def _get_timestamp(self) -> str:
        """Get the current timestamp."""
        return datetime.now().isoformat() 