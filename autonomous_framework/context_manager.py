from typing import Dict, Any, List, Optional
import pandas as pd
import json
import os
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)

class ContextManager:
    """Manages context and memory for the autonomous agent system"""
    
    def __init__(self, session_id=None):
        """Initialize the context manager
        
        Args:
            session_id: Optional session ID to resume
        """
        self.session_id = session_id or str(uuid.uuid4())
        
        # Initialize context
        self.context = {
            'session_id': self.session_id,
            'current_goal': None,
            'tables': {},  # Store DataFrames
            'available_tables': [],  # List of table names
            'tables_metadata': {},  # Metadata about tables
            'variables': {},  # Store variables
            'history': [],  # Store history of operations
            'current_agent': None,  # Current agent
            'next_agent': None,  # Next agent to call
            'errors': []  # Store errors
        }
        
        # Load session if provided
        if session_id:
            self._load_session()
        
        self.storage_dir = os.path.join("./sessions", self.session_id)
        self.memory: Dict[str, Any] = {
            "tables": {},
            "agent_outputs": {},
            "workflow_state": {
                "current_goal": None,
                "completed_steps": [],
                "current_step": None,
                "errors": []
            },
            "metadata": {
                "session_id": self.session_id,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }
        }
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_dir, exist_ok=True)
        
    def store_table(self, table_name: str, df: pd.DataFrame, metadata: Dict = None):
        """Store a table in memory"""
        self.memory["tables"][table_name] = {
            "data": df,
            "metadata": metadata or {},
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "added_at": datetime.now().isoformat()
        }
        self._update_last_modified()
        
        # Save to disk for persistence
        df.to_parquet(os.path.join(self.storage_dir, f"{table_name}.parquet"))
        
    def get_table(self, table_name: str):
        """Get a table from the context
        
        Args:
            table_name: Name of the table
            
        Returns:
            DataFrame or None if not found
        """
        # First try to get from memory tables
        if table_name in self.memory["tables"] and "data" in self.memory["tables"][table_name]:
            return self.memory["tables"][table_name]["data"]
        
        # Then try to get from context tables
        if table_name in self.context["tables"]:
            return self.context["tables"][table_name]
        
        return None
    
    def list_tables(self) -> List[str]:
        """List all available tables"""
        return list(self.memory["tables"].keys())
    
    def get_table_metadata(self, table_name: str) -> Optional[Dict]:
        """Get metadata for a table"""
        table_info = self.memory["tables"].get(table_name)
        return table_info["metadata"] if table_info else None
    
    def store_agent_output(self, agent_name: str, output: Dict[str, Any]):
        """Store output from an agent in the context
        
        Args:
            agent_name: Name of the agent
            output: Output from the agent
        """
        # Store the output
        self.memory["agent_outputs"][agent_name] = output
        
        # Add to completed steps
        self.memory["workflow_state"]["completed_steps"].append(agent_name)
        
        # Update current step
        self.memory["workflow_state"]["current_step"] = agent_name
        
        # Update last modified
        self._update_last_modified()
        
        # Log the addition
        logger.info(f"Stored output from {agent_name} in context")
        
    def get_agent_output(self, agent_name: str) -> Optional[Dict]:
        """Get output from an agent"""
        agent_info = self.memory["agent_outputs"].get(agent_name)
        return agent_info["output"] if agent_info else None
    
    def set_current_goal(self, goal: str):
        """Set the current goal"""
        self.memory["workflow_state"]["current_goal"] = goal
        self._update_last_modified()
        
    def get_current_goal(self) -> Optional[str]:
        """Get the current goal"""
        return self.memory["workflow_state"]["current_goal"]
    
    def set_current_step(self, step: str):
        """Set the current step"""
        self.memory["workflow_state"]["current_step"] = step
        self._update_last_modified()
        
    def get_current_step(self) -> Optional[str]:
        """Get the current step"""
        return self.memory["workflow_state"]["current_step"]
    
    def add_error(self, error: str, agent_name: str = None):
        """Add an error to the workflow state"""
        self.memory["workflow_state"]["errors"].append({
            "error": error,
            "agent": agent_name,
            "timestamp": datetime.now().isoformat()
        })
        self._update_last_modified()
    
    def get_workflow_summary(self) -> Dict:
        """Get a summary of the workflow state"""
        return {
            "goal": self.memory["workflow_state"]["current_goal"],
            "completed_steps": len(self.memory["workflow_state"]["completed_steps"]),
            "current_step": self.memory["workflow_state"]["current_step"],
            "error_count": len(self.memory["workflow_state"]["errors"]),
            "tables": list(self.memory["tables"].keys()),
            "agent_outputs": list(self.memory["agent_outputs"].keys())
        }
    
    def get_context_for_llm(self) -> Dict:
        """Get context formatted for LLM consumption"""
        # Create a simplified version of the context for the LLM
        tables_info = {}
        for name, info in self.memory["tables"].items():
            tables_info[name] = {
                "shape": info["shape"],
                "columns": info["columns"],
                "dtypes": {k: str(v) for k, v in info["dtypes"].items()},  # Convert dtypes to strings
                "metadata": info["metadata"]
            }
            
        # Count how many times each agent has been called
        agent_call_counts = {}
        for agent_name in self.memory["workflow_state"]["completed_steps"]:
            agent_call_counts[agent_name] = agent_call_counts.get(agent_name, 0) + 1
        
        return {
            "tables": tables_info,
            "workflow_state": self.memory["workflow_state"],
            "completed_agent_outputs": self.memory["agent_outputs"],
            "agent_call_counts": agent_call_counts  # Add this to help the meta-agent
        }
    
    def _update_last_modified(self):
        """Update the last modified timestamp"""
        self.memory["metadata"]["last_updated"] = datetime.now().isoformat()
        
    def save_state(self):
        """Save the current state to disk"""
        # Create a copy of memory without the actual dataframes
        memory_copy = self.memory.copy()
        for table_name in memory_copy["tables"]:
            if "data" in memory_copy["tables"][table_name]:
                # Don't save the actual dataframe to JSON
                memory_copy["tables"][table_name]["data"] = "DataFrame stored separately"
        
        # Save to disk with custom encoder
        with open(os.path.join(self.storage_dir, "state.json"), "w") as f:
            json.dump(memory_copy, f, indent=2, cls=NumpyEncoder)
    
    def load_state(self, session_id: str):
        """Load state from disk"""
        self.session_id = session_id
        self.storage_dir = os.path.join(self.storage_dir.split(self.session_id)[0], session_id)
        
        # Load state from disk
        with open(os.path.join(self.storage_dir, "state.json"), "r") as f:
            self.memory = json.load(f)
            
        # Load dataframes
        for table_name in self.memory["tables"]:
            table_path = os.path.join(self.storage_dir, f"{table_name}.parquet")
            if os.path.exists(table_path):
                self.memory["tables"][table_name]["data"] = pd.read_parquet(table_path)

    def add_table(self, table_name: str, df):
        """Add a table to the context
        
        Args:
            table_name: Name of the table
            df: DataFrame object
        """
        self.memory["tables"][table_name] = {
            "data": df,
            "metadata": {},
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "added_at": datetime.now().isoformat()
        }
        self._update_last_modified()
        
        # Save to disk for persistence
        df.to_parquet(os.path.join(self.storage_dir, f"{table_name}.parquet"))
        
        # Log the addition
        print(f"Added table '{table_name}' to context with {df.shape[0]} rows and {df.shape[1]} columns")

    def get_recent_steps(self, count=3):
        """Get the most recent steps in the workflow
        
        Args:
            count: Number of recent steps to return
            
        Returns:
            List of recent step names
        """
        # Get the completed steps from workflow state
        completed_steps = self.memory["workflow_state"]["completed_steps"]
        
        # Return the last 'count' steps
        return completed_steps[-count:] if completed_steps else [] 