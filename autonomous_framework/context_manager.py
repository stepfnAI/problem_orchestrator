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
        self.memory = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            },
            "workflow_state": {
                "current_goal": None,
                "current_step": None,
                "completed_steps": [],
                "errors": []
            },
            "tables": {},
            "tables_metadata": {},  # Add this line to initialize tables_metadata
            "agent_outputs": {},
            "processed_agent_outputs": {},  # New field for processed outputs
            "current_dataframe": None  # Add dataframe storage
        }
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_dir, exist_ok=True)
        
    def store_table(self, table_name: str, df: pd.DataFrame, metadata: Dict = None):
        """Store a table in memory"""
        # Store the actual DataFrame directly in the tables dictionary
        self.memory["tables"][table_name] = df
        
        # Also store metadata separately
        self.memory["tables_metadata"][table_name] = {
            "metadata": metadata or {},
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "added_at": datetime.now().isoformat()
        }
        
        # Set as current dataframe if none exists
        if self.memory["current_dataframe"] is None:
            self.memory["current_dataframe"] = df
        
        self._update_last_modified()
        
        # Save to disk for persistence
        df.to_parquet(os.path.join(self.storage_dir, f"{table_name}.parquet"))
        
    def get_table(self, table_name: str) -> Optional[pd.DataFrame]:
        """Get a table from the context
        
        Args:
            table_name: Name of the table
            
        Returns:
            DataFrame or None if not found
        """
        # Check if it's in the tables dictionary
        if table_name in self.memory["tables"]:
            table_data = self.memory["tables"][table_name]
            if isinstance(table_data, pd.DataFrame):
                return table_data
        
        # If we get here, the table wasn't found in memory
        # Try to load from disk as a fallback
        try:
            parquet_path = os.path.join(self.storage_dir, f"{table_name}.parquet")
            if os.path.exists(parquet_path):
                df = pd.read_parquet(parquet_path)
                # Store it in memory for future use
                self.memory["tables"][table_name] = df
                return df
        except Exception as e:
            logger.error(f"Error loading table from disk: {str(e)}")
        
        # If all else fails, return the current dataframe as a last resort
        if self.memory["current_dataframe"] is not None:
            return self.memory["current_dataframe"]
        
        # If we get here, the table wasn't found
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
    
    def add_error(self, error_message: str):
        """Add an error to the workflow state
        
        Args:
            error_message: Error message to add
        """
        self.memory["workflow_state"]["errors"].append({
            "message": error_message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update last modified timestamp
        self._update_last_modified()
        
        # Log the error
        logger.error(f"Added error to workflow: {error_message}")
    
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

    def get_recent_steps(self, n: int = 3) -> List[str]:
        """Get the most recent n steps from the workflow
        
        Args:
            n: Number of recent steps to return
            
        Returns:
            List of recent step names
        """
        completed_steps = self.memory["workflow_state"]["completed_steps"]
        return completed_steps[-n:] if completed_steps else []

    def get_current_dataframe(self) -> pd.DataFrame:
        """Get the current working dataframe"""
        return self.memory["current_dataframe"]

    def update_dataframe(self, df: pd.DataFrame):
        """Update the current working dataframe"""
        self.memory["current_dataframe"] = df
        # Also update the table metadata
        self.memory["tables"]["current"] = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": {k: str(v) for k, v in df.dtypes.items()},
            "metadata": {
                "last_updated": datetime.now().isoformat()
            }
        }

    def _load_session(self):
        """Load session from disk"""
        # Implementation of _load_session method
        pass 

    def store_processed_output(self, agent_name, processed_output):
        """Store processed output from an agent"""
        self.memory["processed_agent_outputs"][agent_name] = processed_output
        self._update_last_modified()
        
        # Log the storage
        logger.info(f"Stored processed output from {agent_name}")

    def get_processed_output(self, agent_name):
        """Get processed output from an agent"""
        return self.memory["processed_agent_outputs"].get(agent_name)

    def get_all_processed_outputs(self):
        """Get all processed outputs"""
        return self.memory["processed_agent_outputs"]

    def add_step_to_workflow(self, step_name: str):
        """Add a step to the workflow history
        
        Args:
            step_name: Name of the step/agent that was executed
        """
        # Add to completed steps
        self.memory["workflow_state"]["completed_steps"].append(step_name)
        
        # Update current step
        self.memory["workflow_state"]["current_step"] = step_name
        
        # Update last modified timestamp
        self._update_last_modified()
        
        # Log the addition
        logger.info(f"Added step '{step_name}' to workflow") 