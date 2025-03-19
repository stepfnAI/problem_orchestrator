"""
Data Lineage Manager - Tracks the transformation history of all data tables.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DataLineageManager:
    """
    Tracks the transformation history of all data tables.
    
    This service maintains a graph of table transformations, tracks parent-child
    relationships between tables, and records all operations performed on each table.
    """
    
    def __init__(self):
        """Initialize the Data Lineage Manager."""
        self.tables = {}  # table_id -> table_info
        logger.info("Initialized Data Lineage Manager")
    
    def record_transformation(self, input_tables: List[str], output_table: str, 
                             flow_id: str, transformation_summary: str) -> None:
        """
        Record a transformation that creates a new table.
        
        Args:
            input_tables: List of input table IDs
            output_table: ID of the output table
            flow_id: ID of the flow that performed the transformation
            transformation_summary: Summary of the transformation
        """
        # Create or update output table record
        if output_table in self.tables:
            table_info = self.tables[output_table]
        else:
            table_info = {
                "table_id": output_table,
                "created_at": self._get_timestamp(),
                "parent_tables": [],
                "creating_flow": None,
                "flow_history": [],
                "schema": {},
                "child_tables": []
            }
            self.tables[output_table] = table_info
        
        # Update output table info
        table_info["parent_tables"] = input_tables
        table_info["creating_flow"] = flow_id
        table_info["flow_history"].append({
            "flow_id": flow_id,
            "timestamp": self._get_timestamp(),
            "summary": transformation_summary
        })
        
        # Update parent tables' child_tables
        for input_table in input_tables:
            if input_table in self.tables:
                if output_table not in self.tables[input_table]["child_tables"]:
                    self.tables[input_table]["child_tables"].append(output_table)
        
        logger.info(f"Recorded transformation: {input_tables} -> {output_table} by {flow_id}")
    
    def get_table_lineage(self, table_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the lineage information for a table.
        
        Args:
            table_id: ID of the table
            
        Returns:
            Dictionary with table lineage information, or None if not found
        """
        return self.tables.get(table_id)
    
    def get_all_tables(self) -> List[str]:
        """
        Get a list of all table IDs.
        
        Returns:
            List of table IDs
        """
        return list(self.tables.keys())
    
    def get_final_tables(self) -> List[str]:
        """
        Get a list of final table IDs (tables with no children).
        
        Returns:
            List of final table IDs
        """
        return [table_id for table_id, info in self.tables.items() 
                if not info["child_tables"]]
    
    def get_root_tables(self) -> List[str]:
        """
        Get a list of root table IDs (tables with no parents).
        
        Returns:
            List of root table IDs
        """
        return [table_id for table_id, info in self.tables.items() 
                if not info["parent_tables"]]
    
    def get_transformation_path(self, from_table: str, to_table: str) -> List[Dict[str, Any]]:
        """
        Get the transformation path from one table to another.
        
        Args:
            from_table: ID of the source table
            to_table: ID of the destination table
            
        Returns:
            List of transformation steps, or empty list if no path exists
        """
        # This is a simplified implementation
        # In a real system, this would use graph traversal algorithms
        
        if from_table == to_table:
            return []
        
        if to_table not in self.tables:
            return []
        
        to_info = self.tables[to_table]
        
        if from_table in to_info["parent_tables"]:
            return [{
                "from": from_table,
                "to": to_table,
                "flow_id": to_info["creating_flow"],
                "summary": to_info["flow_history"][-1]["summary"]
            }]
        
        # Try to find a path through each parent
        for parent in to_info["parent_tables"]:
            path = self.get_transformation_path(from_table, parent)
            if path:
                path.append({
                    "from": parent,
                    "to": to_table,
                    "flow_id": to_info["creating_flow"],
                    "summary": to_info["flow_history"][-1]["summary"]
                })
                return path
        
        return []
    
    def _get_timestamp(self) -> str:
        """Get the current timestamp."""
        return datetime.now().isoformat() 