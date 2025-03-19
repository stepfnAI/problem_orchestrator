"""
Mapping Flow - Maps raw data fields to standardized formats.
"""

from typing import Dict, Any
import logging
import json

from .base_flow import BaseFlow

logger = logging.getLogger(__name__)

class MappingFlow(BaseFlow):
    """
    Flow for mapping raw data fields to standardized formats.
    
    This flow analyzes the schema of input tables and proposes mappings
    for each field (ID, date, categorical, numerical, target).
    """
    
    def __init__(self, flow_id: str = "mapping_flow"):
        """
        Initialize the Mapping Flow.
        
        Args:
            flow_id: Unique identifier for the flow
        """
        super().__init__(flow_id)
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the mapping flow.
        
        Args:
            input_data: Input data for the flow, including:
                - problem_context: Context of the problem
                - current_state: Current state of the system
                - input_tables: List of input tables
                
        Returns:
            Flow execution results
        """
        logger.info("Executing Mapping Flow")
        
        # Extract input data
        problem_context = input_data.get("problem_context", {})
        current_state = input_data.get("current_state", {})
        input_tables = input_data.get("input_tables", [])
        
        # Check if we have input tables
        if not input_tables:
            return {
                "status": "failed",
                "error": "No input tables provided"
            }
        
        # Get the mapping agent from the meta agent
        meta_agent = self._get_meta_agent()
        mapping_agent = meta_agent.mapping_agent
        
        # Process each input table
        mapped_tables = {}
        for table_id in input_tables:
            logger.info(f"Processing table: {table_id}")
            
            # Get table schema from state
            table_schema = current_state.get("tables", {}).get(table_id, {}).get("schema", {})
            
            if not table_schema:
                logger.warning(f"No schema found for table {table_id}")
                continue
            
            print("\n=== Mapping Agent Analysis ===")
            print(f"Input Schema: {table_schema}")
            
            # Get field mappings
            field_mappings = mapping_agent.propose_field_mappings(table_schema)
            print(f"\nStandard Field Mappings:")
            print(json.dumps(field_mappings, indent=2))
            print("===========================\n")
            
            # Store the mappings
            mapped_tables[table_id] = {
                "field_mappings": field_mappings
            }
        
        # Create output table with mappings
        output_table_id = f"mapped_data_{self._generate_id()}"
        
        # Prepare output table schema
        output_schema = {}
        for table_id, table_info in mapped_tables.items():
            for field, field_type in current_state.get("tables", {}).get(table_id, {}).get("schema", {}).items():
                mapping = table_info["field_mappings"].get(field, "OTHER")
                output_schema[field] = {
                    "type": field_type,
                    "standard_name": mapping,
                    "source_table": table_id
                }
        
        # Return results
        return {
            "status": "completed",
            "input_tables": input_tables,
            "output_table": output_table_id,
            "mapped_tables": mapped_tables,
            "summary": f"Mapped {len(output_schema)} fields across {len(input_tables)} tables",
            "tables": {
                output_table_id: {
                    "schema": output_schema,
                    "creating_flow": self.flow_id,
                    "flow_history": [
                        {
                            "flow_id": self.flow_id,
                            "timestamp": self._get_timestamp(),
                            "summary": f"Created standard field mappings for {len(output_schema)} fields from {len(input_tables)} tables"
                        }
                    ]
                }
            }
        }
    
    # Implement the abstract method
    def _execute_implementation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implementation of the flow execution.
        
        This method is called by the base class's execute method.
        
        Args:
            input_data: Input data for the flow
            
        Returns:
            Flow execution results
        """
        # Simply delegate to the execute method
        # This is a workaround to maintain compatibility with the abstract base class
        return self.execute(input_data)
    
    def _get_meta_agent(self):
        """Get the meta agent instance."""
        # First try to get the meta agent from the input data
        if hasattr(self, 'input_data') and self.input_data and 'meta_agent' in self.input_data:
            return self.input_data['meta_agent']
        
        # If not available, use the singleton pattern
        from meta_agent.meta_agent import MetaAgent
        return MetaAgent.get_instance() 