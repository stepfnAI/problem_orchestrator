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
    
    def _execute_implementation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implementation of the flow execution.
        
        This method is called by the base class's execute method.
        
        Args:
            input_data: Input data for the flow
                
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
        
        # Step 1: Get mapping recommendations from the mapping agent
        mapped_tables = self._get_mapping_recommendations(input_tables, current_state)
        if not mapped_tables:
            return {
                "status": "failed",
                "error": "Failed to get mapping recommendations"
            }
        
        # Step 2: Get user confirmation for the mappings
        confirmed_mappings = self._get_user_confirmation(mapped_tables)
        if not confirmed_mappings:
            return {
                "status": "failed",
                "error": "User rejected the mappings"
            }
        
        # Step 3: Create output table with confirmed mappings
        result = self._create_output_table(confirmed_mappings, input_tables, current_state)
        
        return result
    
    def _get_mapping_recommendations(self, input_tables: list, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get mapping recommendations from the mapping agent.
        
        Args:
            input_tables: List of input table IDs
            current_state: Current state of the system
            
        Returns:
            Dictionary of mapped tables
        """
        logger.info("Getting mapping recommendations")
        
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
            
            # Get field mappings - standard_name -> original_field
            field_mappings = mapping_agent.propose_field_mappings(table_schema)
            print(f"\nStandard Field Mappings:")
            print(json.dumps(field_mappings, indent=2))
            print("===========================\n")
            
            # Store the mappings
            mapped_tables[table_id] = {
                "field_mappings": field_mappings,
                "schema": table_schema
            }
        
        return mapped_tables
    
    def _get_user_confirmation(self, mapped_tables: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get user confirmation for the mappings.
        
        Args:
            mapped_tables: Dictionary of mapped tables
            
        Returns:
            Dictionary of confirmed mappings
        """
        logger.info("Getting user confirmation for mappings")
        
        # Get the user input service from the meta agent
        meta_agent = self._get_meta_agent()
        user_input_service = meta_agent.user_input_service
        
        # Force interactive mode for this critical step
        original_interactive = user_input_service.interactive
        user_input_service.interactive = True
        
        confirmed_mappings = {}
        
        try:
            for table_id, table_info in mapped_tables.items():
                field_mappings = table_info["field_mappings"]
                
                print(f"\n=== Confirm Mappings for Table: {table_id} ===")
                print("Please review the proposed field mappings:")
                
                # Display mappings in a user-friendly format
                for std_name, field in field_mappings.items():
                    print(f"  {std_name} -> {field}")
                
                # Ask for confirmation
                confirmation = user_input_service.request_input(
                    question=f"Do you confirm these mappings for table {table_id}?",
                    response_format="yes_no",
                    requester_id="mapping_flow"
                )
                
                if confirmation == "yes":
                    print(f"Mappings confirmed for table {table_id}")
                    confirmed_mappings[table_id] = table_info
                else:
                    print(f"Mappings rejected for table {table_id}")
                    
                    # Allow user to manually adjust mappings
                    print("\nWould you like to manually adjust the mappings?")
                    adjust = user_input_service.request_input(
                        question="Adjust mappings?",
                        response_format="yes_no",
                        requester_id="mapping_flow"
                    )
                    
                    if adjust == "yes":
                        adjusted_mappings = {}
                        schema = table_info["schema"]
                        
                        print("\nAvailable fields:")
                        for i, field in enumerate(schema.keys()):
                            print(f"  {i+1}. {field}")
                        
                        print("\nStandard categories:")
                        std_categories = ["ID", "TIMESTAMP", "CATEGORY", "METRIC", "REVENUE", "PRODUCT", "TARGET", "OTHER"]
                        for i, category in enumerate(std_categories):
                            print(f"  {i+1}. {category}")
                        
                        print("\nEnter mappings in format 'STANDARD_CATEGORY:field_name'")
                        print("Enter 'done' when finished")
                        
                        while True:
                            mapping_input = input("> ")
                            if mapping_input.lower() == 'done':
                                break
                            
                            try:
                                std_cat, field = mapping_input.split(":")
                                std_cat = std_cat.strip().upper()
                                field = field.strip()
                                
                                if std_cat in std_categories and field in schema:
                                    adjusted_mappings[std_cat] = field
                                    print(f"Added mapping: {std_cat} -> {field}")
                                else:
                                    print("Invalid category or field name")
                            except ValueError:
                                print("Invalid format. Use 'STANDARD_CATEGORY:field_name'")
                        
                        if adjusted_mappings:
                            confirmed_mappings[table_id] = {
                                "field_mappings": adjusted_mappings,
                                "schema": schema
                            }
                            print(f"Adjusted mappings saved for table {table_id}")
                        else:
                            print(f"No adjusted mappings provided for table {table_id}")
        finally:
            # Restore original interactive mode
            user_input_service.interactive = original_interactive
        
        return confirmed_mappings
    
    def _create_output_table(self, confirmed_mappings: Dict[str, Any], input_tables: list, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create output table with confirmed mappings.
        
        Args:
            confirmed_mappings: Dictionary of confirmed mappings
            input_tables: List of input table IDs
            current_state: Current state of the system
            
        Returns:
            Flow execution results
        """
        logger.info("Creating output table with confirmed mappings")
        
        # Create output table with mappings
        output_table_id = f"mapped_data_{self._generate_id()}"
        
        # Prepare output table schema
        output_schema = {}
        for table_id, table_info in confirmed_mappings.items():
            # Create a reverse mapping for easier lookup (original_field -> standard_name)
            reverse_mapping = {}
            for std_name, orig_field in table_info["field_mappings"].items():
                if isinstance(orig_field, list):
                    # Handle case where multiple original fields map to one standard field
                    for field in orig_field:
                        reverse_mapping[field] = std_name
                else:
                    reverse_mapping[orig_field] = std_name
            
            # Build the output schema
            for field, field_type in current_state.get("tables", {}).get(table_id, {}).get("schema", {}).items():
                standard_name = reverse_mapping.get(field, "OTHER")
                output_schema[field] = {
                    "type": field_type,
                    "standard_name": standard_name,
                    "source_table": table_id
                }
        
        # Return results
        return {
            "status": "completed",
            "input_tables": input_tables,
            "output_table": output_table_id,
            "mapped_tables": confirmed_mappings,
            "summary": f"Mapped {len(output_schema)} fields across {len(confirmed_mappings)} tables",
            "tables": {
                output_table_id: {
                    "schema": output_schema,
                    "creating_flow": self.flow_id,
                    "flow_history": [
                        {
                            "flow_id": self.flow_id,
                            "timestamp": self._get_timestamp(),
                            "summary": f"Created standard field mappings for {len(output_schema)} fields from {len(confirmed_mappings)} tables"
                        }
                    ]
                }
            }
        }
    
    def _get_meta_agent(self):
        """Get the meta agent instance."""
        # First try to get the meta agent from the input data
        if hasattr(self, 'input_data') and self.input_data and 'meta_agent' in self.input_data:
            return self.input_data['meta_agent']
        
        # If not available, use the singleton pattern
        from meta_agent.meta_agent import MetaAgent
        return MetaAgent.get_instance() 