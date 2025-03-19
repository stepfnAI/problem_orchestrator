"""
Feature Suggestion Flow - Suggests features based on mapped data.
"""

from typing import Dict, Any
import logging

from .base_flow import BaseFlow

logger = logging.getLogger(__name__)

class FeatureSuggestionFlow(BaseFlow):
    """
    Flow for suggesting features based on mapped data.
    
    This flow analyzes the mapped data and suggests features
    that might be useful for the problem.
    """
    
    def __init__(self, flow_id: str = "feature_suggestion_flow"):
        """
        Initialize the Feature Suggestion Flow.
        
        Args:
            flow_id: Unique identifier for the flow
        """
        super().__init__(flow_id)
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the feature suggestion flow.
        
        Args:
            input_data: Input data for the flow, including:
                - problem_context: Context of the problem
                - current_state: Current state of the system
                - input_tables: List of input tables
                
        Returns:
            Flow execution results
        """
        logger.info("Executing Feature Suggestion Flow")
        
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
        
        # Get the feature agent from the meta agent
        from meta_agent.meta_agent import MetaAgent
        meta_agent = self._get_meta_agent()
        feature_agent = meta_agent.feature_agent
        
        # Get problem type
        problem_type = problem_context.get("problem_type", "unknown")
        
        # Process each input table
        suggested_features = {}
        for table_id in input_tables:
            # Get table schema with mappings from state
            table_info = current_state.get("tables", {}).get(table_id, {})
            schema = table_info.get("schema", {})
            
            if not schema:
                logger.warning(f"No schema found for table {table_id}")
                continue
            
            # Prepare schema with mappings for the feature agent
            schema_with_mappings = {}
            for field, field_info in schema.items():
                if isinstance(field_info, dict):
                    schema_with_mappings[field] = field_info
                else:
                    # Handle case where schema is just field -> type mapping
                    schema_with_mappings[field] = {
                        "type": field_info,
                        "mapping": "unknown"
                    }
            
            # Suggest features
            features = feature_agent.suggest_features(schema_with_mappings, problem_type)
            
            # Evaluate features
            evaluated_features = feature_agent.evaluate_features(
                {field: info.get("type", "unknown") for field, info in schema_with_mappings.items()},
                problem_type,
                features
            )
            
            # Store the suggestions
            suggested_features[table_id] = evaluated_features
        
        # Create output table with feature suggestions
        output_table_id = f"feature_suggestions_{self._generate_id()}"
        
        # Flatten all suggestions into a single list
        all_suggestions = []
        for table_id, features in suggested_features.items():
            for feature in features:
                feature_copy = feature.copy()
                feature_copy["source_table"] = table_id
                all_suggestions.append(feature_copy)
        
        # Sort by importance
        all_suggestions = sorted(all_suggestions, key=lambda x: x.get("importance", 0), reverse=True)
        
        # Return results
        return {
            "status": "completed",
            "input_tables": input_tables,
            "output_table": output_table_id,
            "suggested_features": suggested_features,
            "summary": f"Suggested {len(all_suggestions)} features across {len(input_tables)} tables",
            "tables": {
                output_table_id: {
                    "features": all_suggestions,
                    "creating_flow": self.flow_id,
                    "flow_history": [
                        {
                            "flow_id": self.flow_id,
                            "timestamp": self._get_timestamp(),
                            "summary": f"Created feature suggestions with {len(all_suggestions)} features"
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