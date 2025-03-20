"""
Feature Suggestion Flow - Suggests features based on mapped data.
"""

from typing import Dict, List, Any
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime

from .base_flow import BaseFlow

logger = logging.getLogger(__name__)

class FeatureSuggestionFlow(BaseFlow):
    """
    Flow for suggesting features based on mapped data.
    
    This flow analyzes the mapped data and suggests features
    that might be useful for the problem, then generates code
    to implement the approved features.
    """
    
    def __init__(self, flow_id: str = "feature_suggestion_flow"):
        """
        Initialize the Feature Suggestion Flow.
        
        Args:
            flow_id: Unique identifier for the flow
        """
        super().__init__(flow_id)
    
    def _execute_implementation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the feature suggestion flow implementation.
        
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
        
        # Get the meta agent
        meta_agent = self._get_meta_agent()
        
        # Get problem type
        problem_type = problem_context.get("problem_type", "unknown")
        problem_description = problem_context.get("description", "")
        
        # Step 1: Generate feature suggestions
        suggested_features = self._generate_feature_suggestions(
            input_tables, 
            current_state, 
            problem_type,
            problem_description
        )
        
        if not suggested_features:
            return {
                "status": "failed",
                "error": "Failed to generate feature suggestions"
            }
        
        # Step 2: Get user confirmation for each suggestion
        approved_features = self._get_user_confirmation(suggested_features)
        
        if not approved_features:
            return {
                "status": "completed",
                "summary": "No features were approved by the user",
                "details": {
                    "suggested_features": suggested_features,
                    "approved_features": {}
                }
            }
        
        # Step 3: Generate implementation code for approved features
        implemented_features = self._implement_features(
            approved_features, 
            input_tables, 
            current_state
        )
        
        if not implemented_features:
            return {
                "status": "failed",
                "error": "Failed to implement approved features"
            }
        
        # Step 4: Create output table with implemented features
        output_table_id = f"features_{self._generate_id()}"
        
        # Create a summary of implemented features
        feature_summary = []
        for table_name, features in implemented_features.items():
            feature_summary.append(f"Table {table_name}: {len(features)} features implemented")
        
        return {
            "status": "completed",
            "input_tables": input_tables,
            "output_table": output_table_id,
            "summary": f"Implemented {sum(len(features) for features in implemented_features.values())} features across {len(implemented_features)} tables",
            "details": {
                "suggested_features": suggested_features,
                "approved_features": approved_features,
                "implemented_features": implemented_features
            },
            "tables": {
                output_table_id: {
                    "schema": self._create_output_schema(implemented_features),
                    "creating_flow": self.flow_id,
                    "flow_history": [
                        {
                            "flow_id": self.flow_id,
                            "timestamp": self._get_timestamp(),
                            "summary": f"Created {sum(len(features) for features in implemented_features.values())} features"
                        }
                    ]
                }
            }
        }
    
    def _generate_feature_suggestions(self, input_tables: List[str], current_state: Dict[str, Any], 
                                     problem_type: str, problem_description: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate feature suggestions using the Feature Suggestion Agent.
        
        Args:
            input_tables: List of input table IDs
            current_state: Current state of the system
            problem_type: Type of problem (regression, classification, etc.)
            problem_description: Description of the problem
            
        Returns:
            Dictionary of suggested features by table
        """
        logger.info("Generating feature suggestions")
        
        # Get the feature agent from the meta agent
        meta_agent = self._get_meta_agent()
        feature_agent = meta_agent.feature_agent
        
        # Process each input table
        suggested_features = {}
        
        for table_id in input_tables:
            logger.info(f"Processing table: {table_id}")
            
            # Get table schema from state
            table_info = current_state.get("tables", {}).get(table_id, {})
            schema = table_info.get("schema", {})
            
            if not schema:
                logger.warning(f"No schema found for table {table_id}")
                continue
            
            # Prepare schema with standard names if available
            schema_with_standard_names = {}
            for field, field_info in schema.items():
                if isinstance(field_info, dict) and "standard_name" in field_info:
                    standard_name = field_info["standard_name"]
                    field_type = field_info.get("type", "unknown")
                    if isinstance(field_type, dict):
                        field_type = field_type.get("type", "unknown")
                    
                    schema_with_standard_names[field] = {
                        "type": field_type,
                        "standard_name": standard_name
                    }
                else:
                    schema_with_standard_names[field] = {
                        "type": field_info if isinstance(field_info, str) else "unknown",
                        "standard_name": "OTHER"
                    }
            
            print(f"\n=== Feature Suggestion Analysis for Table: {table_id} ===")
            print(f"Problem Type: {problem_type}")
            print(f"Problem Description: {problem_description}")
            print(f"Schema with Standard Names: {json.dumps(schema_with_standard_names, indent=2)}")
            
            # Get feature suggestions
            try:
                features = feature_agent.suggest_features(
                    schema_with_standard_names, 
                    problem_type, 
                    problem_description
                )
                
                # Format and display suggestions
                print("\nSuggested Features:")
                for i, feature in enumerate(features):
                    print(f"{i+1}. {feature['name']}: {feature['description']}")
                    print(f"   Rationale: {feature['rationale']}")
                    print(f"   Complexity: {feature['complexity']}")
                    print()
                
                suggested_features[table_id] = features
                logger.info(f"Generated {len(features)} feature suggestions for table {table_id}")
                
            except Exception as e:
                logger.error(f"Error generating feature suggestions for table {table_id}: {str(e)}")
                print(f"Error generating feature suggestions: {str(e)}")
        
        print("===========================\n")
        return suggested_features
    
    def _get_user_confirmation(self, suggested_features: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get user confirmation for each suggested feature.
        
        Args:
            suggested_features: Dictionary of suggested features by table
            
        Returns:
            Dictionary of approved features by table
        """
        logger.info("Getting user confirmation for feature suggestions")
        
        # Get the user input service from the meta agent
        meta_agent = self._get_meta_agent()
        user_input_service = meta_agent.user_input_service
        
        # Force interactive mode for this critical step
        original_interactive = user_input_service.interactive
        user_input_service.interactive = True
        
        approved_features = {}
        
        try:
            for table_id, features in suggested_features.items():
                approved_for_table = []
                
                print(f"\n=== Feature Suggestions for Table: {table_id} ===")
                print("Please review each suggested feature:")
                
                for feature in features:
                    print(f"\nFeature: {feature['name']}")
                    print(f"Description: {feature['description']}")
                    print(f"Rationale: {feature['rationale']}")
                    print(f"Complexity: {feature['complexity']}")
                    
                    # Ask for confirmation
                    confirmation = user_input_service.request_input(
                        question=f"Do you want to implement this feature?",
                        response_format="yes_no",
                        requester_id="feature_suggestion_flow"
                    )
                    
                    if confirmation == "yes":
                        print(f"Feature '{feature['name']}' approved")
                        approved_for_table.append(feature)
                    else:
                        print(f"Feature '{feature['name']}' rejected")
                
                if approved_for_table:
                    approved_features[table_id] = approved_for_table
                    print(f"Approved {len(approved_for_table)} features for table {table_id}")
                else:
                    print(f"No features approved for table {table_id}")
        
        finally:
            # Restore original interactive mode
            user_input_service.interactive = original_interactive
        
        return approved_features
    
    def _implement_features(self, approved_features: Dict[str, List[Dict[str, Any]]], 
                           input_tables: List[str], current_state: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate implementation code for approved features.
        
        Args:
            approved_features: Dictionary of approved features by table
            input_tables: List of input table IDs
            current_state: Current state of the system
            
        Returns:
            Dictionary of implemented features by table
        """
        logger.info("Implementing approved features")
        
        # Get the feature implementation agent
        meta_agent = self._get_meta_agent()
        feature_agent = meta_agent.feature_agent
        
        implemented_features = {}
        
        for table_id, features in approved_features.items():
            implemented_for_table = []
            
            # Get table schema
            table_info = current_state.get("tables", {}).get(table_id, {})
            schema = table_info.get("schema", {})
            
            if not schema:
                logger.warning(f"No schema found for table {table_id}")
                continue
            
            print(f"\n=== Implementing Features for Table: {table_id} ===")
            
            for feature in features:
                print(f"\nImplementing feature: {feature['name']}")
                
                try:
                    # Generate implementation code
                    implementation = feature_agent.generate_feature_code(
                        feature=feature,
                        schema=schema,
                        table_name=table_id
                    )
                    
                    # Add implementation to feature
                    feature['implementation'] = implementation
                    
                    # Display implementation
                    print(f"Implementation code generated:")
                    print(implementation['code'])
                    
                    # Add to implemented features
                    implemented_for_table.append(feature)
                    
                except Exception as e:
                    logger.error(f"Error implementing feature {feature['name']}: {str(e)}")
                    print(f"Error implementing feature: {str(e)}")
            
            if implemented_for_table:
                implemented_features[table_id] = implemented_for_table
                print(f"Implemented {len(implemented_for_table)} features for table {table_id}")
            else:
                print(f"No features implemented for table {table_id}")
        
        return implemented_features
    
    def _create_output_schema(self, implemented_features: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Create output schema for the implemented features.
        
        Args:
            implemented_features: Dictionary of implemented features by table
            
        Returns:
            Output schema dictionary
        """
        output_schema = {}
        
        for table_id, features in implemented_features.items():
            for feature in features:
                feature_name = feature['name']
                feature_type = feature.get('data_type', 'numeric')  # Default to numeric
                
                output_schema[feature_name] = {
                    "type": feature_type,
                    "source_table": table_id,
                    "feature_description": feature['description'],
                    "created_by": self.flow_id
                }
        
        return output_schema
    
    def _get_meta_agent(self):
        """Get the meta agent instance."""
        # First try to get the meta agent from the input data
        if hasattr(self, 'input_data') and self.input_data and 'meta_agent' in self.input_data:
            return self.input_data['meta_agent']
        
        # If not available, use the singleton pattern
        from meta_agent.meta_agent import MetaAgent
        return MetaAgent.get_instance() 