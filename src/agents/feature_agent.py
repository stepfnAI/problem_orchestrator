"""
Feature Agent - Specialized agent for feature engineering tasks.
"""

from typing import Dict, List, Any
import logging
import json

from .specialized_agent import SpecializedAgent
from .model_config import ModelConfig
from .llm_agent import LLMAgent

logger = logging.getLogger(__name__)

class FeatureAgent(SpecializedAgent):
    """
    Specialized agent for feature engineering tasks.
    
    This agent suggests and evaluates features based on the dataset schema
    and problem type.
    """
    
    def __init__(self, model_name: str = "default"):
        """
        Initialize the Feature Agent.
        
        Args:
            model_name: Name of the model to use
        """
        super().__init__(model_name)
        self.llm_agent = LLMAgent(model_name=model_name)
    
    def suggest_features(self, schema: Dict[str, Any], problem_type: str) -> List[Dict[str, Any]]:
        """
        Suggest features based on the schema and problem type.
        
        Args:
            schema: Dictionary mapping field names to their metadata
            problem_type: Type of problem (e.g., regression, classification)
            
        Returns:
            List of suggested features
        """
        logger.info(f"Suggesting features for {problem_type} problem with {len(schema)} fields")
        
        # Create a prompt for feature suggestions
        schema_str = json.dumps(schema, indent=2)
        
        prompt = f"""
        Given the following schema and problem type, suggest features that might be useful.
        
        Problem Type: {problem_type}
        
        Schema:
        {schema_str}
        
        Please suggest features that could be derived from the existing fields.
        For each feature, provide:
        1. A name
        2. A description
        3. The formula or logic to create it
        4. An importance score (1-10)
        
        Format your response as a JSON array of feature objects:
        [
            {{
                "name": "feature_name",
                "description": "Feature description",
                "formula": "SQL or pseudocode formula",
                "importance": 8
            }},
            ...
        ]
        
        Only include the JSON array in your response, no additional text.
        """
        
        try:
            # Make the LLM call
            response = self.llm_agent.generate_text(prompt, max_tokens=1000)
            
            # Parse the JSON response
            # Find JSON content between square brackets
            json_content = response
            if '[' in response and ']' in response:
                start_idx = response.find('[')
                end_idx = response.rfind(']') + 1
                json_content = response[start_idx:end_idx]
            
            features = json.loads(json_content)
            return features
        except Exception as e:
            logger.error(f"Error suggesting features: {str(e)}")
            # Fallback feature suggestions
            features = []
            
            # Generate some basic feature suggestions based on the schema
            for field, info in schema.items():
                field_type = info.get("type", "")
                mapping = info.get("mapping", "")
                
                if mapping == "date":
                    # Date-based features
                    features.append({
                        "name": f"month_{field}",
                        "description": f"Month extracted from {field}",
                        "formula": f"EXTRACT(MONTH FROM {field})",
                        "importance": 8
                    })
                    
                    features.append({
                        "name": f"day_of_week_{field}",
                        "description": f"Day of week extracted from {field}",
                        "formula": f"EXTRACT(DOW FROM {field})",
                        "importance": 7
                    })
                
                elif mapping == "categorical":
                    # Categorical features
                    features.append({
                        "name": f"encoded_{field}",
                        "description": f"One-hot encoded version of {field}",
                        "formula": f"ONE_HOT_ENCODE({field})",
                        "importance": 8
                    })
                
                elif mapping == "numerical":
                    # Numerical features
                    features.append({
                        "name": f"normalized_{field}",
                        "description": f"Normalized version of {field}",
                        "formula": f"({field} - MIN({field})) / (MAX({field}) - MIN({field}))",
                        "importance": 6
                    })
            
            return features
    
    def evaluate_features(self, schema: Dict[str, str], problem_type: str, 
                         features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate proposed features and rank them by importance.
        
        Args:
            schema: Dictionary mapping field names to their types
            problem_type: Type of problem (regression, classification, etc.)
            features: List of feature dictionaries to evaluate
            
        Returns:
            List of evaluated features with importance scores
        """
        # Format the schema for the prompt
        schema_str = "\n".join([f"{field}: {field_type}" for field, field_type in schema.items()])
        
        # Format the features for the prompt
        features_str = "\n".join([
            f"{i+1}. {feature['name']}: {feature['description']} - {feature['formula']}"
            for i, feature in enumerate(features)
        ])
        
        # Get the formatted prompt
        prompt = self.get_prompt("feature_evaluation", 
                                schema=schema_str,
                                problem_type=problem_type,
                                features=features_str)
        
        if not prompt:
            logger.warning("Feature evaluation prompt not found, using fallback logic")
            return features  # Just return the original features as fallback
        
        # Get model config for this task
        config = ModelConfig.get_config(self.model_name, "feature_evaluation")
        
        # In a real implementation, this would call the LLM with the prompt
        # For now, just return the features with some dummy scores
        for feature in features:
            if "importance" not in feature:
                feature["importance"] = 5  # Default importance
        
        # Sort by importance (descending)
        return sorted(features, key=lambda x: x.get("importance", 0), reverse=True) 