"""
Feature Agent - Suggests and implements features for data science problems.
"""

from typing import Dict, List, Any
import logging
import json

from .specialized_agent import SpecializedAgent
from .llm_agent import LLMAgent

logger = logging.getLogger(__name__)

class FeatureAgent(SpecializedAgent):
    """
    Agent for suggesting and implementing features for data science problems.
    
    This agent analyzes data schemas and problem types to suggest
    relevant features, and generates code to implement those features.
    """
    
    def __init__(self, model_name: str = "default"):
        """
        Initialize the Feature Agent.
        
        Args:
            model_name: Name of the model to use
        """
        super().__init__(model_name)
        self.llm_agent = LLMAgent(model_name=model_name)
        self.agent_type = "feature_agent"
    
    def suggest_features(self, schema: Dict[str, Any], problem_type: str, 
                        problem_description: str = "") -> List[Dict[str, Any]]:
        """
        Suggest features based on the schema and problem type.
        
        Args:
            schema: Schema of the data
            problem_type: Type of problem (regression, classification, etc.)
            problem_description: Description of the problem
            
        Returns:
            List of suggested features
        """
        logger.info("Suggesting features")
        
        # Create prompt for LLM
        prompt = f"""
        You are an expert feature engineer for data science problems.
        
        Given the following data schema and problem type, suggest relevant features that would improve model performance.
        
        Schema:
        {json.dumps(schema, indent=2)}
        
        Problem Type: {problem_type}
        Problem Description: {problem_description}
        
        For each feature, provide:
        1. A descriptive name
        2. A clear description of what the feature represents
        3. The rationale for why this feature would be useful for the problem
        4. The complexity of implementing this feature (Low, Medium, High)
        5. The expected data type of the feature (numeric, categorical, boolean, etc.)
        
        Respond with a JSON array of feature suggestions, with each suggestion having the following structure:
        {{
            "name": "feature_name",
            "description": "Description of what this feature represents",
            "rationale": "Why this feature would be useful for the problem",
            "complexity": "Low|Medium|High",
            "data_type": "numeric|categorical|boolean|date",
            "input_fields": ["field1", "field2"]  // Fields used to create this feature
        }}
        
        Focus on features that are:
        - Relevant to the problem type
        - Diverse in nature (aggregations, ratios, time-based, etc.)
        - Implementable with the available data
        - Likely to have predictive power
        
        Suggest 3-5 high-quality features.
        """
        
        # Make the LLM call
        logger.info("Sending feature suggestion prompt to LLM")
        response = self.llm_agent.generate_text(prompt, max_tokens=1000)
        
        # Parse the JSON response
        try:
            # Extract JSON from response
            json_content = response
            if '[' in response and ']' in response:
                start_idx = response.find('[')
                end_idx = response.rfind(']') + 1
                json_content = response[start_idx:end_idx]
            
            suggestions = json.loads(json_content)
            logger.info(f"Successfully parsed {len(suggestions)} feature suggestions from LLM response")
            return suggestions
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            # Return empty list if parsing fails
            return []
    
    def generate_feature_code(self, feature: Dict[str, Any], schema: Dict[str, Any], 
                             table_name: str) -> Dict[str, Any]:
        """
        Generate code to implement a feature.
        
        Args:
            feature: Feature to implement
            schema: Schema of the data
            table_name: Name of the table
            
        Returns:
            Dictionary with implementation details
        """
        logger.info(f"Generating code for feature: {feature['name']}")
        
        # Create prompt for LLM
        prompt = f"""
        You are an expert data scientist who writes clean, efficient code to implement features.
        
        I need you to write Python code to implement the following feature:
        
        Feature Name: {feature['name']}
        Description: {feature['description']}
        Input Fields: {', '.join(feature.get('input_fields', []))}
        
        Data Schema:
        {json.dumps(schema, indent=2)}
        
        Table Name: {table_name}
        
        Write Python code that:
        1. Assumes the data is in a pandas DataFrame named 'df'
        2. Creates the feature and adds it as a new column to the DataFrame
        3. Handles potential edge cases (nulls, divisions by zero, etc.)
        4. Is efficient and readable
        
        Your response should be a JSON object with:
        1. "code": The Python code to implement the feature
        2. "explanation": A brief explanation of how the code works
        3. "dependencies": Any Python packages required beyond pandas and numpy
        
        Example response:
        {{
            "code": "df['price_per_unit'] = df['total_price'] / df['quantity'].replace(0, np.nan)",
            "explanation": "This creates a price per unit feature by dividing total price by quantity, replacing zeros with NaN to avoid division by zero errors.",
            "dependencies": []
        }}
        """
        
        # Make the LLM call
        logger.info("Sending feature implementation prompt to LLM")
        response = self.llm_agent.generate_text(prompt, max_tokens=800)
        
        # Parse the JSON response
        try:
            # Extract JSON from response
            json_content = response
            if '{' in response and '}' in response:
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                json_content = response[start_idx:end_idx]
            
            implementation = json.loads(json_content)
            logger.info(f"Successfully parsed feature implementation from LLM response")
            return implementation
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            # Return basic implementation if parsing fails
            return {
                "code": f"# Failed to generate code for {feature['name']}\n# Error: {str(e)}",
                "explanation": "Code generation failed",
                "dependencies": []
            } 