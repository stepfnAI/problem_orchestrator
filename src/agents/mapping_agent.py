"""
Mapping Agent - Maps input fields to standardized field names.
"""

from typing import Dict, List, Any
import logging
import json

from .specialized_agent import SpecializedAgent
from .llm_agent import LLMAgent

logger = logging.getLogger(__name__)

class MappingAgent(SpecializedAgent):
    """
    Specialized agent for mapping input fields to standardized field names.
    """
    
    def __init__(self, model_name: str = "default"):
        """Initialize the Mapping Agent."""
        super().__init__(model_name)
        self.llm_agent = LLMAgent(model_name=model_name)
    
    def propose_field_mappings(self, schema: Dict[str, str]) -> Dict[str, str]:
        """
        Map input fields to standard field names.
        
        Args:
            schema: Dictionary mapping field names to their types
            
        Returns:
            Dictionary mapping standard field names to input fields
        """
        logger.info(f"Proposing field mappings for {len(schema)} fields")
        
        # Create a prompt for field mappings
        schema_str = "\n".join([f"- {field}: {field_type}" for field, field_type in schema.items()])
        
        prompt = f"""
        Given the following input fields, map them to standard field names.

        Input Fields:
        {schema_str}

        Standard Field Categories:
        - ID: Any unique identifier fields
        - PRODUCT: Product or service related fields
        - TIMESTAMP: Date/time related fields
        - TARGET: Fields that could be prediction targets (e.g., churn, revenue)
        - REVENUE: Fields related to monetary values
        - CATEGORY: Categorical or descriptive fields
        - METRIC: Numerical measurement fields
        - OTHER: Fields that don't fit above categories

        Return a JSON mapping where:
        - Keys are the standard field names (from the categories above)
        - Values are the input field names they most closely match

        Example:
        {{
            "ID": "user_id",
            "PRODUCT": "item_name",
            "TIMESTAMP": "purchase_ts",
            "TARGET": "is_churned",
            "REVENUE": "total_spent"
        }}

        Notes:
        - Not every standard field category needs to be used
        - Only include standard categories that have a clear match in the input data
        - You can map multiple input fields to the same standard category if needed

        Only return the JSON mapping, no other text.
        """
        
        # Make the LLM call
        logger.info("Sending field mapping prompt to LLM")
        response = self.llm_agent.generate_text(prompt, max_tokens=500)
        
        # Parse the JSON response
        json_content = response
        if '{' in response and '}' in response:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            json_content = response[start_idx:end_idx]
        
        try:
            mappings = json.loads(json_content)
            logger.info(f"Successfully parsed field mappings from LLM response: {mappings}")
            return mappings
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            # Return empty mappings if parsing fails
            return {} 