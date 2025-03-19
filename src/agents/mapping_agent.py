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
            Dictionary mapping input fields to standard field names
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
        - Keys are the input field names
        - Values are the standard field names they most closely match

        Example:
        {{
            "user_id": "ID",
            "item_name": "PRODUCT",
            "purchase_ts": "TIMESTAMP",
            "is_churned": "TARGET",
            "total_spent": "REVENUE"
        }}

        Only return the JSON mapping, no other text.
        """
        
        try:
            # Make the LLM call
            logger.info("Sending field mapping prompt to LLM")
            response = self.llm_agent.generate_text(prompt, max_tokens=500)
            print(f">> LLM response: {response}")  # temp
            
            # Parse the JSON response
            json_content = response
            if '{' in response and '}' in response:
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                json_content = response[start_idx:end_idx]
            
            mappings = json.loads(json_content)
            logger.info(f"Successfully parsed field mappings from LLM response: {mappings}")
            return mappings
            
        except Exception as e:
            logger.error(f"Error proposing field mappings: {str(e)}")
            logger.info("Using fallback mapping logic due to error")
            # Fallback mappings based on standard field names
            mappings = {}
            for field, field_type in schema.items():
                if "id" in field.lower():
                    mappings[field] = "ID"
                elif "product" in field.lower() or "item" in field.lower():
                    mappings[field] = "PRODUCT"
                elif "date" in field.lower() or "time" in field.lower():
                    mappings[field] = "TIMESTAMP"
                elif "churn" in field.lower() or "target" in field.lower():
                    mappings[field] = "TARGET"
                elif "amount" in field.lower() or "price" in field.lower() or "revenue" in field.lower():
                    mappings[field] = "REVENUE"
                elif field_type == "string":
                    mappings[field] = "CATEGORY"
                elif field_type == "numeric":
                    mappings[field] = "METRIC"
                else:
                    mappings[field] = "OTHER"
            return mappings 