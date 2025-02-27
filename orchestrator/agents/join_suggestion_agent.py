from typing import Dict, List, Any
import pandas as pd
from sfn_blueprint import SFNAgent, Task, SFNAIHandler, SFNPromptManager
import os
from orchestrator.config.model_config import MODEL_CONFIG, DEFAULT_LLM_PROVIDER
import json

class SFNJoinSuggestionAgent(SFNAgent):
    """Agent responsible for suggesting the next best join between available tables"""
    
    def __init__(self):
        super().__init__(name="Join Suggester", role="Data Engineer")
        self.ai_handler = SFNAIHandler()
        self.llm_provider = DEFAULT_LLM_PROVIDER
        self.model_config = MODEL_CONFIG["join_suggester"]
        
        # Initialize prompt manager
        parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        prompt_config_path = os.path.join(parent_path, 'config', 'prompt_config.json')
        self.prompt_manager = SFNPromptManager(prompt_config_path)
        
    def execute_task(self, task: Task) -> Dict[str, Any]:
        """Suggests the next best join between available tables"""
        # Extract data from task
        task_data = task.data
        available_tables = task_data.get('available_tables', [])
        tables_metadata = task_data.get('tables_metadata', [])
        other_info = task_data.get('other_info', '')
        
        if not available_tables or len(available_tables) < 2:
            raise ValueError("At least two tables are required for join suggestion")
        
        if not tables_metadata or len(tables_metadata) != len(available_tables):
            raise ValueError("Metadata must be provided for all available tables")
        
        return self._suggest_join(available_tables, tables_metadata, other_info)

    def _suggest_join(self, available_tables: List[str], tables_metadata: List[Dict], other_info: str) -> Dict[str, Any]:
        """Suggest the next best join between available tables"""
        # Use AI to suggest the next join
        suggestions = self._get_ai_suggestions(available_tables, tables_metadata, other_info)
        
        # Validate and return join suggestions
        return self._validate_suggestions(suggestions, available_tables)

    def _get_ai_suggestions(self, available_tables: List[str], tables_metadata: List[Dict], other_info: str) -> Dict[str, Any]:
        """Get AI suggestions for the next join"""
        system_prompt, user_prompt = self.prompt_manager.get_prompt(
            agent_type='join_suggester',
            llm_provider=self.llm_provider,
            prompt_type='main',
            available_tables=available_tables,
            tables_metadata=json.dumps(tables_metadata, indent=2),
            other_info=other_info
        )

        configuration = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.model_config[self.llm_provider]["temperature"],
            "max_tokens": self.model_config[self.llm_provider]["max_tokens"]
        }

        response, _ = self.ai_handler.route_to(
            llm_provider=self.llm_provider,
            configuration=configuration,
            model=self.model_config[self.llm_provider]["model"]
        )
        
        try:
            if isinstance(response, dict):
                content = response['choices'][0]['message']['content']
            elif hasattr(response, 'choices'):
                content = response.choices[0].message.content
            else:
                content = response
                
            # Parse JSON response
            cleaned_str = content.strip()
            cleaned_str = cleaned_str.replace('```json', '').replace('```', '')
            start_idx = cleaned_str.find('{')
            end_idx = cleaned_str.rfind('}')
            if start_idx != -1 and end_idx != -1:
                cleaned_str = cleaned_str[start_idx:end_idx + 1]
            print(f">>>Cleaned string: {cleaned_str}")
            return json.loads(cleaned_str)
        except Exception as e:
            print(f"Error parsing AI response: {str(e)}")
            # Return empty suggestions
            return {
                "tables_to_join": [],
                "type_of_join": "",
                "joining_fields": [],
                "explanation": "Failed to generate join suggestions"
            }

    def _validate_suggestions(self, suggestions: Dict[str, Any], available_tables: List[str]) -> Dict[str, Any]:
        """Validate the join suggestions"""
        print(f">>>Suggestions before validation: {suggestions}")
        
        # Check if suggested tables are in available tables
        tables_to_join = suggestions.get("tables_to_join", [])
        if not isinstance(tables_to_join, list):
            tables_to_join = [tables_to_join]
            suggestions["tables_to_join"] = tables_to_join
            
        for table in tables_to_join:
            if table not in available_tables:
                print(f"Warning: Suggested table '{table}' is not in available tables")
        
        # Check if join type is valid
        join_type = suggestions.get("type_of_join", "")
        valid_join_types = ["inner", "left", "right", "outer"]
        if join_type.lower() not in valid_join_types:
            print(f"Warning: Invalid join type '{join_type}'")
            suggestions["type_of_join"] = "inner"  # Default to inner join
        
        # Check if joining fields are provided
        joining_fields = suggestions.get("joining_fields", [])
        if not joining_fields:
            print("Warning: No joining fields provided")
        
        print(f">>>Suggestions after validation: {suggestions}")
        return suggestions
    
    def get_validation_params(self, response, task):
        """Get validation parameters"""
        available_tables = task.data.get('available_tables', [])
        return self.prompt_manager.get_prompt(
            agent_type='join_suggester',
            llm_provider=self.llm_provider,
            prompt_type='validation',
            actual_output=json.dumps(response, indent=2),
            available_tables=available_tables
        )
