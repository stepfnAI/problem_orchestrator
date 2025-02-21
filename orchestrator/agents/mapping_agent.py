from typing import Dict, List
import pandas as pd
from sfn_blueprint import SFNAgent, Task, SFNAIHandler, SFNPromptManager
import os
from orchestrator.config.model_config import MODEL_CONFIG, DEFAULT_LLM_PROVIDER
import json

class SFNDataMappingAgent(SFNAgent):
    """Agent responsible for mapping columns based on problem type"""
    
    def __init__(self):
        super().__init__(name="Field Mapper", role="Data Analyst")
        self.ai_handler = SFNAIHandler()
        self.llm_provider = DEFAULT_LLM_PROVIDER
        self.model_config = MODEL_CONFIG["data_mapper"]
        
        # Initialize prompt manager
        parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        prompt_config_path = os.path.join(parent_path, 'config', 'prompt_config.json')
        self.prompt_manager = SFNPromptManager(prompt_config_path)
        
    def execute_task(self, task: Task) -> Dict[str, str]:
        """Maps dataset columns to required fields based on problem type"""
        # Extract data from task
        task_data = task.data
        df = task_data['df']
        problem_type = task_data['problem_type']
        
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Task data must contain a pandas DataFrame")

        columns = df.columns.tolist()
        return self._identify_fields(columns, problem_type)

    def _identify_fields(self, columns: List[str], problem_type: str) -> Dict[str, str]:
        """Identify fields based on problem type"""
        # Configure mapping based on problem type
        mapping_config = {
            'classification': ['id', 'target'],
            'regression': ['id', 'target'],
            'recommendation': ['product_id', 'customer_id', 'interaction_value'],
            'clustering': ['features'],
            'forecasting': ['timestamp', 'target']
        }
        
        required_fields = mapping_config.get(problem_type, [])
        
        # Use AI to suggest mappings
        suggestions = self._get_ai_suggestions(columns, required_fields)
        
        # Validate and return mappings
        return self._validate_mappings(suggestions, required_fields)

    def _get_ai_suggestions(self, columns: List[str], required_fields: List[str]) -> Dict[str, str]:
        """Get AI suggestions for mappings"""
        system_prompt, user_prompt = self.prompt_manager.get_prompt(
            agent_type='data_mapper',
            llm_provider=self.llm_provider,
            prompt_type='main',
            columns=columns,
            required_fields=required_fields
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
            
            return json.loads(cleaned_str)
        except Exception as e:
            print(f"Error parsing AI response: {str(e)}")
            return {field: None for field in required_fields}

    def _validate_mappings(self, mappings: Dict[str, str], required_fields: List[str]) -> Dict[str, str]:
        """Validate the mappings"""
        return {field: mappings.get(field) for field in required_fields}

    def get_validation_params(self, response, task):
        """Get validation parameters"""
        return self.prompt_manager.get_prompt(
            agent_type='data_mapper',
            llm_provider=self.llm_provider,
            prompt_type='validation',
            actual_output=response
        ) 