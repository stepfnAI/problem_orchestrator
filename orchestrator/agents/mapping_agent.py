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
        mandatory_mapping = {
            'classification': ['id'], #'target'
            'regression': ['id'], #'target'
            'recommendation': ['product_id'],
            'clustering': ['id'],
            'forecasting': ['timestamp', 'target']
        }
        
        optional_mapping = {
            'classification': ['product_id', 'timestamp', 'revenue'],
            'regression': ['product_id', 'timestamp', 'revenue'],
            'recommendation': ['id', 'interaction_value', 'timestamp'],
            'clustering': ['product_id', 'timestamp', 'revenue'],
            'forecasting': ['product_id', 'id', 'revenue']
        }
        
        mandatory_fields = mandatory_mapping.get(problem_type, [])
        optional_fields = optional_mapping.get(problem_type, [])
        
        # Use AI to suggest mappings
        suggestions = self._get_ai_suggestions(columns, problem_type, mandatory_fields, optional_fields)
        
        # Validate and return mappings
        return self._validate_mappings(suggestions, mandatory_fields, optional_fields)

    def _get_ai_suggestions(self, columns: List[str], problem_type: str, mandatory_fields: List[str], optional_fields: List[str]) -> Dict[str, str]:
        """Get AI suggestions for mappings"""
        system_prompt, user_prompt = self.prompt_manager.get_prompt(
            agent_type='data_mapper',
            llm_provider=self.llm_provider,
            prompt_type='main',
            columns=columns,
            problem_type=problem_type,
            mandatory_fields=mandatory_fields,
            optional_fields=optional_fields
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
            # Return empty mappings for all fields
            return {field: None for field in mandatory_fields + optional_fields}

    def _validate_mappings(self, mappings: Dict[str, str], mandatory_fields: List[str], optional_fields: List[str]) -> Dict[str, str]:
        """Validate the mappings"""
        print(f">>>Mappings{mappings}")
        
        # # Check if all mandatory fields are mapped
        # for field in mandatory_fields:
        #     if field not in mappings or mappings[field] is None:
        #         print(f"Warning: Mandatory field '{field}' is not mapped")
        
        # # Create result dictionary with all fields (mandatory + optional)
        # result = {}
        # for field in mandatory_fields + optional_fields:
        #     result[field] = mappings.get(field)
        
        # print(f">>>Mappings after validation: {result}")
        return mappings
    
    def get_validation_params(self, response, task):
        """Get validation parameters"""
        return self.prompt_manager.get_prompt(
            agent_type='data_mapper',
            llm_provider=self.llm_provider,
            prompt_type='validation',
            actual_output=response
        ) 