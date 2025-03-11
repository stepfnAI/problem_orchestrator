from typing import Dict, List, Any
import pandas as pd
from sfn_blueprint import SFNAgent, Task, SFNAIHandler, SFNPromptManager
import os
import json
from orchestrator.config.model_config import MODEL_CONFIG, DEFAULT_LLM_PROVIDER

class SFNTargetGeneratorAgent(SFNAgent):
    """Agent responsible for generating code to create target columns based on user instructions"""
    
    def __init__(self, llm_provider=DEFAULT_LLM_PROVIDER):
        super().__init__(name="Target Generator", role="Data Scientist")
        self.ai_handler = SFNAIHandler()
        self.llm_provider = llm_provider
        self.model_config = MODEL_CONFIG["target_generator"]
        
        # Initialize prompt manager
        parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        prompt_config_path = os.path.join(parent_path, 'config', 'prompt_config.json')
        self.prompt_manager = SFNPromptManager(prompt_config_path)
        
    def execute_task(self, task: Task) -> Dict[str, Any]:
        """Generate code to create a target column based on user instructions"""
        # Extract data from task
        if not isinstance(task.data, dict):
            raise ValueError("Task data must be a dictionary")
            
        # Extract required fields
        user_instructions = task.data.get('user_instructions', '')
        df = task.data.get('df')
        problem_type = task.data.get('problem_type', '')
        error_message = task.data.get('error_message', '')
        
        if not user_instructions:
            raise ValueError("User instructions are required")
            
        if not isinstance(df, pd.DataFrame):
            raise ValueError("DataFrame is required")
            
        if not problem_type:
            raise ValueError("Problem type is required")
            
        # Generate target column code
        return self._generate_target_code(
            user_instructions=user_instructions,
            df=df,
            problem_type=problem_type,
            error_message=error_message
        )
        
    def _generate_target_code(self, user_instructions: str, df: pd.DataFrame, 
                             problem_type: str, error_message: str = '') -> Dict[str, Any]:
        """Generate code to create a target column"""
        # Prepare dataframe information
        df_shape = df.shape
        df_columns = df.columns.tolist()
        df_sample = df.head(5).to_dict(orient='records')
        
        # Format sample data for prompt
        sample_str = json.dumps(df_sample, indent=2)
        
        # Prepare prompt parameters
        prompt_kwargs = {
            'user_instructions': user_instructions,
            'df_shape': str(df_shape),
            'df_columns': ', '.join(df_columns),
            'df_sample': sample_str,
            'problem_type': problem_type,
            'error_message': error_message
        }
        
        # Get prompt from prompt manager
        system_prompt, user_prompt = self.prompt_manager.get_prompt(
            agent_type='target_generator',
            llm_provider=self.llm_provider,
            prompt_type='main',
            **prompt_kwargs
        )
        
        # Configure LLM request
        provider_config = self.model_config.get(self.llm_provider)
        configuration = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": provider_config["temperature"],
            "max_tokens": provider_config["max_tokens"]
        }
        
        # Get response from LLM
        response, _ = self.ai_handler.route_to(
            llm_provider=self.llm_provider,
            configuration=configuration,
            model=provider_config['model']
        )
        
        # Parse and return the response
        return self._parse_response(response)
        
    def _parse_response(self, response) -> Dict[str, Any]:
        """Parse LLM response into structured output"""
        try:
            if isinstance(response, dict):
                content = response['choices'][0]['message']['content']
            else:
                content = response
                
            # Clean the response
            cleaned_str = content.strip()
            cleaned_str = cleaned_str.replace('```json', '').replace('```', '')
            
            # Extract JSON content
            start_idx = cleaned_str.find('{')
            end_idx = cleaned_str.rfind('}')
            if start_idx != -1 and end_idx != -1:
                cleaned_str = cleaned_str[start_idx:end_idx + 1]
                
            # Parse JSON
            parsed = json.loads(cleaned_str)
            
            # Extract code, explanation, and preview
            return {
                'code': parsed.get('code', ''),
                'explanation': parsed.get('explanation', ''),
                'preview': parsed.get('preview', '')
            }
            
        except Exception as e:
            print(f"Error parsing target generator response: {str(e)}")
            return {
                'code': '',
                'explanation': f'Error parsing response: {str(e)}',
                'preview': ''
            }
            
    def get_validation_params(self, response, task):
        """Get parameters for validation"""
        # Extract code from response
        code = response.get('code', '')
        user_instructions = task.data.get('user_instructions', '')
        problem_type = task.data.get('problem_type', '')
        
        # Get validation prompt
        prompts = self.prompt_manager.get_prompt(
            agent_type='target_generator',
            llm_provider=self.llm_provider,
            prompt_type='validation',
            code=code,
            user_instructions=user_instructions,
            problem_type=problem_type
        )
        
        return prompts 