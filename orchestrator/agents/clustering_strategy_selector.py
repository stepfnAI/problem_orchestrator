from typing import Dict
from sfn_blueprint import SFNAgent, Task, SFNAIHandler, SFNPromptManager
import os
from orchestrator.config.model_config import MODEL_CONFIG, DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER
import json

class SFNClusterSelectionAgent(SFNAgent):
    """Agent responsible for selecting the best clustering model"""
    
    def __init__(self, llm_provider=DEFAULT_LLM_PROVIDER):
        super().__init__(name="Cluster Model Selector", role="Data Scientist")
        self.ai_handler = SFNAIHandler()
        self.llm_provider = llm_provider
        self.model_config = MODEL_CONFIG["clustering_strategy_selector"]
        parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        prompt_config_path = os.path.join(parent_path, 'config', 'prompt_config.json')
        self.prompt_manager = SFNPromptManager(prompt_config_path)

    def execute_task(self, task: Task) -> Dict:
        """Select best clustering model based on performance metrics"""
        if not isinstance(task.data, dict):
            raise ValueError("Task data must be a dictionary")
            
        clustering_results = task.data.get('clustering_results')
        if not clustering_results:
            raise ValueError("No clustering results provided")
            
        # Get model recommendation from LLM
        recommendation = self._get_model_recommendation(
            clustering_results,
            task.data.get('custom_instructions', '')
        )
        
        return recommendation

    def _get_model_recommendation(self, clustering_results: Dict, custom_instructions: str) -> Dict:
        """Get model recommendation from LLM"""
        # Prepare prompt parameters
        prompt_kwargs = {
            'clustering_results': json.dumps(clustering_results, indent=2),
            'custom_instructions': custom_instructions
        }
        
        # Get response from LLM
        system_prompt, user_prompt = self.prompt_manager.get_prompt(
            agent_type='clustering_strategy_selector',
            llm_provider=self.llm_provider,
            prompt_type='main',
            **prompt_kwargs
        )
        
        provider_config = self.model_config.get(self.llm_provider)
        configuration = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": provider_config["temperature"],
            "max_tokens": provider_config["max_tokens"]
        }
        
        response, _ = self.ai_handler.route_to(
            llm_provider=self.llm_provider,
            configuration=configuration,
            model=provider_config['model']
        )
        
        return self._parse_response(response)

    def _parse_response(self, response) -> Dict:
        """Parse LLM response into structured recommendation"""
        try:
            if isinstance(response, dict):
                content = response['choices'][0]['message']['content']
            else:
                content = response
                
            # Clean the response
            cleaned_str = content.strip()
            cleaned_str = cleaned_str.replace('```json', '').replace('```', '')
            start_idx = cleaned_str.find('{')
            end_idx = cleaned_str.rfind('}')
            if start_idx != -1 and end_idx != -1:
                cleaned_str = cleaned_str[start_idx:end_idx + 1]
                
            recommendation = json.loads(cleaned_str)
            return {
                'selected_model': recommendation.get('selected_model'),
                'explanation': recommendation.get('explanation'),
                'comparison_summary': recommendation.get('comparison_summary'),
                'model_rankings': recommendation.get('model_rankings', [])
            }
            
        except Exception as e:
            print(f"Error parsing recommendation: {str(e)}")
            return {
                'selected_model': 'kmeans',  # Default to kmeans
                'explanation': f'Error parsing recommendation: {str(e)}',
                'comparison_summary': '',
                'model_rankings': []
            }

    def get_validation_params(self, response, task):
        """Get parameters for validation"""
        prompts = self.prompt_manager.get_prompt(
            agent_type='clustering_strategy_selector',
            llm_provider=self.llm_provider,
            prompt_type='validation',
            actual_output=response
        )
        return prompts 