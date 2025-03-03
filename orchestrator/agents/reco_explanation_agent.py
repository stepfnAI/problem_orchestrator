from typing import Dict, List
import pandas as pd
from sfn_blueprint import SFNAgent, Task, SFNAIHandler, SFNPromptManager
import os
from orchestrator.config.model_config import MODEL_CONFIG, DEFAULT_LLM_PROVIDER
import json

class SFNRecommendationExplanationAgent(SFNAgent):
    """Agent responsible for generating human-readable explanations for recommendations"""
    
    def __init__(self, llm_provider=DEFAULT_LLM_PROVIDER):
        super().__init__(name="Recommendation Explainer", role="Explanation Generator")
        self.ai_handler = SFNAIHandler()
        self.llm_provider = llm_provider
        self.model_config = MODEL_CONFIG["recommendation_explainer"]
        
        parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        prompt_config_path = os.path.join(parent_path, 'config', 'prompt_config.json')
        self.prompt_manager = SFNPromptManager(prompt_config_path)

    def execute_task(self, task: Task) -> Dict:
        """Generate explanations for recommendations"""
        if not isinstance(task.data, dict):
            raise ValueError("Task data must be a dictionary")

        required_keys = ['user_profile', 'recommendations', 'similar_users_info']
        if not all(key in task.data for key in required_keys):
            raise ValueError(f"Task data must contain: {required_keys}")

        return self._generate_explanations(
            task.data['user_profile'],
            task.data['recommendations'],
            task.data['similar_users_info']
        )

    def _generate_explanations(self, user_profile: Dict, 
                             recommendations: List[Dict],
                             similar_users_info: Dict) -> Dict:
        """Generate explanations using LLM"""
        system_prompt, user_prompt = self.prompt_manager.get_prompt(
            agent_type='recommendation_explainer',
            llm_provider=self.llm_provider,
            prompt_type='main',
            user_profile=user_profile,
            recommendations=recommendations,
            similar_users_info=similar_users_info
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
        
        return self._parse_response(response)

    def _parse_response(self, response) -> Dict:
        """Parse LLM response"""
        try:
            if isinstance(response, dict):
                content = response['choices'][0]['message']['content']
            else:
                content = response
            return json.loads(content)
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            return {"recommendations": []}

    def get_validation_params(self, response, task):
        """Get validation parameters"""
        return self.prompt_manager.get_prompt(
            agent_type='recommendation_explainer',
            llm_provider=self.llm_provider,
            prompt_type='validation',
            actual_output=response
        ) 