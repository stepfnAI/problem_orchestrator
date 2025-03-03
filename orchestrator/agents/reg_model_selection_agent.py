from typing import Dict
from sfn_blueprint import SFNAgent, Task, SFNAIHandler, SFNPromptManager
import os
from orchestrator.config.model_config import MODEL_CONFIG, DEFAULT_LLM_MODEL
from orchestrator.utils.reg_model_manager import ModelManager
import json

class SFNModelSelectionAgent(SFNAgent):
    """Agent responsible for selecting the best model based on performance metrics"""
    
    def __init__(self, llm_provider='openai', analysis_type="regression"):
        super().__init__(name="Model Selection", role="ML Engineer")
        self.ai_handler = SFNAIHandler()
        self.llm_provider = llm_provider
        self.analysis_type = analysis_type
        self.model_config = MODEL_CONFIG["model_selector"]
        parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        prompt_config_path = os.path.join(parent_path, 'config', 'prompt_config.json')
        self.prompt_manager = SFNPromptManager(prompt_config_path)
        self.model_manager = ModelManager()

    def execute_task(self, task: Task) -> Dict:
        """Select best model based on performance metrics"""
        if not isinstance(task.data, dict):
            raise ValueError("Task data must be a dictionary")
            
        selection_info = task.data.get('selection_info')
        if not selection_info:
            raise ValueError("No model results provided for selection")
            
        # Get metrics guide based on analysis type
        metrics_guide = self._get_metrics_guide()
        
        # Prepare prompt parameters
        prompt_params = {
            "analysis_type": self.analysis_type,
            "selection_info": selection_info,
            "metrics_guide": metrics_guide,
            "custom_instructions": task.data.get('custom_instructions', '')
        }
        
        # Get response from LLM
        system_prompt, user_prompt = self.prompt_manager.get_prompt(
            agent_type='model_selector',
            llm_provider=self.llm_provider,
            prompt_type='main',
            **prompt_params
        )
        
        response = self._get_llm_response(system_prompt, user_prompt)
        return self._parse_llm_response(response)
        
    def _get_metrics_guide(self) -> str:
        """Get appropriate metrics guide based on analysis type"""
        if self.analysis_type == "forecasting":
            return (
                "For forecasting models:\n"
                "- MAPE (Mean Absolute Percentage Error): Lower is better\n"
                "- RMSE (Root Mean Square Error): Lower is better\n"
                "- MAE (Mean Absolute Error): Lower is better\n"
                "Consider:\n"
                "- MAPE for relative error across different scales\n"
                "- RMSE for penalizing larger errors\n"
                "- MAE for absolute error magnitude"
            )
        else:  # regression
            return (
                "For regression models:\n"
                "- R² Score: Higher is better (1.0 is perfect)\n"
                "- MSE (Mean Squared Error): Lower is better\n"
                "- MAE (Mean Absolute Error): Lower is better\n"
                "Consider:\n"
                "- R² for overall fit quality\n"
                "- MSE for penalizing larger errors\n"
                "- MAE for absolute error magnitude"
            )

    def _get_llm_response(self, system_prompt: str, user_prompt: str):
        """Get response from LLM"""
        provider_config = self.model_config.get(self.llm_provider, {
            "model": DEFAULT_LLM_MODEL,
            "temperature": 0.3,
            "max_tokens": 1000,
            "n": 1,
            "stop": None
        })
        
        configuration = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": provider_config["temperature"],
            "max_tokens": provider_config["max_tokens"],
            "n": provider_config["n"],
            "stop": provider_config["stop"]
        }

        response, _ = self.ai_handler.route_to(
            llm_provider=self.llm_provider,
            configuration=configuration,
            model=provider_config['model']
        )
        
        return response

    def _parse_llm_response(self, response) -> Dict:
        """Parse LLM response into structured recommendation"""
        try:
            if isinstance(response, dict):
                content = response['choices'][0]['message']['content']
            else:
                content = response
            
            # Clean response by removing markdown code blocks and 'json' word
            if "```" in content:
                # Extract content between code block markers
                parts = content.split("```")
                for part in parts:
                    if "{" in part and "}" in part:
                        # Remove "json" if it's at the start
                        if part.startswith("json"):
                            part = part[4:]
                        content = part.strip()
                        break
            
            # Parse the cleaned JSON content
            recommendation = json.loads(content)  # Using json.loads instead of eval for safety
            return recommendation
            
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            print(f"Raw content:\n{content}")
            raise ValueError("Failed to parse LLM response") 

    def get_validation_params(self, response, task):
        """Get parameters for validation
        
        :param response: The recommendation response to validate
        :param task: Task object containing the model results
        :return: Dictionary with validation parameters
        """
        if not isinstance(task.data, dict):
            raise ValueError("Task data must be a dictionary")

        # Get validation prompts from prompt manager
        prompts = self.prompt_manager.get_prompt(
            agent_type='model_selector',
            llm_provider=self.llm_provider,
            prompt_type='validation',
            actual_output=response
        )
        return prompts 