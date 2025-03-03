from typing import Dict
import pandas as pd
from sfn_blueprint import SFNAgent, Task, SFNAIHandler, SFNPromptManager
import os
from recommendation_agent.config.model_config import MODEL_CONFIG, DEFAULT_LLM_PROVIDER
import json

class SFNApproachSelectionAgent(SFNAgent):
    """Agent responsible for suggesting recommendation approach"""
    
    def __init__(self, llm_provider=DEFAULT_LLM_PROVIDER):
        super().__init__(name="Approach Selector", role="Recommendation Strategy Analyst")
        self.ai_handler = SFNAIHandler()
        self.llm_provider = llm_provider
        self.model_config = MODEL_CONFIG["approach_selector"]
        
        parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        prompt_config_path = os.path.join(parent_path, 'config', 'prompt_config.json')
        self.prompt_manager = SFNPromptManager(prompt_config_path)

    def execute_task(self, task: Task) -> Dict:
        """Analyze data and suggest recommendation approach"""
        if not isinstance(task.data, dict):
            raise ValueError("Task data must be a dictionary")

        df = task.data.get('df')
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Task data must contain a DataFrame")
        print(">><<<df+++++++++++")
        # Analyze dataset characteristics
        analysis = self._analyze_dataset(df)
        print(">><<<analysis+++++++++++", analysis)
        # Get approach suggestion
        suggestion = self._get_approach_suggestion(analysis)
        print(">><<<suggestion+++++++++++", suggestion)
        if suggestion is None:
            return {}  # Return empty dict instead of None
        return suggestion

    def _analyze_dataset(self, df) -> Dict:
        """Analyze dataset characteristics relevant for approach selection"""
        analysis = {
            'user_count': df['cust_id'].nunique() if 'cust_id' in df.columns else 0,
            'item_count': df['product_id'].nunique(),
            'interaction_density': len(df) / (df['product_id'].nunique() * 
                                            (df['cust_id'].nunique() if 'cust_id' in df.columns else 1)),
            'has_user_features': 'cust_id' in df.columns,
            'has_item_features': True,  # Since we're working with SaaS products
            'avg_interactions_per_item': df['product_id'].value_counts().mean()
        }
        return analysis

    def _get_approach_suggestion(self, analysis) -> Dict:
        """Generate approach suggestion based on analysis"""
        try:
            # Format analysis text
            analysis_text = self._format_analysis_text(analysis)
            
            # Get prompt configuration - get both system and user prompts
            system_prompt, user_prompt = self.prompt_manager.get_prompt(
                agent_type='approach_selector',
                llm_provider=self.llm_provider,
                prompt_type='main',
                context=analysis_text
            )
            
            if not system_prompt or not user_prompt:
                print("Error: Could not get prompt configuration")
                return {}
            
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
            print(">><<<approach response", response)
            # Clean the content string
            cleaned_str = response.strip()
            cleaned_str = cleaned_str.replace('```json', '').replace('```', '')
            start_idx = cleaned_str.find('{')
            end_idx = cleaned_str.rfind('}')
            if start_idx != -1 and end_idx != -1:
                cleaned_str = cleaned_str[start_idx:end_idx + 1]
            
            # Parse LLM response to get structured suggestion
            parsed_response = json.loads(cleaned_str)
            return {
                'suggested_approach': parsed_response.get('approach'),
                'explanation': parsed_response.get('explanation'),
                'confidence': float(parsed_response.get('confidence', 0.0)),
                'analysis': analysis
            }
        except Exception as e:
            print(f"Error in approach suggestion: {str(e)}")
            return {}

    def _format_analysis_text(self, analysis) -> str:
        """Format analysis for prompt"""
        return "\n".join([
            f"Dataset Characteristics:",
            f"- Number of Users: {analysis['user_count']}",
            f"- Number of Products: {analysis['item_count']}",
            f"- Interaction Density: {analysis['interaction_density']:.2f}",
            f"- Has User Features: {analysis['has_user_features']}",
            f"- Has Item Features: {analysis['has_item_features']}",
            f"- Average Interactions per Item: {analysis['avg_interactions_per_item']:.2f}"
        ])

    def get_validation_params(self, response, task):
        """Get validation parameters"""
        return self.prompt_manager.get_prompt(
            agent_type='approach_selector',
            llm_provider=self.llm_provider,
            prompt_type='validation',
            actual_output=response
        ) 