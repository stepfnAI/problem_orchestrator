from typing import Dict, Any, List, Optional
from sfn_blueprint import SFNAgent, Task, SFNAIHandler, SFNPromptManager
import os
from orchestrator.config.model_config import MODEL_CONFIG, DEFAULT_LLM_PROVIDER
from autonomous_framework.agent_registry import AgentRegistry
from autonomous_framework.context_manager import ContextManager
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class MetaAgent(SFNAgent):
    """
    Meta-Agent that orchestrates the workflow by deciding which agent to call next
    """
    
    def __init__(self, agent_registry: AgentRegistry, context_manager: ContextManager):
        """Initialize the meta agent
        
        Args:
            agent_registry: Agent registry
            context_manager: Context manager
        """
        super().__init__(name="Meta Agent", role="Workflow Orchestrator")
        self.agent_registry = agent_registry
        self.context_manager = context_manager
        self.ai_handler = SFNAIHandler()
        
        # Get the correct path to the prompt config file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        prompt_config_path = os.path.join(project_root, 'orchestrator', 'config', 'prompt_config.json')
        
        # Initialize prompt manager with the correct path
        self.prompt_manager = SFNPromptManager(prompt_config_path)
        
        # Get model configuration
        self.llm_provider = DEFAULT_LLM_PROVIDER
        self.model_config = MODEL_CONFIG.get("meta_agent", {}).get(self.llm_provider, {})
    
    def execute_task(self, task: Task) -> Dict[str, Any]:
        """
        Decide which agent to call next based on the current context
        
        Args:
            task: Task object containing:
                - goal: The overall goal (e.g., "churn prediction")
                - current_state: Current state of the workflow
                - user_input: Any additional input from the user
                
        Returns:
            Dict containing:
                - next_agent: Name of the next agent to call
                - reasoning: Explanation of why this agent was chosen
                - inputs: Inputs to provide to the next agent
        """
        # Extract task data
        goal = task.data.get('goal', self.context_manager.get_current_goal())
        user_input = task.data.get('user_input', '')
        
        # If goal is not set, set it
        if not self.context_manager.get_current_goal() and goal:
            self.context_manager.set_current_goal(goal)
        
        # Get context for LLM
        context = self.context_manager.get_context_for_llm()
        
        # Get agent information for LLM
        agent_info = self.agent_registry.get_agent_info_for_llm()
        
        # Decide next agent
        next_agent_decision = self._decide_next_agent(goal, context, agent_info, user_input)
        
        # Update context with current step
        if next_agent_decision.get('next_agent'):
            self.context_manager.set_current_step(next_agent_decision['next_agent'])
        
        return next_agent_decision
    
    def _decide_next_agent(self, goal: str, context: Dict, agent_info: List[Dict], 
                          user_input: str) -> Dict[str, Any]:
        """
        Use LLM to decide which agent to call next
        """
        system_prompt, user_prompt = self.prompt_manager.get_prompt(
            agent_type='meta_agent',
            llm_provider=self.llm_provider,
            prompt_type='main',
            goal=goal,
            context=json.dumps(context, indent=2, cls=NumpyEncoder),
            agent_info=json.dumps(agent_info, indent=2, cls=NumpyEncoder),
            user_input=user_input
        )
        print(">>> META SYSTEM AGENT PROMPT:")
        print(system_prompt)
        print(">>> END META SYSTEM AGENT PROMPT")
        print(">>> META USER AGENT PROMPT:")
        print(user_prompt)
        print(">>> END META USER AGENT PROMPT")
        configuration = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.model_config["temperature"],
            "max_tokens": self.model_config["max_tokens"]
        }
        
        response, _ = self.ai_handler.route_to(
            llm_provider=self.llm_provider,
            configuration=configuration,
            model=self.model_config["model"]
        )
        print(">>> META AGENT RESPONSE:")
        print(response)
        print(">>> END META AGENT RESPONSE")
        return self._parse_response(response)
    
    def _parse_response(self, response) -> Dict[str, Any]:
        """Parse LLM response"""
        try:
            if isinstance(response, dict):
                content = response['choices'][0]['message']['content']
            else:
                content = response
                
            # Clean the content string
            cleaned_str = content.strip()
            cleaned_str = cleaned_str.replace('```json', '').replace('```', '')
            
            # Extract JSON content
            start_idx = cleaned_str.find('{')
            end_idx = cleaned_str.rfind('}')
            if start_idx != -1 and end_idx != -1:
                cleaned_str = cleaned_str[start_idx:end_idx + 1]
                
            # Parse JSON
            parsed = json.loads(cleaned_str)
            
            return {
                'next_agent': parsed.get('next_agent', ''),
                'reasoning': parsed.get('reasoning', ''),
                'inputs': parsed.get('inputs', {})
            }
            
        except Exception as e:
            print(f"Error parsing meta agent response: {str(e)}")
            return {
                'next_agent': '',
                'reasoning': f'Error parsing response: {str(e)}',
                'inputs': {}
            } 