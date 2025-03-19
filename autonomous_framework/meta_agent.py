from typing import Dict, Any, List, Optional
from sfn_blueprint import SFNAgent, Task, SFNAIHandler, SFNPromptManager
import os
from orchestrator.config.model_config import MODEL_CONFIG, DEFAULT_LLM_PROVIDER
from autonomous_framework.agent_registry import AgentRegistry
from autonomous_framework.context_manager import ContextManager
import json
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

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
        
        self.last_decided_agent = None
        self.executed_agents = set()
    
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
        context = self._build_context()
        
        # Get agent information for LLM
        agent_info = self.agent_registry.get_agent_info_for_llm()
        
        # Get agent decision
        next_agent_decision = self._decide_next_agent(goal, context, agent_info, user_input)
        
        # Update context with current step
        if next_agent_decision.get('next_agent'):
            self.context_manager.set_current_step(next_agent_decision['next_agent'])
        
        return next_agent_decision
    
    def _make_json_serializable(self, obj):
        """
        Recursively convert any DataFrame objects to serializable dictionaries
        
        Args:
            obj: Any Python object that might contain DataFrames
            
        Returns:
            JSON serializable version of the object
        """
        if isinstance(obj, pd.DataFrame):
            return {
                "shape": obj.shape,
                "columns": obj.columns.tolist(),
                "dtypes": {k: str(v) for k, v in obj.dtypes.items()},
                "data_sample": obj.head(5).to_dict(orient='records')
            }
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_json_serializable(item) for item in obj)
        elif isinstance(obj, set):
            return {self._make_json_serializable(item) for item in obj}
        else:
            return obj

    def _build_context(self) -> Dict[str, Any]:
        """Build context for the meta agent"""
        # Get current goal
        current_goal = self.context_manager.get_current_goal()
        
        # Get available tables directly from memory
        tables = self.context_manager.memory["tables"]
        
        # Get workflow state directly from memory
        workflow_state = self.context_manager.memory["workflow_state"]
        
        # Get agent outputs directly from memory
        agent_outputs = self.context_manager.memory["agent_outputs"]
        
        # Get processed agent outputs directly from memory
        processed_outputs = self.context_manager.memory["processed_agent_outputs"]
        
        # Get errors from workflow state
        errors = workflow_state.get("errors", [])
        
        # Get current and completed steps
        current_step = workflow_state.get("current_step")
        completed_steps = workflow_state.get("completed_steps", [])
        
        # Combine agent outputs and processed outputs
        completed_agent_outputs = {}
        for agent_name in agent_outputs:
            completed_agent_outputs[agent_name] = {
                "raw_output": agent_outputs[agent_name],
                "processed_output": processed_outputs.get(agent_name)
            }
        
        # Count how many times each agent has been called
        agent_call_counts = {}
        for step in completed_steps:
            agent_call_counts[step] = agent_call_counts.get(step, 0) + 1
        
        # Build context
        context = {
            "tables": self._make_json_serializable(tables),
            "workflow_state": self._make_json_serializable(workflow_state),
            "completed_agent_outputs": completed_agent_outputs,
            "agent_call_counts": agent_call_counts,
            "current_step": current_step,
            "completed_steps": completed_steps,
            "errors": errors,
            "goal": current_goal
        }
        
        return context
    
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
        # print(">>> META SYSTEM AGENT PROMPT:")
        # print(system_prompt)
        # print(">>> END META SYSTEM AGENT PROMPT")
        # print(">>> META USER AGENT PROMPT:")
        # print(user_prompt)
        # print(">>> END META USER AGENT PROMPT")
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

    def decide_next_agent(self, agent_output):
        """
        Decide which agent to call next based on the output of the current agent
        
        Args:
            agent_output: Output from the current agent
            
        Returns:
            Name of the next agent to call or None if workflow is complete
        """
        # Get the current workflow state
        current_goal = self.context_manager.get_current_goal()
        completed_steps = self.context_manager.get_recent_steps(10)  # Get more steps for context
        current_step = self.context_manager.get_current_step()
        
        # Check if there was an error in the current agent's output
        has_error = False
        if isinstance(agent_output, dict) and 'error' in agent_output:
            has_error = True
        
        # Create task for meta agent with enhanced context
        meta_task = Task(
            description="Decide next agent to call",
            data={
                'goal': current_goal,
                'user_input': '',
                'current_agent_output': agent_output,
                'current_step': current_step,
                'completed_steps': completed_steps,
                'has_error': has_error,
                'workflow_state': self.context_manager.get_workflow_summary()
            }
        )
        
        # Call execute_task to get the next agent decision
        next_agent_decision = self.execute_task(meta_task)
        
        # Log the decision for debugging
        logger.info(f"Meta agent decided next agent: {next_agent_decision.get('next_agent')}")
        logger.info(f"Reasoning: {next_agent_decision.get('reasoning')[:100]}...")
        
        # Return just the next agent name
        return next_agent_decision.get('next_agent')

    def mark_agent_executed(self, agent_name: str):
        """
        Mark an agent as executed to prevent re-deciding without execution
        
        Args:
            agent_name: Name of the agent that was executed
        """
        self.executed_agents.add(agent_name)
    
    def _get_last_decision(self) -> Dict[str, Any]:
        """
        Get the last agent decision that hasn't been executed yet
        
        Returns:
            The last decision dict
        """
        # This would need to be implemented to retrieve the last decision
        # For now, we'll return a simple dict
        return {
            "next_agent": self.last_decided_agent,
            "reasoning": "Continuing with previously decided agent",
            "inputs": {}
        }
    
    def _get_workflow_state(self) -> Dict[str, Any]:
        """
        Get the current state of the workflow
        
        Returns:
            Dict containing the current workflow state
        """
        # Get the workflow steps
        steps = self.context_manager.get_workflow_steps()
        
        # Get the agent outputs
        agent_outputs = {}
        for step in steps:
            agent_name = step.get("agent")
            if agent_name:
                agent_outputs[agent_name] = self.context_manager.get_agent_output(agent_name)
        
        # Get the available tables
        tables = list(self.context_manager.memory.get("tables", {}).keys())
        
        # Construct the workflow state
        workflow_state = {
            "steps": steps,
            "current_step": steps[-1] if steps else None,
            "agent_outputs": agent_outputs,
            "available_tables": tables
        }
        
        return workflow_state
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response into a structured decision
        
        Args:
            response: The raw LLM response
            
        Returns:
            Dict containing the parsed decision
        """
        try:
            # Try to parse as JSON
            # First, extract JSON if it's wrapped in markdown code blocks
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].strip()
            else:
                json_str = response.strip()
            
            decision = json.loads(json_str)
            return decision
        except Exception as e:
            logger.error(f"Error parsing meta agent response: {str(e)}")
            raise 