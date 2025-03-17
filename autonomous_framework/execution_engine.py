from typing import Dict, Any, Optional, Type
from sfn_blueprint import Task
from autonomous_framework.agent_registry import AgentRegistry
from autonomous_framework.context_manager import ContextManager
from autonomous_framework.meta_agent import MetaAgent

class ExecutionEngine:
    """
    Engine that executes agents based on Meta-Agent decisions
    """
    
    def __init__(self, agent_registry: AgentRegistry, context_manager: ContextManager):
        self.agent_registry = agent_registry
        self.context_manager = context_manager
        self.meta_agent = MetaAgent(agent_registry, context_manager)
        
    def run_step(self, goal=None, user_input=None):
        """Run a single step of the workflow
        
        Args:
            goal: Goal for the workflow (only needed for first step)
            user_input: User input for this step
            
        Returns:
            Dict containing step results
        """
        # Create task for meta agent
        meta_task = Task(
            description="Decide next agent to call",
            data={
                'goal': goal,
                'user_input': user_input or ''
            }
        )
        
        # Call meta agent to decide next agent
        meta_result = self.meta_agent.execute_task(meta_task)
        
        # If no next agent, return result
        if not meta_result.get('next_agent'):
            return {
                'agent': 'meta_agent',
                'reasoning': meta_result.get('reasoning', 'No next agent determined'),
                'output': {'error': 'No next agent determined'},
                'next_agent': None
            }
        
        # Get next agent
        next_agent_name = meta_result['next_agent']
        
        # Check if this would create a loop (same agent called repeatedly)
        # Get the last few steps from the context
        recent_steps = self.context_manager.get_recent_steps(3)
        if recent_steps and all(step == next_agent_name for step in recent_steps):
            print(f"Preventing loop: {next_agent_name} has been called repeatedly")
            # Force a different agent by excluding the current one
            alternative_agent = self._suggest_alternative_agent(next_agent_name)
            if alternative_agent:
                print(f"Suggesting alternative agent: {alternative_agent}")
                next_agent_name = alternative_agent
            else:
                return {
                    'agent': 'meta_agent',
                    'reasoning': 'Loop detected and no alternative agent available',
                    'output': {'error': 'Loop detected and no alternative agent available'},
                    'next_agent': None
                }
        
        # Get the agent
        next_agent = self.agent_registry.get_agent(next_agent_name)
        if not next_agent:
            return {
                'agent': 'meta_agent',
                'reasoning': meta_result.get('reasoning', ''),
                'output': {'error': f'Agent {next_agent_name} not found'},
                'next_agent': None
            }
        
        # Add context manager to inputs if the agent has a set_context_manager method
        if hasattr(next_agent, 'set_context_manager'):
            next_agent.set_context_manager(self.context_manager)
        
        # Resolve any DataFrame references in the inputs
        inputs = meta_result.get('inputs', {})
        resolved_inputs = {}
        
        for key, value in inputs.items():
            # Special handling for DataFrame references
            if key == 'df' and isinstance(value, str):
                # Try to get the DataFrame from the context manager
                df = self.context_manager.get_table(value)
                if df is not None:
                    resolved_inputs[key] = df
                else:
                    # If we can't find the DataFrame, return an error
                    return {
                        'agent': next_agent_name,
                        'reasoning': meta_result.get('reasoning', ''),
                        'output': {'error': f"Table '{value}' not found in context"},
                        'next_agent': None
                    }
            else:
                # For all other inputs, pass them through unchanged
                resolved_inputs[key] = value
        
        # Create task for next agent with resolved inputs
        agent_task = Task(
            description=f"Execute {next_agent_name} task",
            data=resolved_inputs
        )
        
        # Call next agent
        try:
            agent_result = next_agent.execute_task(agent_task)
            
            # Store the agent output in the context
            self.context_manager.store_agent_output(next_agent_name, agent_result)
            
            # Return step result
            return {
                'agent': next_agent_name,
                'reasoning': meta_result.get('reasoning', ''),
                'output': agent_result,
                'next_agent': meta_result.get('next_agent')
            }
        except Exception as e:
            # Handle agent execution error
            return {
                'agent': next_agent_name,
                'reasoning': meta_result.get('reasoning', ''),
                'output': {'error': str(e)},
                'next_agent': None
            }
    
    def _suggest_alternative_agent(self, current_agent):
        """Suggest an alternative agent to prevent loops"""
        # Get all available agents
        available_agents = self.agent_registry.list_agents()
        
        # Remove the current agent from the list
        if current_agent in available_agents:
            available_agents.remove(current_agent)
        
        # If we have other agents, suggest the next logical one based on workflow
        if not available_agents:
            return None
        
        # For churn prediction, suggest a logical next step
        if current_agent == "data_analyzer":
            # After data analysis, field mapping is often the next step
            if "field_mapper" in available_agents:
                return "field_mapper"
            # Or target generation
            elif "target_generator" in available_agents:
                return "target_generator"
        
        # Default: just return the first available agent
        return available_agents[0]
    
    def run_workflow(self, goal: str, max_steps: int = 10) -> Dict[str, Any]:
        """
        Run the entire workflow until completion or max steps
        
        Args:
            goal: The overall goal (e.g., "churn prediction")
            max_steps: Maximum number of steps to run
            
        Returns:
            Dict containing:
                - steps: List of steps that were executed
                - final_output: Output from the final agent
                - completed: Whether the workflow completed successfully
        """
        steps = []
        agent_counts = {}  # Track how many times each agent has been called
        
        for i in range(max_steps):
            step_result = self.run_step(goal if i == 0 else None)
            steps.append(step_result)
            
            # Track agent usage
            agent_name = step_result['agent']
            agent_counts[agent_name] = agent_counts.get(agent_name, 0) + 1
            
            # If no next agent or error, stop
            if not step_result.get('next_agent') or 'error' in step_result.get('output', {}):
                break
            
            # If the same agent has been called more than 3 times in a row, force a change
            if agent_counts.get(agent_name, 0) > 3 and step_result.get('next_agent') == agent_name:
                # Store the current agent's output in the context
                self.context_manager.store_agent_output(agent_name, step_result['output'])
                
                # Force a different agent selection
                print(f"Detected potential loop with {agent_name}. Forcing agent change.")
                break
        
        return {
            'steps': steps,
            'final_output': steps[-1]['output'] if steps else None,
            'completed': len(steps) > 0 and 'error' not in steps[-1].get('output', {})
        } 