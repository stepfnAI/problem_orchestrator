from typing import Dict, Any, Optional, Type
from sfn_blueprint import Task
from autonomous_framework.agent_registry import AgentRegistry
from autonomous_framework.context_manager import ContextManager
from autonomous_framework.meta_agent import MetaAgent
from autonomous_framework.agent_output_processor import AgentOutputProcessor
from autonomous_framework.agent_input_processor import AgentInputProcessor

class ExecutionEngine:
    """
    Engine that executes agents based on Meta-Agent decisions
    """
    
    def __init__(self, agent_registry: AgentRegistry, context_manager: ContextManager):
        self.agent_registry = agent_registry
        self.context_manager = context_manager
        self.meta_agent = MetaAgent(agent_registry, context_manager)
        self.output_processor = AgentOutputProcessor(context_manager)
        self.input_processor = AgentInputProcessor(context_manager)
        self.df = None  # Add dataframe storage at engine level
        
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
        if len(recent_steps) >= 2 and all(step == next_agent_name for step in recent_steps[-2:]):
            # If the same agent would be called 3 times in a row, this might be a loop
            return {
                'agent': 'meta_agent',
                'reasoning': f"Detected potential loop with agent {next_agent_name}",
                'output': {'error': f"Potential loop detected with agent {next_agent_name}"},
                'next_agent': None
            }
        
        # Get agent instance
        agent = self.agent_registry.get_agent(next_agent_name)
        if not agent:
            return {
                'agent': 'meta_agent',
                'reasoning': meta_result.get('reasoning', ''),
                'output': {'error': f"Agent {next_agent_name} not found"},
                'next_agent': None
            }
        
        # Process inputs for the agent
        processed_inputs = self.input_processor.process(next_agent_name, meta_result.get('inputs', {}))
        
        # Create task for agent
        agent_task = Task(
            description=f"Execute {next_agent_name}",
            data=processed_inputs
        )
        
        # Call agent
        try:
            agent_output = agent.execute_task(agent_task)
            
            # Process the agent output based on its type
            current_df = self.context_manager.get_current_dataframe()
            processed_output = self.output_processor.process(
                next_agent_name, 
                agent_output,
                current_df
            )
            
            # Store both raw and processed outputs
            self.context_manager.store_agent_output(next_agent_name, agent_output)
            if processed_output != agent_output:
                self.context_manager.store_processed_output(next_agent_name, processed_output)
            
            # Update current dataframe if it was modified
            if isinstance(processed_output, dict) and processed_output.get("success") and "processed_data" in processed_output:
                self.context_manager.update_dataframe(processed_output["processed_data"])
                
            # Store step in context
            self.context_manager.add_step_to_workflow(next_agent_name)
            
            # Mark the agent as executed in the meta agent
            self.meta_agent.mark_agent_executed(next_agent_name)
            
            # Return result
            return {
                'agent': next_agent_name,
                'reasoning': meta_result.get('reasoning', ''),
                'output': agent_output,
                'processed_output': processed_output if processed_output != agent_output else None,
                'next_agent': self.meta_agent.decide_next_agent(agent_output)
            }
        except Exception as e:
            # Handle error
            error_message = f"Error executing agent {next_agent_name}: {str(e)}"
            print(error_message)
            
            # Store error in context
            self.context_manager.add_error(error_message)
            
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

    def execute_agent_output(self, agent_name: str, agent_output: Dict):
        """Execute the output from agents that generate code"""
        if agent_name == "target_generator":
            return self._execute_target_generation(agent_output)
        # Add other agent output executions as needed
        return None

    def _execute_target_generation(self, agent_output: Dict) -> Dict:
        """Execute target generation code and return results"""
        try:
            # Get current dataframe
            df = self.context_manager.get_current_dataframe()
            
            # Execute the generated code
            import pandas as pd
            import numpy as np
            
            # Create a copy of df to avoid modifying original
            df_copy = df.copy()
            
            # Execute the code with proper context
            exec_locals = {
                'df': df_copy,
                'pd': pd,
                'np': np
            }
            
            # Execute the code
            exec(agent_output['code'], {}, exec_locals)
            
            # Get the resulting dataframe
            df_with_target = exec_locals['df']
            
            # Verify target column exists
            if 'target' not in df_with_target.columns:
                raise ValueError("Target column not created by the code")

            # Store the updated dataframe
            self.context_manager.update_dataframe(df_with_target)
            
            # Return execution results
            return {
                "success": True,
                "target_preview": df_with_target[['target']].head().to_dict(),
                "target_stats": {
                    "distribution": df_with_target['target'].value_counts().to_dict(),
                    "null_count": df_with_target['target'].isnull().sum()
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            } 