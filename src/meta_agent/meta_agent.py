"""
Meta Agent - Central orchestrator for the problem-solving framework.
"""

from typing import Dict, List, Any, Optional
import logging
import json

from flow_manager.flow_manager import FlowManager
from services.state_manager import StateManager
from services.user_input_service import UserInputService
from services.data_lineage_manager import DataLineageManager
from agents.mapping_agent import MappingAgent
from agents.feature_agent import FeatureAgent
from agents.llm_agent import LLMAgent

logger = logging.getLogger(__name__)

class MetaAgent:
    """
    Central orchestrator that determines the sequence of specialized flows
    needed to solve a problem.
    """
    
    # Singleton instance
    _instance = None
    
    @classmethod
    def get_instance(cls, config=None):
        """
        Get the singleton instance of the MetaAgent.
        
        Args:
            config: Configuration dictionary for the Meta Agent (only used if creating a new instance)
            
        Returns:
            Singleton instance of the MetaAgent
        """
        if cls._instance is None:
            if config is None:
                config = {"model_name": "default"}
            cls._instance = cls(config)
        return cls._instance
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Meta Agent.
        
        Args:
            config: Configuration dictionary for the Meta Agent
        """
        self.config = config
        self.flow_manager = FlowManager()
        
        # Use the singleton instance of StateManager
        self.state_manager = StateManager.get_instance()
        
        # Check if interactive mode is enabled
        interactive = config.get("interactive", True)
        self.user_input_service = UserInputService(interactive=interactive)
        
        self.data_lineage_manager = DataLineageManager()
        self.problem_context = {}
        
        # Initialize agents
        model_name = config.get("model_name", "default")
        self.mapping_agent = MappingAgent(model_name=model_name)
        self.feature_agent = FeatureAgent(model_name=model_name)
        
        # Set this instance as the singleton instance
        MetaAgent._instance = self
        
    def initialize_problem(self, problem_description: str) -> None:
        """
        Initialize a new problem based on the description.
        
        Args:
            problem_description: User-provided description of the problem
        """
        logger.info(f"Initializing problem: {problem_description}")
        
        # Ask user for problem type
        problem_type = self.user_input_service.request_input(
            question="What type of problem are we solving?",
            options=["regression", "classification", "clustering", "forecasting"],
            response_format="single_select"
        )
        
        self.problem_context["problem_type"] = problem_type
        self.problem_context["description"] = problem_description
        
        # Get current state to preserve available_tables
        current_state = self.state_manager.get_state() or {}
        available_tables = current_state.get("available_tables", [])
        
        # Initialize state for the new problem
        new_state = self.state_manager.initialize_state(self.problem_context)
        
        # Preserve available_tables in the new state
        if available_tables:
            new_state["available_tables"] = available_tables
            self.state_manager.update_state(new_state)
        
        logger.info(f"Problem initialized with type: {problem_type}")
    
    def orchestrate_workflow(self) -> Dict[str, Any]:
        """
        Orchestrate the workflow based on the problem context.
        
        Returns:
            Dict containing the results of the workflow
        """
        logger.info("Starting workflow orchestration")
        
        # Determine initial flow
        if not self._determine_flow():
            return {"status": "failed", "reason": "Could not determine initial flow"}
        
        # Execute flows until the problem is solved or failed
        while not self._is_workflow_done():
            current_flow = self._get_next_flow()
            if not current_flow:
                break
                
            flow_result = self.flow_manager.execute_flow(
                flow_id=current_flow,
                input_data=self._prepare_flow_input(current_flow)
            )
            
            self._process_flow_result(current_flow, flow_result)
        
        return self._prepare_final_results()
    
    def _determine_flow(self, current_flow: str = None, result: Dict[str, Any] = None) -> bool:
        """
        Determine the next flow to execute.
        
        Args:
            current_flow: Currently executing flow (if any)
            result: Result of the current flow (if any)
            
        Returns:
            True if a next flow was determined, False otherwise
        """
        logger.info("Determining next flow")
        
        # Get current state
        state = self.state_manager.get_state()
        
        # Get problem context
        problem_type = self.problem_context.get("problem_type", "unknown")
        problem_desc = self.problem_context.get("description", "No description")
        
        # Get available flows
        available_flows = self.flow_manager.get_available_flows()
        
        # Get flow history
        flow_history = state.get("flow_history", [])
        flow_history_str = "Flow History:\n"
        if flow_history:
            for entry in flow_history:
                flow_history_str += f"- {entry['flow_id']}: {entry['summary']}\n"
        else:
            flow_history_str += "- No flows executed yet\n"
        
        # Get available tables
        available_tables = state.get("available_tables", [])
        tables_str = "Available Tables:\n"
        if available_tables:
            for table in available_tables:
                tables_str += f"- {table}\n"
        else:
            tables_str += "- No tables available\n"
        
        # Create context string
        context_str = f"{tables_str}\n{flow_history_str}"
        
        # Create prompt for LLM
        prompt = f"""
        You are an intelligent Meta Agent orchestrating a data science workflow.
        
        Current Problem: {problem_desc}
        Problem Type: {problem_type}
        
        Current State:
        {context_str}
        
        {flow_history_str}
        
        Available Flows:
        {', '.join(available_flows)}
        
        Based on the current state and problem context, determine the next flow to execute.
        
        Respond with a JSON object that includes:
        1. "next_flow": The name of the next flow to execute, or "COMPLETE" if the workflow is finished
        2. "reasoning": A conversational explanation of your thinking, as if you're talking to a human colleague. Be friendly and explain your decision in a way that shows your thought process.
        
        *** NOTE: IF MAPPING_FLOW IS NOT DONE YET, START WITH MAPPING_FLOW FIRST
        
        Example response:
        {{
            "next_flow": "mapping_flow",
            "reasoning": "I think we should start by mapping the data fields to standard formats. This will help us understand what we're working with and prepare the data for feature engineering."
        }}
        
        Or if the workflow is complete:
        {{
            "next_flow": "COMPLETE",
            "reasoning": "Great! We've completed all the necessary steps for this problem. We've mapped the data, created useful features, and trained a model that should help predict customer churn effectively."
        }}
        """
        
        # Make the LLM call
        llm_agent = LLMAgent(model_name=self.config.get("model_name", "default"))
        response = llm_agent.generate_text(prompt, max_tokens=250, temperature=0.7)
        
        # Parse the response
        try:
            # Extract JSON from response
            json_content = response
            if '{' in response and '}' in response:
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                json_content = response[start_idx:end_idx]
            
            decision = json.loads(json_content)
            next_flow = decision.get("next_flow")
            reasoning = decision.get("reasoning", "No reasoning provided")
            
            # Print the reasoning in a conversational way
            print(f"\n>> META AGENT: {reasoning}")
            
            if next_flow == "COMPLETE":
                logger.info("Workflow complete, no more flows needed")
                self.state_manager.set_state_status("completed")
                self.state_manager.set_next_flow(None)
                return True
            
            if next_flow in available_flows:
                logger.info(f"Next flow determined: {next_flow}")
                self.state_manager.set_next_flow(next_flow)
                return True
            else:
                logger.warning(f"Invalid flow determined: {next_flow}")
                return False
        
        except Exception as e:
            logger.error(f"Error determining next flow: {str(e)}")
            return False
    
    def _is_workflow_done(self) -> bool:
        """Check if the workflow is done (completed or failed)."""
        status = self.state_manager.get_state().get("status")
        return status in ["completed", "failed"]
    
    def _get_next_flow(self) -> Optional[str]:
        """Get the next flow to execute."""
        return self.state_manager.get_next_flow()
    
    def _prepare_flow_input(self, flow_id: str) -> Dict[str, Any]:
        """Prepare input for the specified flow."""
        state = self.state_manager.get_state()
        
        # For mapping flow, filter out already mapped tables
        input_tables = state.get("available_tables", [])
        if flow_id == "mapping_flow":
            # Only include original data tables, not derived tables
            input_tables = [table for table in input_tables if not table.startswith("mapped_data_")]
        
        return {
            "problem_context": self.problem_context,
            "current_state": state,
            "input_tables": input_tables,
        }
    
    def _process_flow_result(self, flow_id: str, result: Dict[str, Any]) -> None:
        """
        Process the result of a flow execution.
        
        Args:
            flow_id: ID of the flow
            result: Result of the flow execution
        """
        logger.info(f"Processing result of flow: {flow_id}")
        
        # Check if the flow was successful
        if result.get("status") == "failed":
            logger.error(f"Flow {flow_id} failed: {result.get('error')}")
            print(f">> Flow {flow_id} failed: {result.get('error')}")
            
            # Set workflow status to failed
            self.state_manager.set_state_status("failed")
            
            # Set next flow to None to stop the workflow
            self.state_manager.set_next_flow(None)
            
            # Record the failure in the state
            state = self.state_manager.get_state()
            if "failures" not in state:
                state["failures"] = []
            
            state["failures"].append({
                "flow_id": flow_id,
                "timestamp": self._get_timestamp(),
                "error": result.get("error", "Unknown error")
            })
            
            self.state_manager.update_state(state)
            
            # Print workflow failure message
            print(f">> Workflow failed at flow: {flow_id}")
            print(f">> Error: {result.get('error')}")
            
            return
        
        # Add flow to history
        self.state_manager.add_flow_to_history(flow_id, result)
        
        # Update available tables if output table is provided
        if "output_table" in result:
            state = self.state_manager.get_state()
            if "available_tables" not in state:
                state["available_tables"] = []
            
            if result["output_table"] not in state["available_tables"]:
                state["available_tables"].append(result["output_table"])
            
            self.state_manager.update_state(state)
        
        # Record data lineage if applicable
        if "output_table" in result:
            self.data_lineage_manager.record_transformation(
                input_tables=result.get("input_tables", []),
                output_table=result["output_table"],
                flow_id=flow_id,
                transformation_summary=result.get("summary", "")
            )
        
        # Update tables in state if provided
        if "tables" in result:
            state = self.state_manager.get_state()
            if "tables" not in state:
                state["tables"] = {}
            
            state["tables"].update(result["tables"])
            self.state_manager.update_state(state)
        
        # Determine next flow
        self._determine_flow()
    
    def _prepare_final_results(self) -> Dict[str, Any]:
        """Prepare the final results of the workflow."""
        state = self.state_manager.get_state()
        return {
            "status": state.get("status", "unknown"),
            "problem_type": self.problem_context.get("problem_type"),
            "results": state.get("results", {}),
            "tables": self.data_lineage_manager.get_final_tables(),
        } 