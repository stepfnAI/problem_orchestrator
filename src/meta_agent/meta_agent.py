"""
Meta Agent - Central orchestrator for the problem-solving framework.
"""

from typing import Dict, List, Any, Optional
import logging

from flow_manager.flow_manager import FlowManager
from services.state_manager import StateManager
from services.user_input_service import UserInputService
from services.data_lineage_manager import DataLineageManager
from agents.mapping_agent import MappingAgent
from agents.feature_agent import FeatureAgent

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
        self.state_manager = StateManager()
        
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
        
        # Initialize state for the new problem
        self.state_manager.initialize_state(self.problem_context)
        
        logger.info(f"Problem initialized with type: {problem_type}")
    
    def orchestrate_workflow(self) -> Dict[str, Any]:
        """
        Orchestrate the workflow based on the problem context.
        
        Returns:
            Dict containing the results of the workflow
        """
        logger.info("Starting workflow orchestration")
        
        # Determine initial flow based on problem type
        if not self._determine_initial_flow():
            return {"status": "failed", "reason": "Could not determine initial flow"}
        
        # Execute flows until the problem is solved
        while not self._is_problem_solved():
            current_flow = self._get_next_flow()
            if not current_flow:
                break
                
            flow_result = self.flow_manager.execute_flow(
                flow_id=current_flow,
                input_data=self._prepare_flow_input(current_flow)
            )
            
            self._process_flow_result(current_flow, flow_result)
        
        return self._prepare_final_results()
    
    def _determine_initial_flow(self) -> bool:
        """Determine the initial flow based on problem context."""
        problem_type = self.problem_context.get("problem_type")
        
        # For all problem types, we typically start with mapping
        self.state_manager.set_next_flow("mapping_flow")
        return True
    
    def _is_problem_solved(self) -> bool:
        """Check if the problem has been solved."""
        return self.state_manager.get_state().get("status") == "completed"
    
    def _get_next_flow(self) -> Optional[str]:
        """Get the next flow to execute."""
        return self.state_manager.get_next_flow()
    
    def _prepare_flow_input(self, flow_id: str) -> Dict[str, Any]:
        """Prepare input for the specified flow."""
        state = self.state_manager.get_state()
        return {
            "problem_context": self.problem_context,
            "current_state": state,
            "input_tables": state.get("available_tables", []),
        }
    
    def _process_flow_result(self, flow_id: str, result: Dict[str, Any]) -> None:
        """Process the result from a flow execution."""
        if result.get("status") == "completed":
            # Update state with flow results
            self.state_manager.update_state_with_flow_result(flow_id, result)
            
            # Update data lineage
            if "output_table" in result:
                self.data_lineage_manager.record_transformation(
                    input_tables=result.get("input_tables", []),
                    output_table=result["output_table"],
                    flow_id=flow_id,
                    transformation_summary=result.get("summary", "")
                )
            
            # Determine next flow
            self._determine_next_flow(flow_id, result)
        elif result.get("status") == "failed":
            logger.error(f"Flow {flow_id} failed: {result.get('error')}")
            self.state_manager.set_state_status("failed")
    
    def _determine_next_flow(self, current_flow: str, result: Dict[str, Any]) -> None:
        """Determine the next flow based on current flow result."""
        # This is a simplified version - in reality, this would be more complex
        flow_sequence = {
            "mapping_flow": "feature_suggestion_flow",
            "feature_suggestion_flow": "data_splitting_flow",
            "data_splitting_flow": "model_training_flow",
            "model_training_flow": None  # End of simple workflow
        }
        
        next_flow = flow_sequence.get(current_flow)
        if next_flow:
            self.state_manager.set_next_flow(next_flow)
        else:
            self.state_manager.set_state_status("completed")
    
    def _prepare_final_results(self) -> Dict[str, Any]:
        """Prepare the final results of the workflow."""
        state = self.state_manager.get_state()
        return {
            "status": state.get("status", "unknown"),
            "problem_type": self.problem_context.get("problem_type"),
            "results": state.get("results", {}),
            "tables": self.data_lineage_manager.get_final_tables(),
        } 