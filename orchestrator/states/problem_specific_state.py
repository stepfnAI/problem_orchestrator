from pathlib import Path
import sys
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from orchestrator.states.usecase_states.clustering.clustering_main_orchestrator import ClusteringOrchestrator

logger = logging.getLogger(__name__)

class ProblemSpecificState:
    """
    State for handling problem-specific execution based on the problem type.
    This state directs the flow to the appropriate problem-specific orchestrator.
    """
    
    def __init__(self, session, view):
        self.name = "Problem Specific Execution"
        self.session = session
        self.view = view
        
    def execute(self) -> bool:
        """
        Execute the problem-specific state logic.
        
        Returns:
            bool: True if the state execution is complete, False otherwise
        """
        logger.info("Executing Problem Specific State")
        
        # Get problem type from session
        problem_type = self.session.get('next_state', 'unknown')
        logger.info(f"Problem type: {problem_type}")
        
        # Route to the appropriate problem-specific orchestrator
        if problem_type == 'clustering':
            logger.info("Transitioning to Clustering workflow")
            clustering_orchestrator = ClusteringOrchestrator(self.session, self.view)
            clustering_orchestrator.run()
            return True
        elif problem_type == 'classification':
            # TODO: Implement classification orchestrator
            self.view.show_message("Classification workflow not yet implemented", "warning")
            return False
        elif problem_type == 'regression':
            # TODO: Implement regression orchestrator
            self.view.show_message("Regression workflow not yet implemented", "warning")
            return False
        elif problem_type == 'forecasting':
            # TODO: Implement forecasting orchestrator
            self.view.show_message("Forecasting workflow not yet implemented", "warning")
            return False
        elif problem_type == 'recommendation':
            # TODO: Implement recommendation orchestrator
            self.view.show_message("Recommendation workflow not yet implemented", "warning")
            return False
        else:
            self.view.show_message(f"Unknown problem type: {problem_type}", "error")
            return False
            
    def _show_state_summary(self):
        """Show a summary of the state execution"""
        problem_type = self.session.get('next_state', 'unknown')
        summary = f"Problem type: {problem_type.capitalize()}"
        self.view.display_markdown(summary) 