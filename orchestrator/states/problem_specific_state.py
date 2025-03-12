from pathlib import Path
import sys
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from orchestrator.states.usecase_states.clustering.clustering_main_orchestrator import ClusteringOrchestrator
from orchestrator.states.usecase_states.recommendation.recommendation_main_orchestrator import RecommendationOrchestrator
from orchestrator.states.usecase_states.regression_forecasting.regression_main_orchestrator import RegressionApp
from orchestrator.states.usecase_states.classification.classification_main_orchestrator import ClassificationApp
from orchestrator.states.target_generation_state import TargetGenerationState

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
        
        # Fetch mappings and problem details if not already done
        if not self.session.get('field_mappings'):
            self._fetch_mappings_and_problem_details()
            
        # Try to load the joined dataframe
        try:
            from orchestrator.storage.db_connector import DatabaseConnector
            db = DatabaseConnector()
            session_id = self.session.get('session_id')
            
            # Load the joined dataframe and set it if not None
            df = db.get_joined_dataframe(session_id)
            if df is not None:
                self.session.set('df', df)
                logger.info(f"Loaded joined dataframe with shape: {df.shape}")
        except Exception as e:
            logger.error(f"Error loading joined dataframe: {str(e)}")
                
        # Execute target generation if needed
        if problem_type in ['classification', 'regression', 'forecasting']:
            target_state = TargetGenerationState(self.session, self.view)
            
            target_result = target_state.execute()
            if not target_result:
                return False
                
            # Debug print to verify target column exists after target generation
            df = self.session.get('df')
            mappings = self.session.get('field_mappings', {})
            print(f"DEBUG: After target generation - df columns: {list(df.columns) if df is not None else 'None'}")
            print(f"DEBUG: After target generation - target in mappings: {mappings.get('target')}")
            print(f"DEBUG: After target generation - target in df columns: {mappings.get('target') in df.columns if df is not None and mappings.get('target') else False}")
        
        # Route to the appropriate problem-specific orchestrator
        if problem_type == 'clustering':
            logger.info("Transitioning to Clustering workflow")
            clustering_orchestrator = ClusteringOrchestrator(self.session, self.view)
            clustering_orchestrator.run()
            return True
        elif problem_type == 'recommendation':
            logger.info("Transitioning to Recommendation workflow")
            recommendation_orchestrator = RecommendationOrchestrator(self.session, self.view)
            recommendation_orchestrator.run()
            return True
        elif problem_type == 'regression':
            logger.info("Transitioning to Regression workflow")
            # Set flag for regression (not forecasting)
            self.session.set('is_forecasting', False)
            regression_app = RegressionApp(self.session, self.view)
            regression_app.run()
            return True
        elif problem_type == 'forecasting':
            logger.info("Transitioning to Forecasting workflow")
            # Set flag for forecasting
            self.session.set('is_forecasting', True)
            regression_app = RegressionApp(self.session, self.view)
            regression_app.run()
            return True
        elif problem_type == 'classification':
            logger.info("Transitioning to Classification workflow")
            classification_app = ClassificationApp(self.session, self.view)
            classification_app.run()
            return True
        else:
            self.view.show_message(f"Unknown problem type: {problem_type}", "error")
            return False
    
    def _fetch_mappings_and_problem_details(self):
        """
        Fetch mappings from mapping_summary table and problem statement details
        from onboarding_summary table in the database.
        """
        try:
            from orchestrator.storage.db_connector import DatabaseConnector
            db = DatabaseConnector()
            session_id = self.session.get('session_id')
            
            # Fetch mappings from mapping_summary table
            mappings = db.get_mapping_summary(session_id)
            if mappings:
                logger.info(f"Retrieved mappings: {mappings}")
                self.session.set('field_mappings', mappings)
            else:
                logger.warning("No mappings found in database")
            
            # Fetch problem statement details from onboarding_summary table
            problem_details = db.get_onboarding_summary(session_id)
            if problem_details:
                logger.info(f"Retrieved problem details: {problem_details}")
                self.session.set('problem_details', problem_details)
                
                # Set forecast periods if available (for forecasting)
                if 'forecast_periods' in problem_details:
                    self.session.set('forecast_periods', problem_details.get('forecast_periods', 3))
            else:
                logger.warning("No problem details found in database")
                
        except Exception as e:
            logger.error(f"Error fetching mappings and problem details: {str(e)}")
            self.view.show_message(f"Error loading data: {str(e)}", "error")
            
    def _show_state_summary(self):
        """Show a summary of the state execution"""
        problem_type = self.session.get('next_state', 'unknown')
        summary = f"Problem type: {problem_type.capitalize()}"
        self.view.display_markdown(summary) 