from orchestrator.states.base_state import BaseState
from orchestrator.storage.db_connector import DatabaseConnector

class AggregationState(BaseState):
    """State handling data aggregation"""
    
    def __init__(self, session_manager, view):
        self.session = session_manager
        self.view = view
        self.db = DatabaseConnector()
        
    def execute(self):
        """Execute the aggregation workflow"""
        print("Executing Aggregation State")
        
        # Display a simple message
        self.view.display_header("Data Aggregation")
        self.view.display_markdown("This is a placeholder for the Aggregation state.")
        
        # Add a button to complete this state for testing
        if self.view.display_button("Complete Aggregation (Test)"):
            self.session.set('aggregation_complete', True)
            return True
            
        return False
        
    def _show_state_summary(self):
        """Display a summary of the completed aggregation state"""
        print("Showing Aggregation State Summary")
        summary_message = "✅ Aggregation Complete"
        self.view.show_message(summary_message, "success") 