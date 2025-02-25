import sys
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from sfn_blueprint.utils.session_manager import SFNSessionManager
from orchestrator.utils.streamlit_views import StreamlitView
from orchestrator.states.onboarding_state import OnboardingState
from orchestrator.states.mapping_state import MappingState
from orchestrator.storage.db_connector import DatabaseConnector
import uuid
from orchestrator.states.aggregation_state import AggregationState

class Orchestrator:
    def __init__(self):
        self.view = StreamlitView("StepFn AI Orchestrator")
        self.session = SFNSessionManager()
        self.db = DatabaseConnector()
        
        # Initialize session ID if not exists
        if not self.session.get('session_id'):
            self.session.set('session_id', str(uuid.uuid4()))
            
        # Initialize current state if not exists
        if not self.session.get('current_state'):
            self.session.set('current_state', 0)
        
        # Create state instances
        self.states = [
            {
                'id': 'onboarding',
                'name': 'Data Onboarding',
                'description': 'Upload data and define problem statement',
                'class': OnboardingState(self.session, self.view)
            },
            {
                'id': 'mapping',
                'name': 'Column Mapping',
                'description': 'Map and validate columns for each table',
                'class': MappingState(self.session, self.view)
            },
            {
                'id': 'aggregation',
                'name': 'Data Aggregation',
                'description': 'Aggregate and transform data for modeling',
                'class': AggregationState(self.session, self.view)
            }
            # Add more states as needed
        ]
        
        # Define workflow steps
        self.steps = [
            {'id': 'onboarding', 'name': 'Data Onboarding'},
            {'id': 'mapping', 'name': 'Column Mapping'},
            {'id': 'aggregation', 'name': 'Data Aggregation'},
            {'id': 'model_setup', 'name': 'Model Setup'},
            {'id': 'training', 'name': 'Training & Evaluation'}
        ]

    def run(self):
        """Run the orchestrator"""
        # Display header with global reset button
        col1, col2 = self.view.create_columns([5, 1])
        with col1:
            self.view.display_title()
        with col2:
            # Use HTML to keep emoji and text on same line
            if self.view.display_button("🔄 Reset All", key="global_reset_button", use_container_width=False):
                # Clear session and database for this session
                session_id = self.session.get('session_id')
                self.db.reset_session_data(session_id)
                self.session.clear()
                self.view.rerun_script()
        
        # Show progress
        self._display_progress()
        
        # Show completed states summaries
        self._display_completed_summaries()
        
        # Execute current state
        current_state = self.states[self.session.get('current_state')]
        
        # Display current state header with state reset button
        col1, col2 = self.view.create_columns([4, 1])
        with col1:
            self.view.display_subheader(f"Current Step: {current_state['name']}")
            self.view.display_markdown(f"_{current_state['description']}_")
        with col2:
            # Use HTML to keep emoji and text on same line
            if self.view.display_button("🔄 Reset Step", key="reset_state_button", use_container_width=False):
                # Reset only the current state
                self._reset_current_state()
                self.view.rerun_script()
        
        self.view.display_markdown("---")
        
        # Execute state
        if current_state['class'].execute():
            # If state execution complete, move to next state
            if self.session.get('current_state') < len(self.states) - 1:
                self.session.set('current_state', self.session.get('current_state') + 1)
                self.view.rerun_script()
                
    def _display_progress(self):
        """Display workflow progress"""
        # Calculate progress
        total_states = len(self.states)
        current_state = self.session.get('current_state')
        progress = (current_state) / total_states
        
        # Create progress bar
        progress_container = self.view.create_container()
        progress_bar = self.view.load_progress_bar(progress)
            
        # Show steps
        columns = self.view.create_columns(total_states)
        for idx, state in enumerate(self.states):
            if idx < current_state:
                columns[idx].markdown(f"✅ {state['name']}")
            elif idx == current_state:
                columns[idx].markdown(f"🔄 {state['name']}")
            else:
                columns[idx].markdown(f"⏳ {state['name']}")
                        
        self.view.display_markdown("---")
        
    def _display_completed_summaries(self):
        """Display summaries of completed states"""
        current_state = self.session.get('current_state')
        
        for idx, state in enumerate(self.states):
            if idx < current_state:
                state_name = state['name']
                state_class = state['class']
                
                # Create a container for each state summary
                container = self.view.create_container()
                
                with container:
                    # Check if the state has a show_state_summary method
                    if hasattr(state_class, '_show_state_summary'):
                        # Call the state's show_state_summary method
                        state_class._show_state_summary()
                    else:
                        # Fall back to the old method if show_state_summary doesn't exist
                        summary_key = f"step_{idx+1}_summary"
                        summary = self.session.get(summary_key)
                        if summary:
                            self.view.display_markdown(summary)
                
                # Add a separator between state summaries
                if idx < current_state - 1:
                    self.view.display_markdown("---")

    def _reset_current_state(self):
        """Reset only the current state"""
        current_state_idx = self.session.get('current_state')
        current_state = self.states[current_state_idx]
        state_name = current_state['name'].lower()
        
        # Clear state-specific session variables
        if state_name == 'onboarding':
            self.session.set('data_upload_complete', False)
            self.session.set('problem_statement_complete', False)
            self.session.set('target_column_complete', False)
            self.session.set('summary_complete', False)
            self.session.set('onboarding_complete', False)
            self.session.set('uploaded_tables', [])  # Clear uploaded tables
            self.session.set('show_options', False)  # Reset upload UI state
        elif state_name == 'mapping':
            self.session.set('tables_processed', False)
            self.session.set('prediction_level_complete', False)
            self.session.set('mapping_summary_complete', False)
            self.session.set('mapping_complete', False)
            self.session.set('mapping_results', {})
        
        # Reset state-specific database entries
        session_id = self.session.get('session_id')
        self.db.reset_state_data(session_id, state_name)

def main():
    orchestrator = Orchestrator()
    orchestrator.run()

if __name__ == "__main__":
    main()