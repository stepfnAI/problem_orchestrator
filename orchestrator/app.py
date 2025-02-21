import sys
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from sfn_blueprint.utils.session_manager import SFNSessionManager
from orchestrator.utils.streamlit_views import StreamlitView
from orchestrator.states.onboarding_state import OnboardingState
import uuid

class Orchestrator:
    def __init__(self):
        self.view = StreamlitView("StepFn AI Orchestrator")
        self.session = SFNSessionManager()
        
        # Initialize session ID if not exists
        if not self.session.get('session_id'):
            self.session.set('session_id', str(uuid.uuid4()))
            
        # Initialize current state if not exists
        if not self.session.get('current_state'):
            self.session.set('current_state', 0)
        
        # Define workflow states
        self.states = [
            {
                'name': 'Onboarding',
                'description': 'Upload data and define problem statement',
                'class': OnboardingState(self.session, self.view)
            }
            # Add more states here as we develop them
            # {
            #     'name': 'Mapping',
            #     'description': 'Map and validate columns',
            #     'class': MappingState(self.session, self.view)
            # },
        ]

    def run(self):
        """Run the orchestrator"""
        # Display header
        self.view.display_title()
        
        # Show progress
        self._display_progress()
        
        # Show completed states summaries
        self._display_completed_summaries()
        
        # Execute current state
        current_state = self.states[self.session.get('current_state')]
        
        # Display current state header with reset button
        col1, col2 = self.view.create_columns([7, 1])
        with col1:
            self.view.display_subheader(f"Current Step: {current_state['name']}")
            self.view.display_markdown(f"_{current_state['description']}_")
        with col2:
            if self.view.display_button("🔄 Reset", key="reset_button"):
                # Clear session and rerun
                self.session.clear()
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
                summary_key = f"step_{idx+1}_summary"
                summary = self.session.get(summary_key)
                if summary:
                    # Create expandable section using container and checkbox
                    container = self.view.create_container()
                    expanded = self.view.checkbox(
                        f"✅ {state['name']} Summary",
                        key=f"expand_summary_{idx}",
                        value=False
                    )
                    if expanded:
                        container.markdown(summary)

def main():
    orchestrator = Orchestrator()
    orchestrator.run()

if __name__ == "__main__":
    main()