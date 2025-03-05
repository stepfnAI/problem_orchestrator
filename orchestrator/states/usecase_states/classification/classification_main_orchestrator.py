import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from orchestrator.states.usecase_states.classification.classif_step4_splitting import DataSplitting
from orchestrator.states.usecase_states.classification.classif_step5_model_training import ModelTraining
from orchestrator.states.usecase_states.classification.classif_step6_model_selection import ModelSelection
from orchestrator.states.usecase_states.classification.classif_step7_model_inference import ModelInference
from sfn_blueprint.utils.session_manager import SFNSessionManager
from orchestrator.utils.classif_model_manager import ModelManager

class ClassificationApp:
    def __init__(self, session_manager, view):
        self.view = view
        self.session = session_manager
        self.orchestrators = {
            4: DataSplitting(self.session, self.view, validation_window=3),
            5: ModelTraining(self.session, self.view),
            6: ModelSelection(self.session, self.view),
            7: ModelInference(self.session, self.view)
        }
        self.step_titles = {
            4: "Data Splitting",
            5: "Model Training",
            6: "Model Selection",
            7: "Model Inference"
        }
        
        # Initialize current step if not exists
        if not self.session.get('current_step'):
            self.session.set('current_step', 4)  # Start at data splitting
        
        # Load data from previous steps
        self._load_data_from_previous_steps()
        
    def run(self):
        """Main application flow"""
        # Get current step
        current_step = self.session.get('current_step', 4)
        
        # Display completed steps
        self._display_completed_steps(current_step)
        
        # Display current step header
        self.view.display_header(f"Step {current_step}: {self.step_titles[current_step]}")
        
        # Execute current step
        if current_step in self.orchestrators:
            self.view.display_markdown("---")
            result = self.orchestrators[current_step].execute()
            
            # Only advance if not the final step and execution was successful
            if result and current_step < max(self.orchestrators.keys()):
                self._advance_step()
            # For final step, just stay on the same page
            elif result and current_step == max(self.orchestrators.keys()):
                self.view.show_message("✅ Analysis complete!", "success")
                return
    
    def _display_completed_steps(self, current_step):
        """Display summary of completed steps"""
        for step in range(4, current_step):
            if self.session.get(f'step_{step}_complete'):
                self.view.display_header(f"Step {step}: {self.step_titles[step]}")
                self._display_step_summary(step)
                self.view.display_markdown("---")
    
    def _display_step_summary(self, step):
        """Display summary for a completed step"""
        summary = self.session.get(f'step_{step}_summary')
        if summary:
            self.view.show_message(summary, "success")
    
    def _advance_step(self):
        """Advance to the next step"""
        current_step = self.session.get('current_step', 4)
        self.session.set('current_step', current_step + 1)
        self.view.rerun_script()

    def _load_data_from_previous_steps(self):
        """Load data from previous steps in the main workflow"""
        try:
            # Get the joined dataframe from the join state
            from orchestrator.storage.db_connector import DatabaseConnector
            import sqlite3
            import pandas as pd
            import json
            
            db = DatabaseConnector()
            session_id = self.session.get('session_id')
            
            # Load the joined dataframe if not already in session
            if self.session.get('df') is None:
                # Try to get the final table name from join summary
                with sqlite3.connect(db.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT final_table_name, final_mappings FROM join_summary WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
                        (session_id,)
                    )
                    result = cursor.fetchone()
                    
                    if not result:
                        self.view.show_message("❌ No join summary found in database", "error")
                        return False
                        
                    final_table_name = result[0]
                    final_mappings_json = result[1]
                    
                    print(f"<<<>>> Found final table name in classification: {final_table_name}")
                    print(f"<<<>>> Found final mappings in classification: {final_mappings_json}")
                    
                    # Load the final table data
                    df = pd.read_sql(f"SELECT * FROM {final_table_name}", conn)
                    
                    if df.empty:
                        self.view.show_message("❌ Final table is empty", "error")
                        return False
                    
                    # Store the dataframe in session
                    self.session.set('df', df)
                    self.session.set('final_table_name', final_table_name)
                    self.view.show_message("✅ Successfully loaded joined data from previous steps", "success")
                    
                    # Load the final mappings if available
                    if final_mappings_json:
                        try:
                            field_mappings = json.loads(final_mappings_json)
                            print(f">>> Loaded field mappings from join summary: {field_mappings}")
                            self.session.set('field_mappings', field_mappings)
                        except Exception as e:
                            print(f">>> Error parsing final mappings: {str(e)}")
                            # We'll try to get mappings from mapping summary as fallback
                            field_mappings = None
                    else:
                        field_mappings = None
                        
                    # If we couldn't get mappings from join summary, try mapping summary as fallback
                    if not field_mappings:
                        # Try to get mappings from mapping summary
                        cursor.execute(
                            "SELECT mappings FROM mappings_summary WHERE session_id = ? AND table_name = '_final_mappings' ORDER BY created_at DESC LIMIT 1",
                            (session_id,)
                        )
                        mapping_result = cursor.fetchone()
                        
                        if mapping_result:
                            try:
                                field_mappings = json.loads(mapping_result[0])
                                print(f">>> Loaded field mappings from mapping summary: {field_mappings}")
                                self.session.set('field_mappings', field_mappings)
                            except Exception as e:
                                print(f">>> Error parsing mappings from mapping summary: {str(e)}")
                                field_mappings = None
                    
                    # If we still don't have mappings, try to get them from the original mapping state
                    if not field_mappings:
                        field_mappings = self._fetch_mappings_from_db(db, session_id)
                        if field_mappings:
                            self.session.set('field_mappings', field_mappings)
                        else:
                            self.view.show_message("⚠️ No field mappings found", "warning")
            
            return True  # Return True to indicate successful loading
                
        except Exception as e:
            import traceback
            print(f">>> Error loading data from previous steps: {str(e)}")
            print(f">>> Traceback: {traceback.format_exc()}")
            self.view.show_message(f"Error loading data from previous steps: {str(e)}", "error")
            return False

    def _fetch_mappings_from_db(self, db, session_id):
        """Fetch and process field mappings from database
        
        Returns:
            dict: Processed field mappings with all required fields
        """
        try:
            import sqlite3
            import json
            
            with sqlite3.connect(db.db_path) as conn:
                cursor = conn.cursor()
                
                # First try to get table-specific mappings
                cursor.execute(
                    "SELECT table_name, mappings FROM mappings_summary WHERE session_id = ? AND table_name != '_state_summary'",
                    (session_id,)
                )
                results = cursor.fetchall()
                
                # If no table-specific mappings, try to get from state summary
                if not results:
                    cursor.execute(
                        "SELECT mappings FROM mappings_summary WHERE session_id = ? AND table_name = '_state_summary'",
                        (session_id,)
                    )
                    state_result = cursor.fetchone()
                    
                    if state_result:
                        # Parse state summary mappings
                        state_mappings = json.loads(state_result[0])
                        print(f"Found mappings in state summary: {state_mappings}")
                        
                        # Extract any field mappings from state summary
                        field_mappings = {}
                        # Look for any key that might be a field mapping
                        for key, value in state_mappings.items():
                            if key not in ['tables_mapped', 'mandatory_columns_mapped', 'prediction_level', 
                                          'has_product_mapping', 'problem_type', 'completion_time']:
                                field_mappings[key] = value
                    else:
                        # No mappings found at all
                        print("No mappings found in database")
                        return None
                else:
                    # Combine mappings from all tables
                    field_mappings = {}
                    for table_name, mappings_json in results:
                        table_mappings = json.loads(mappings_json)
                        field_mappings.update(table_mappings)
                
                # Get the dataframe to check columns
                df = self.session.get('df')
                if df is None:
                    print("DataFrame not loaded yet")
                    return field_mappings  # Return what we have so far
                
                # For classification, ensure we have target column
                if 'target' not in field_mappings:
                    # Try to find a suitable target column from the dataframe
                    # This is just a fallback if no target was specified in mappings
                    potential_targets = ['target', 'label', 'class', 'outcome', 'result']
                    for col in potential_targets:
                        if col in df.columns:
                            field_mappings['target'] = col
                            break
                
                print(f"Final field mappings: {field_mappings}")
                return field_mappings
                
        except Exception as e:
            print(f"Error fetching mappings: {str(e)}")
            return None 