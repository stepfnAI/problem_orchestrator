from orchestrator.states.base_state import BaseState
from orchestrator.storage.db_connector import DatabaseConnector
from orchestrator.agents.mapping_agent import SFNDataMappingAgent
from typing import List, Dict, Optional
import pandas as pd
import json
from sfn_blueprint import Task


class MappingState(BaseState):
    """State handling the mapping process for all uploaded tables"""
    
    MANDATORY_COLUMNS = {
        'classification': ['id'],
        'regression': ['id'],
        'recommendation': ['product_id'],
        'clustering': [],
        'forecasting': ['timestamp']
    }

    def __init__(self, session_manager, view):
        self.session = session_manager
        self.view = view
        self.db = DatabaseConnector()
        self.mapping_agent = SFNDataMappingAgent()
        
    def execute(self):
        """Execute the mapping workflow"""
        # Check if state is already complete
        if self.session.get('mapping_complete'):
            return True
            
        # Step 1: Fetch onboarding summary if not done
        if not self.session.get('onboarding_summary'):
            if not self._fetch_onboarding_summary():
                return False
                
        # Step 2: Process each table if not done
        if not self.session.get('tables_processed'):
            if not self._process_tables():
                return False
                
        # Step 3: Handle prediction level if needed
        if not self.session.get('prediction_level_complete'):
            if self._needs_prediction_level() and not self._handle_prediction_level():
                return False
            else:
                self.session.set('prediction_level_complete', True)
                
        # Step 4: Generate and save state summary
        if not self.session.get('mapping_summary_complete'):
            if not self._generate_state_summary():
                return False
                
        return True
        
    def _fetch_onboarding_summary(self) -> bool:
        """Fetch onboarding summary from database"""
        try:
            session_id = self.session.get('session_id')
            summary = self.db.fetch_state_summary('onboarding', session_id)
            
            if not summary:
                self.view.show_message("❌ Onboarding summary not found", "error")
                return False
                
            # Parse and store relevant information
            self.session.set('problem_type', summary['problem_type'])
            self.session.set('target_column', summary['target_column'])
            self.session.set('table_names', json.loads(summary['table_names']))
            self.session.set('onboarding_summary', summary)
            return True
            
        except Exception as e:
            self.view.show_message(f"Error fetching onboarding summary: {str(e)}", "error")
            return False
            
    def _process_tables(self) -> bool:
        """Process each uploaded table"""
        table_names = self.session.get('table_names', [])
        problem_type = self.session.get('problem_type')
        
        # Initialize mapping results if not exists
        if not self.session.get('mapping_results'):
            self.session.set('mapping_results', {})
            
        mapping_results = self.session.get('mapping_results')
        
        # Process each table that hasn't been mapped yet
        for table_name in table_names:
            if table_name not in mapping_results:
                if not self._process_single_table(table_name):
                    return False
                    
        self.session.set('tables_processed', True)
        return True
        
    def _process_single_table(self, table_name: str) -> bool:
        """Process a single table for mapping"""
        try:
            # Fetch table from database
            session_id = self.session.get('session_id')
            df = self.db.fetch_table('onboarding', table_name, session_id)
            
            if df is None:
                self.view.show_message(f"❌ Could not fetch table {table_name}", "error")
                return False
                
            # Get AI suggestions for mapping
            problem_type = self.session.get('problem_type')
            
            # Create task with additional info in data dictionary
            task_data = {
                'df': df,
                'problem_type': problem_type,
                'table_name': table_name
            }
            
            # Updated Task creation to match the correct signature
            mapping_task = Task("Map columns", task_data)
            mappings = self.mapping_agent.execute_task(mapping_task)
            
            # Validate mandatory columns
            mandatory_cols = self.MANDATORY_COLUMNS.get(problem_type, [])
            
            # Display mapping interface
            self.view.display_subheader(f"Column Mapping for {table_name}")
            self.view.display_markdown("Please confirm or modify the suggested mappings:")
            
            confirmed_mappings = {}
            for col_type in mandatory_cols:
                suggested_col = mappings.get(col_type)
                selected_col = self.view.select_box(
                    f"Select {col_type} column:",
                    options=[""] + list(df.columns),
                    default=suggested_col,
                    key=f"mapping_{table_name}_{col_type}"
                )
                confirmed_mappings[col_type] = selected_col
                
            if self.view.display_button(f"Confirm Mappings for {table_name}"):
                # Validate mappings
                if not all(confirmed_mappings.values()):
                    self.view.show_message("❌ All mandatory columns must be mapped", "error")
                    return False
                    
                # Store mappings
                mapping_results = self.session.get('mapping_results', {})
                mapping_results[table_name] = confirmed_mappings
                self.session.set('mapping_results', mapping_results)
                
                # Save mappings to database
                if not self._save_mappings_to_db(table_name, confirmed_mappings):
                    return False
                    
                return True
                
            return False
            
        except Exception as e:
            self.view.show_message(f"Error processing table {table_name}: {str(e)}", "error")
            return False
            
    def _needs_prediction_level(self) -> bool:
        """Check if prediction level selection is needed"""
        problem_type = self.session.get('problem_type')
        mapping_results = self.session.get('mapping_results', {})
        
        if problem_type not in ['classification', 'regression']:
            return False
            
        # Check if any table has product mapping
        for mappings in mapping_results.values():
            if mappings.get('product_id'):
                return True
                
        return False
        
    def _handle_prediction_level(self) -> bool:
        """Handle prediction level selection"""
        self.view.display_subheader("Prediction Level Configuration")
        
        selected_level = self.view.radio_select(
            "Select prediction level:",
            options=["Customer Level", "Customer + Product Level"],
            key="prediction_level"
        )
        
        if self.view.display_button("Confirm Prediction Level"):
            self.session.set('prediction_level', selected_level)
            self.session.set('prediction_level_complete', True)
            return True
            
        return False
        
    def _generate_state_summary(self) -> bool:
        """Generate and save mapping state summary"""
        try:
            mapping_results = self.session.get('mapping_results', {})
            prediction_level = self.session.get('prediction_level')
            
            summary = {
                'tables_mapped': json.dumps(list(mapping_results.keys())),
                'mandatory_columns_mapped': 'True',  # We validate this during mapping
                'prediction_level': prediction_level if prediction_level else 'N/A',
                'completion_time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save summary to database
            session_id = self.session.get('session_id')
            if self.db.save_state_summary('mapping', summary, session_id):
                self.session.set('mapping_summary_complete', True)
                self.session.set('mapping_complete', True)
                return True
                
            return False
            
        except Exception as e:
            self.view.show_message(f"Error generating summary: {str(e)}", "error")
            return False
            
    def _save_mappings_to_db(self, table_name: str, mappings: Dict) -> bool:
        """Save mappings to database"""
        try:
            session_id = self.session.get('session_id')
            mapping_data = {
                'table_name': table_name,
                'mappings': json.dumps(mappings),
                'session_id': session_id
            }
            return self.db.save_table_mappings(mapping_data)
        except Exception as e:
            self.view.show_message(f"Error saving mappings: {str(e)}", "error")
            return False 