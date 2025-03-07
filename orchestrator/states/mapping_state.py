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
        'clustering': ['id'],
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
            self._show_state_summary()  # Show summary if already complete
            return True
            
        # Step 1: Fetch onboarding summary if not done
        if not self.session.get('onboarding_summary'):
            if not self._fetch_onboarding_summary():
                return False
                
        # Step 2: Process each table if not done
        if not self.session.get('tables_processed'):
            # Display mapping status is now handled inside _process_tables
            # Removed duplicate call to _display_mapping_status()
            if not self._process_tables():
                return False
                
        # Step 3: Handle prediction level if needed
        if not self.session.get('prediction_level_complete'):
            if self._needs_prediction_level() and not self._handle_prediction_level():
                return False
            else:
                self.session.set('prediction_level_complete', True)
                
        # Step 4: Show final summary and proceed button
        if self.session.get('tables_processed') and self.session.get('prediction_level_complete') and not self.session.get('mapping_summary_complete'):
            # Show final summary
            self._show_final_summary()
            
            # Add proceed button
            if self.view.display_button("▶️ Proceed to Next Step"):
                # Generate and save state summary
                if not self._generate_state_summary():
                    return False
                self.session.set('mapping_complete', True)
                self._show_state_summary()  # Show final summary with success styling
                return True
                
        return False
        
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
            
            # Store recommendation approach if available
            if 'recommendation_approach' in summary and summary['recommendation_approach']:
                self.session.set('recommendation_approach', summary['recommendation_approach'])
                print(f">><<Loaded recommendation approach: {summary['recommendation_approach']}")
            
            # Store is_time_series if available
            if 'is_time_series' in summary and summary['is_time_series']:
                self.session.set('is_time_series', summary['is_time_series'] == 'True')
            
            self.session.set('onboarding_summary', summary)
            
            # Update mandatory columns based on problem type and recommendation approach
            self._update_mandatory_columns()
            
            return True
            
        except Exception as e:
            self.view.show_message(f"Error fetching onboarding summary: {str(e)}", "error")
            return False
            
    def _update_mandatory_columns(self):
        """Update mandatory columns based on problem type and recommendation approach"""
        problem_type = self.session.get('problem_type')
        
        # If problem type is recommendation, check the approach
        if problem_type == 'recommendation':
            recommendation_approach = self.session.get('recommendation_approach')
            print(f">><<Updating mandatory columns for recommendation approach: {recommendation_approach}")
            
            # If user-based approach, add 'id' to mandatory columns
            if recommendation_approach == 'user_based':
                self.MANDATORY_COLUMNS['recommendation'] = ['product_id', 'id']
                print(">><<Added 'id' to mandatory columns for user-based recommendation")
            else:
                # For item-based or any other approach, keep only product_id
                self.MANDATORY_COLUMNS['recommendation'] = ['product_id']
                print(">><<Using default mandatory columns for recommendation")
            
    def _display_mapping_status(self):
        """Display current mapping status for all tables"""
        table_names = self.session.get('table_names', [])
        mapping_results = self.session.get('mapping_results', {})
        problem_type = self.session.get('problem_type', 'Unknown')
        
        # Create status message
        status_msg = f"**Column Mapping Status**\n\n"
        status_msg += f"**Problem Type:** {problem_type}\n\n"
        
        # Count mapped and total tables
        mapped_count = len(mapping_results)
        total_count = len(table_names)
        
        status_msg += f"**Progress:** {mapped_count}/{total_count} tables mapped\n\n"
        
        # Show status for each table
        for idx, table_name in enumerate(table_names):
            status = "✅ Mapping confirmed" if table_name in mapping_results else "⏳ Yet to be mapped"
            status_msg += f"{idx+1}. **{table_name}**: {status}\n"
        
        # Display the status message
        self.view.show_message(status_msg, "info")
        self.view.display_markdown("---")

    def _process_tables(self) -> bool:
        """Process each uploaded table"""
        table_names = self.session.get('table_names', [])
        problem_type = self.session.get('problem_type')
        
        # Initialize mapping results if not exists
        if not self.session.get('mapping_results'):
            self.session.set('mapping_results', {})
            
        mapping_results = self.session.get('mapping_results')
        
        # Get current table being processed
        current_table_idx = self.session.get('current_mapping_table_idx', 0)
        
        # If we've processed all tables, mark as complete and return
        if current_table_idx >= len(table_names):
            self.session.set('tables_processed', True)
            self.view.show_message("✅ All tables have been successfully mapped!", "success")
            return True
        
        # Process current table
        current_table = table_names[current_table_idx]
        
        # If this table is already mapped, move to next
        if current_table in mapping_results:
            self.session.set('current_mapping_table_idx', current_table_idx + 1)
            self.view.rerun_script()
            return False
        
        # Process the current table
        if self._process_single_table(current_table):
            # If successful, increment the index for next run
            self.session.set('current_mapping_table_idx', current_table_idx + 1)
            self.view.rerun_script()
            return False
        
        return False
        
    def _show_mapping_summary(self):
        """Display a summary of all mappings"""
        mapping_results = self.session.get('mapping_results', {})
        problem_type = self.session.get('problem_type', 'Unknown')
        
        # Create summary message
        summary_msg = f"## Mapping Summary\n\n"
        summary_msg += f"**Problem Type:** {problem_type}\n\n"
        
        # Show mappings for each table
        for table_name, mappings in mapping_results.items():
            summary_msg += f"### {table_name}\n\n"
            
            # Separate mandatory and optional fields
            mandatory_cols = self.MANDATORY_COLUMNS.get(problem_type, [])
            
            # Show mandatory fields
            summary_msg += "**Mandatory Fields:**\n\n"
            for field, column in mappings.items():
                if field in mandatory_cols and column:
                    summary_msg += f"- {field}: `{column}`\n"
            
            # Show optional fields
            summary_msg += "\n**Optional Fields:**\n\n"
            for field, column in mappings.items():
                if field not in mandatory_cols and column:
                    summary_msg += f"- {field}: `{column}`\n"
            
            summary_msg += "\n---\n\n"
        
        # Display the summary
        self.view.display_markdown(summary_msg)

    def _process_single_table(self, table_name: str) -> bool:
        """Process a single table for mapping"""
        try:
            # Fetch table from database
            session_id = self.session.get('session_id')
            df = self.db.fetch_table('onboarding', table_name, session_id)
            
            if df is None:
                self.view.show_message(f"❌ Could not fetch table {table_name}", "error")
                return False
            
            # Display mapping status
            self._display_mapping_status()
            
            # Check if we're in confirmation mode
            if self.session.get(f'mapping_confirmed_{table_name}'):
                # Show the confirmed mappings
                self._show_confirmed_mappings(table_name)
                return True
            
            # Get AI suggestions for mapping with spinner
            problem_type = self.session.get('problem_type')
            
            # Check if we already have AI suggestions for this table
            ai_mappings_key = f'ai_mappings_{table_name}'
            mappings = self.session.get(ai_mappings_key)
            
            # Only call the AI if we don't have mappings yet
            if mappings is None:
                # Create task with additional info in data dictionary
                task_data = {
                    'df': df,
                    'problem_type': problem_type,
                    'table_name': table_name
                }
                
                # Show spinner while AI is working
                with self.view.display_spinner('🤖 AI is suggesting column mappings...'):
                    print(f">><<Calling mapping agent for table: {table_name}")
                    # Updated Task creation to match the correct signature
                    mapping_task = Task("Map columns", task_data)
                    mappings = self.mapping_agent.execute_task(mapping_task)
                    print(f">><<Mapping agent call completed for table: {table_name}")
                    
                # Store the AI suggestions in the session
                self.session.set(ai_mappings_key, mappings)
            
            # Get mandatory columns for this problem type
            mandatory_cols = self.MANDATORY_COLUMNS.get(problem_type, [])
            
            # Check if this is time series data
            is_time_series = self.session.get('is_time_series', False)
            
            # If it's time series data and problem type is regression or classification,
            # add timestamp to mandatory columns if not already there
            if is_time_series and problem_type in ['regression', 'classification']:
                if 'timestamp' not in mandatory_cols:
                    mandatory_cols.append('timestamp')
                    print(f">><<Added timestamp to mandatory columns for time series data")
            
            # Get optional columns (all mappings except mandatory ones)
            optional_cols = [k for k in mappings.keys() if k not in mandatory_cols]
            
            # Display mapping interface
            self.view.display_subheader(f"Column Mapping for {table_name}")
            
            # Show validation message for mappings
            print(f">>>Mappings before validation: {mappings}")
            
            # Validate mappings - ensure all mandatory fields have a value
            for field in mandatory_cols:
                if field not in mappings or not mappings[field]:
                    # If mandatory field is missing or empty, set to None
                    mappings[field] = None
            
            print(f">>>Mappings after validation: {mappings}")
            
            # Show success message for AI mappings
            mapped_count = sum(1 for v in mappings.values() if v)
            total_count = len(mappings)
            self.view.show_message(f"AI successfully mapped {mapped_count}/{total_count} fields", "info")
            
            # First show mandatory fields
            confirmed_mappings = {}
            self.view.display_markdown("**Mandatory Fields:**")
            for col_type in mandatory_cols:
                suggested_col = mappings.get(col_type)
                print(f">><<Before select_box for {col_type}, suggested value: {suggested_col}")
                selected_col = self.view.select_box(
                    f"Select {col_type} column:",
                    options=[""] + list(df.columns),
                    default=suggested_col,
                    key=f"mapping_{table_name}_{col_type}"
                )
                print(f">><<After select_box for {col_type}, selected value: {selected_col}")
                confirmed_mappings[col_type] = selected_col
            
            # Then show optional fields if available
            if optional_cols:
                self.view.display_markdown("**Optional Fields:**")
                for col_type in optional_cols:
                    suggested_col = mappings.get(col_type)
                    print(f">><<Before select_box for {col_type}, suggested value: {suggested_col}")
                    selected_col = self.view.select_box(
                        f"Select {col_type} column (optional):",
                        options=[""] + list(df.columns),
                        default=suggested_col,
                        key=f"mapping_{table_name}_{col_type}_opt"
                    )
                    print(f">><<After select_box for {col_type}, selected value: {selected_col}")
                    confirmed_mappings[col_type] = selected_col
            
            # Show confirm button
            if self.view.display_button(f"✅ Confirm Mappings", key=f"confirm_{table_name}"):
                # Validate mappings - only mandatory fields must be mapped
                if not all(confirmed_mappings.get(field) for field in mandatory_cols):
                    self.view.show_message("❌ All mandatory columns must be mapped", "error")
                    return False
                    
                # Store mappings
                mapping_results = self.session.get('mapping_results', {})
                mapping_results[table_name] = confirmed_mappings
                self.session.set('mapping_results', mapping_results)
                
                # Save mappings to database
                if not self._save_mappings_to_db(table_name, confirmed_mappings):
                    return False
                
                # Mark this table as confirmed
                self.session.set(f'mapping_confirmed_{table_name}', True)
                
                # Show success message
                self.view.show_message(f"✅ Mappings for {table_name} confirmed!", "success")
                
                # Rerun to refresh the UI
                self.view.rerun_script()
                return True
            
            return False
            
        except Exception as e:
            self.view.show_message(f"❌ Error processing table {table_name}: {str(e)}", "error")
            return False
            
    def _show_confirmed_mappings(self, table_name: str):
        """Show the confirmed mappings for a table"""
        mapping_results = self.session.get('mapping_results', {})
        problem_type = self.session.get('problem_type', 'Unknown')
        
        if table_name not in mapping_results:
            return
        
        mappings = mapping_results[table_name]
        
        # Display confirmation message
        self.view.display_subheader(f"Confirmed Mappings for {table_name}")
        self.view.show_message(f"✅ Mappings for {table_name} have been confirmed", "success")
        
        # Separate mandatory and optional fields
        mandatory_cols = self.MANDATORY_COLUMNS.get(problem_type, [])
        
        # Create a table to display mappings
        mapping_data = []
        
        # Add mandatory fields
        self.view.display_markdown("**Mandatory Fields:**")
        for field, column in mappings.items():
            if field in mandatory_cols and column:
                mapping_data.append({"Field Type": field, "Mapped Column": column})
        
        self.view.display_table(mapping_data)
        mapping_data = []
        
        # Add optional fields
        self.view.display_markdown("**Optional Fields:**")
        for field, column in mappings.items():
            if field not in mandatory_cols and column:
                mapping_data.append({"Field Type": field, "Mapped Column": column})
        
        self.view.display_table(mapping_data)
        self.view.display_markdown("---")

    def _needs_prediction_level(self) -> bool:
        """Check if prediction level selection is needed"""
        problem_type = self.session.get('problem_type')
        mapping_results = self.session.get('mapping_results', {})
        
        if problem_type not in ['classification', 'regression']:
            return False
            
        # Skip if no mappings
        if not mapping_results:
            return False
            
        # Check if ALL tables have product mapping (not just any table)
        for table_name, mappings in mapping_results.items():
            # If any table doesn't have product_id mapping, return False
            if not mappings.get('product_id'):
                return False
                
        # All tables have product_id mapping
        return True
        
    def _handle_prediction_level(self) -> bool:
        """Handle prediction level selection"""
        self.view.display_subheader("Prediction Level Configuration")
        
        selected_level = self.view.radio_select(
            "Select prediction level:",
            options=["ID Level", "ID+Product Level"],
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
            problem_type = self.session.get('problem_type')
            
            # Check if any table has product mapping
            has_product_mapping = False
            for mappings in mapping_results.values():
                if mappings.get('product_id'):
                    has_product_mapping = True
                    break
            
            # Determine prediction level based on problem type and mappings
            prediction_level_info = 'N/A'
            if problem_type in ['classification', 'regression']:
                if has_product_mapping:
                    # If prediction level was explicitly set, use that
                    if prediction_level:
                        prediction_level_info = prediction_level
                    else:
                        # Default to ID-only if not explicitly set
                        prediction_level_info = 'ID Level'
                else:
                    # If no product mapping, it's always ID-only
                    prediction_level_info = 'ID Level'
            
            # Create a summary entry with enhanced information
            session_id = self.session.get('session_id')
            summary_data = {
                'table_name': '_state_summary',  # Special name to indicate this is the state summary
                'mappings': json.dumps({
                    'tables_mapped': list(mapping_results.keys()),
                    'mandatory_columns_mapped': True,  # We validate this during mapping
                    'prediction_level': prediction_level_info,
                    'has_product_mapping': has_product_mapping,
                    'problem_type': problem_type,
                    'completion_time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                }),
                'session_id': session_id
            }
            
            # Save to the mappings_summary table
            if self.db.save_table_mappings(summary_data):
                self.session.set('mapping_summary_complete', True)
                self.session.set('mapping_complete', True)
                
                # Also save a formatted summary for display
                summary_text = f"✅ Mapping Complete\n\n"
                summary_text += f"Tables mapped: {', '.join(mapping_results.keys())}\n"
                summary_text += f"Problem type: {problem_type}\n"
                
                if prediction_level_info != 'N/A':
                    summary_text += f"Prediction level: {prediction_level_info}\n"
                    if has_product_mapping:
                        summary_text += f"Product mapping: Available\n"
                
                self.session.set('step_2_summary', summary_text)
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

    def _show_final_summary(self):
        """Show final summary after all steps are complete"""
        mapping_results = self.session.get('mapping_results', {})
        problem_type = self.session.get('problem_type', 'Unknown')
        prediction_level = self.session.get('prediction_level', 'N/A')
        
        # Create summary message
        summary_msg = f"#### Mapping Complete\n\n"
        summary_msg += f"**Problem Type:** {problem_type}\n\n"
        
        if prediction_level != 'N/A':
            summary_msg += f"**Prediction Level:** {prediction_level}\n\n"
        
        # Show mappings for each table
        for table_name, mappings in mapping_results.items():
            summary_msg += f"##### {table_name}\n\n"
            
            # Separate mandatory and optional fields
            mandatory_cols = self.MANDATORY_COLUMNS.get(problem_type, [])
            
            # Show all mapped fields
            summary_msg += "**Mapped Fields:**\n"
            for field, column in mappings.items():
                if column:  # Only show fields that have been mapped
                    summary_msg += f"- {field} <-> {column}\n"
            
            summary_msg += "\n---\n"
        
        # Display the summary
        self.view.show_message(summary_msg, "success")

    def _show_state_summary(self):
        """Display a summary of the completed mapping state"""
        mapping_results = self.session.get('mapping_results', {})
        problem_type = self.session.get('problem_type', 'Unknown')
        prediction_level = self.session.get('prediction_level', 'N/A')
        
        # Create summary message
        summary_msg = f"#### ✅ Mapping Complete\n\n"
        summary_msg += f"**Problem Type:** {problem_type}\n"
        
        if prediction_level != 'N/A':
            summary_msg += f"**Prediction Level:** {prediction_level}\n"
        
        # Show mappings for each table
        for table_name, mappings in mapping_results.items():
            summary_msg += f"\n##### {table_name}\n\n"
            
            # Separate mandatory and optional fields
            mandatory_cols = self.MANDATORY_COLUMNS.get(problem_type, [])
            
            # Show all mapped fields together
            summary_msg += "**Mapped Fields:**\n"
            for field, column in mappings.items():
                if column:  # Only show fields that have been mapped
                    summary_msg += f"- {field} <-> {column}\n"
        
        # Display the summary with success styling
        self.view.show_message(summary_msg, "success") 