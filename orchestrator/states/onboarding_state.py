from orchestrator.states.base_state import BaseState
from orchestrator.storage.db_connector import DatabaseConnector
from typing import List, Dict, Optional
import pandas as pd
from copy import deepcopy
import json

class OnboardingState(BaseState):
    """State handling the onboarding process"""
    
    def __init__(self, session_manager, view):
        self.session = session_manager
        self.view = view
        self.db = DatabaseConnector()
        self.problem_types = {
            'classification': 'Predict a categorical outcome (e.g., Customer churn prediction - Will a customer churn or not?)',
            'regression': 'Predict a continuous value (e.g., Revenue prediction - How much revenue will a customer generate?)',
            'forecasting': 'Predict future values based on historical data (e.g., Monthly revenue forecast, Usage trend prediction)',
            'clustering': 'Group similar customers/products (e.g., Customer segmentation based on behavior)',
            'recommendation': 'Suggest items/products (e.g., Product recommendations based on usage patterns)'
        }
        
    def execute(self):
        """Execute the onboarding workflow"""
        # Check if state is already complete
        if self.session.get('onboarding_complete'):
            return True
            
        # Step 1: Handle data upload if not done
        if not self.session.get('data_upload_complete'):
            if not self._handle_data_upload():
                return False
                
        # Step 2: Handle problem statement if not done
        if not self.session.get('problem_statement_complete'):
            if not self._handle_problem_statement():
                return False
                
        # Step 3: Handle target column if needed
        if not self.session.get('target_column_complete'):
            if self._needs_target_column() and not self._handle_target_column():
                return False
            else:
                self.session.set('target_column_complete', True)
                
        # Step 4: Generate and save state summary
        if not self.session.get('summary_complete'):
            if not self._generate_state_summary():
                return False
                
        # If all steps complete, show summary and next step button
        if (self.session.get('data_upload_complete') and 
            self.session.get('problem_statement_complete') and 
            self.session.get('target_column_complete')):
            
            if not self.session.get('summary_complete'):
                if not self._generate_state_summary():
                    return False
                
            self.view.display_subheader("✅ Onboarding Complete!")
            self.view.display_markdown("### Summary")
            self.view.display_markdown(f"- Tables Uploaded: {len(self.session.get('uploaded_tables', []))}")
            self.view.display_markdown(f"- Problem Type: {self.session.get('problem_type').title()}")
            if self.session.get('target_column'):
                self.view.display_markdown(f"- Target Column: {self.session.get('target_column')}")
            
            if self.view.display_button("▶️ Proceed to Next Step"):
                self.session.set('onboarding_complete', True)
                return True
        
        return False
        
    def _handle_data_upload(self) -> bool:
        """Handle multiple table uploads"""
        self.view.display_header("Welcome to StepFn AI Orchestrator! 🚀")
        self.view.display_markdown(
            "Let's begin by uploading all the tables you want to process. "
            "You can upload multiple tables one by one."
        )
        
        # Initialize uploaded_tables in session if not exists
        if not self.session.get('uploaded_tables'):
            self.session.set('uploaded_tables', [])
        
        uploaded_tables = deepcopy(self.session.get('uploaded_tables'))
        
        # Always show current uploads summary if any
        if uploaded_tables:
            self.view.display_subheader("📊 Currently Uploaded Tables")
            summary = "The following tables have been uploaded:\n\n"
            for idx, table in enumerate(uploaded_tables, 1):
                summary += f"{idx}. **{table['name']}** (Rows: {table['rows']:,}, Columns: {table['columns']})\n"
            self.view.display_markdown(summary)
            self.view.display_markdown(f"\nTotal tables uploaded: **{len(uploaded_tables)}**")
            self.view.display_markdown("---")
        
        # If data_upload_complete is set, show summary only
        if self.session.get('data_upload_complete'):
            self.view.display_subheader("📊 Data Upload Summary")
            summary = "Successfully uploaded the following tables:\n\n"
            for idx, table in enumerate(uploaded_tables, 1):
                summary += f"{idx}. **{table['name']}** (Rows: {table['rows']:,}, Columns: {table['columns']})\n"
            self.view.display_markdown(summary)
            return True
        
        # Show file uploader if no tables or in upload mode
        if not uploaded_tables or not self.session.get('show_options'):
            self.view.display_subheader("📤 Upload Table")
            uploaded_file = self.view.file_uploader(
                "Choose a CSV or Excel file to upload",
                accepted_types=["csv", "xlsx"],
                key=f"table_upload_{len(uploaded_tables)}"
            )
            
            if uploaded_file:
                try:
                    # Check if file already uploaded
                    if any(table['name'] == uploaded_file.name for table in uploaded_tables):
                        self.view.show_message(f"⚠️ {uploaded_file.name} has already been uploaded!", "warning")
                        return False
                    
                    # Save file and get DataFrame
                    file_path = self.view.save_uploaded_file(uploaded_file)
                    df = pd.read_csv(file_path) if uploaded_file.name.endswith('.csv') \
                        else pd.read_excel(file_path)
                    
                    # Create new table info
                    new_table = {
                        'name': uploaded_file.name,
                        'path': file_path,
                        'rows': df.shape[0],
                        'columns': df.shape[1]
                    }
                    
                    # Create a new list with the additional table
                    new_uploaded_tables = uploaded_tables + [new_table]
                    self.session.set('uploaded_tables', new_uploaded_tables)
                    
                    # Show success message
                    self.view.show_message(
                        f"✅ Successfully uploaded {uploaded_file.name}",
                        "success"
                    )
                    
                    # Switch to options mode
                    self.session.set('show_options', True)
                    self.view.rerun_script()
                    
                except Exception as e:
                    self.view.show_message(f"❌ Error uploading file: {str(e)}", "error")
                    return False
        
        # Show options after successful upload
        if uploaded_tables:
            self.view.display_markdown("What would you like to do next?")
            
            col1, col2 = self.view.create_columns([1, 1])
            with col1:
                if self.view.display_button("📤 Upload Another Table"):
                    self.session.set('show_options', False)
                    self.view.rerun_script()
            with col2:
                if self.view.display_button("✅ All Tables Uploaded - Continue"):
                    self.session.set('data_upload_complete', True)
                    return True
        
        return False
        
    def _handle_problem_statement(self) -> bool:
        """Handle problem statement selection"""
        self.view.display_subheader("Problem Statement Identification")
                
        # Show problem type descriptions
        info_text = "We support the following types of problems:\n\n"
        for problem_type, description in self.problem_types.items():
            info_text += f"**{problem_type.title()}**: {description}\n\n"
        self.view.display_markdown(info_text)
        
        # Radio selection for problem type
        selected_type = self.view.display_radio(
            "Select your problem type:",
            options=list(self.problem_types.keys()),
            key="problem_type"
        )
        
        if self.view.display_button("Confirm Problem Type"):
            self.session.set('problem_type', selected_type)
            self.session.set('problem_statement_complete', True)
            
            # If clustering or recommendation, skip target column step
            if selected_type in ['clustering', 'recommendation']:
                self.session.set('target_column_complete', True)
                self.session.set('has_target_column', False)
                self.session.set('target_column', None)
            
            return True
            
        return False
        
    def _needs_target_column(self) -> bool:
        """Check if problem type needs target column"""
        problem_type = self.session.get('problem_type')
        return problem_type in ['classification', 'regression', 'forecasting']
        
    def _handle_target_column(self) -> bool:
        """Handle target column selection"""
        problem_type = self.session.get('problem_type')
        
        # Show current progress
        uploaded_tables = self.session.get('uploaded_tables', [])
        self.view.display_markdown("### 📊 Uploaded Data")
        for idx, table in enumerate(uploaded_tables, 1):
            self.view.display_markdown(f"{idx}. **{table['name']}** (Rows: {table['rows']:,}, Columns: {table['columns']})")
        self.view.display_markdown(f"\n**Selected Problem Type**: {problem_type.title()}")
        self.view.display_markdown("---")
        
        self.view.display_subheader("Target Column Configuration")
        
        # First ask if target column exists
        has_target = self.session.get('has_target_column')
        if has_target is None:
            self.view.display_markdown(
                "For this type of problem, we need to know about the target variable "
                "(the value we want to predict). Having the target column in your data "
                "is optional - we can help you create it later if needed."
            )
            
            has_target = self.view.radio_select(
                "Do you have a target column in your data?",
                options=["Yes", "No"],
                key="has_target_radio"
            )
            
            if self.view.display_button("Confirm"):
                self.session.set('has_target_column', has_target == "Yes")
                if has_target == "No":
                    self.session.set('target_column', None)
                    self.session.set('target_column_complete', True)
                self.view.rerun_script()
            return False
        
        # If no target column, we're done
        if has_target == "No":
            self.session.set('target_column_complete', True)
            return True
        
        # If has target, show column selection
        self.view.display_markdown("Please select the column you want to predict:")
        
        # Get all columns from all tables
        all_columns = []
        for table in uploaded_tables:
            df = pd.read_csv(table['path']) if table['name'].endswith('.csv') \
                else pd.read_excel(table['path'])
            all_columns.extend([f"{table['name']}: {col}" for col in df.columns])
        
        # Let user select target column
        selected_column = self.view.select_box(
            "Select target column:",
            options=[""] + all_columns,
            key="target_column"
        )
        
        if self.view.display_button("Confirm Target Column"):
            if not selected_column:
                self.view.show_message("Please select a target column", "error")
                return False
            
            self.session.set('target_column', selected_column)
            self.session.set('target_column_complete', True)
            return True
        
        return False
        
    def _generate_state_summary(self) -> bool:
        """Generate and save state summary"""
        try:
            # Get all required information
            uploaded_tables = self.session.get('uploaded_tables', [])
            problem_type = self.session.get('problem_type')
            target_column = self.session.get('target_column')
            
            # Create summary dictionary with more details
            summary = {
                'num_tables': str(len(uploaded_tables)),
                'problem_type': problem_type,
                'target_column': target_column if target_column else "None",
                'has_target': str(bool(target_column)),
                'table_names': json.dumps([table['name'] for table in uploaded_tables]),
                'table_rows': json.dumps([table['rows'] for table in uploaded_tables]),
                'table_columns': json.dumps([table['columns'] for table in uploaded_tables]),
                'completion_time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save summary to database
            session_id = self.session.get('session_id')
            if not self.db.save_state_summary('onboarding', summary, session_id):
                return False
            
            # Save each uploaded table to database
            if not self.db.save_uploaded_tables(session_id, uploaded_tables):
                self.view.show_message("❌ Error saving tables to database", "error")
                return False
            
            self.session.set('summary_complete', True)
            return True
            
        except Exception as e:
            self.view.show_message(f"Error generating summary: {str(e)}", "error")
            return False
        
    def _save_tables_to_db(self, tables: List[Dict]) -> bool:
        """Save uploaded tables to database"""
        session_id = self.session.get('session_id')
        return self.db.save_uploaded_tables(session_id, tables)
        
    def _save_summary_to_db(self, summary: Dict) -> bool:
        """Save state summary to database"""
        session_id = self.session.get('session_id')
        return self.db.save_state_summary(session_id, summary) 