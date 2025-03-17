import pandas as pd
import logging
from typing import Dict, Any, Optional
from sfn_blueprint import Task
from orchestrator.states.base_state import BaseState
from orchestrator.agents.target_generator_agent import SFNTargetGeneratorAgent
from orchestrator.storage.db_connector import DatabaseConnector

logger = logging.getLogger(__name__)

class TargetGenerationState(BaseState):
    """State for generating target columns based on user instructions"""
    
    def __init__(self, session_manager, view):
        self.session = session_manager
        self.view = view
        self.db = DatabaseConnector()
        self.target_agent = SFNTargetGeneratorAgent()
        
    def execute(self) -> bool:
        """Execute the target generation state logic"""
        # Check if state is already complete
        if self.session.get('target_generation_complete', False):
            print("DEBUG: Target generation already complete, skipping")
            # Double check the value
            print(f"DEBUG: target_generation_complete = {self.session.get('target_generation_complete')}")
            
            # CRITICAL: Verify the target column exists in the dataframe
            df = self.session.get('df')
            mappings = self.session.get('field_mappings', {})
            target_column = mappings.get('target')
            
            if df is not None and target_column and target_column not in df.columns:
                print(f"ERROR: Target column '{target_column}' not found in dataframe in execute")
                # Try to recover by getting the dataframe with target
                df_with_target = self.session.get('df_with_target')
                if df_with_target is not None and 'target' in df_with_target.columns:
                    print("DEBUG: Recovering by using df_with_target")
                    # Update the main dataframe with the one containing the target
                    self.session.set('df', df_with_target)
                    
                    # Verify the dataframe was saved correctly
                    saved_df = self.session.get('df')
                    print(f"DEBUG: After recovery, target column exists: {'target' in saved_df.columns if saved_df is not None else 'No dataframe'}")
            
            self._show_state_summary()
            return True
            
        # Get the dataframe
        df = self.session.get('df')
        if df is None:
            self.view.show_message("❌ No dataframe found in session", "error")
            return False
            
        # Get problem type
        problem_type = self.session.get('next_state', 'unknown')
        if problem_type not in ['classification', 'regression']:
            # Target generation only applies to these problem types
            self.session.set('target_generation_complete', True)
            return True
            
        # Proceed with target generation
        result = self._generate_target(df, problem_type)
        
        # Only return True if target generation is complete
        is_complete = self.session.get('target_generation_complete', False)
        print(f"DEBUG: After _generate_target, target_generation_complete = {is_complete}")
        return is_complete
            
    def _generate_target(self, df: pd.DataFrame, problem_type: str) -> bool:
        """Generate a target column based on user instructions"""
        # Add debug print to track execution flow
        print(f"DEBUG: Entering _generate_target method. Initialized: {self.session.get('target_generation_initialized')}")
        
        # Initialize target generation if not already done
        if not self.session.get('target_generation_initialized'):
            # More debug prints
            print("DEBUG: Target generation not initialized, showing input form")
            
            self.view.display_markdown("## Target Generation")
            self.view.display_markdown(
                "Let's create a target column for your machine learning model. "
                "Please provide instructions on how to create the target."
            )
            
            # Show a preview of the joined data
            self.view.display_markdown("### Preview of Your Data")
            self.view.display_markdown(
                "Here's a preview of your data to help you create the target column. "
                "You can reference any of these columns in your instructions."
            )
            
            self.view.display_dataframe(df.head(5))
            
            # Get suggested target instructions based on data and problem type
            suggested_instructions = self._get_target_suggestions(df, problem_type)
            
            self.view.display_markdown("### Target Creation Instructions")
            self.view.display_markdown(
                "Now, please provide instructions for creating your target column. "
                "Below are some suggestions based on your data:"
            )
            
            # Display the suggested instructions
            self.view.display_markdown(f"**Suggested approaches:**")
            for i, suggestion in enumerate(suggested_instructions):
                self.view.display_markdown(f"{i+1}. {suggestion}")
            
            # Get user instructions
            user_instructions = self.view.text_area(
                "Instructions for target creation",
                key="target_instructions"
            )
            
            # Add a button to begin target preparation
            if not self.view.display_button("Begin Target Preparation with this Approach", key="begin_target_prep"):
                # If button not clicked, return False to keep the UI state
                return False
                
            if not user_instructions:
                # If no instructions provided, use the first suggestion
                user_instructions = suggested_instructions[0]
                self.view.show_message(f"Using suggested approach: {user_instructions}", "info")
                
            # Button was clicked and instructions are provided
            # Store instructions and mark as initialized 
            self.view.show_message("Starting target generation...", "info")
            self.session.set('target_generation_instructions', user_instructions)
            self.session.set('target_generation_initialized', True)
            
        # Get stored instructions
        user_instructions = self.session.get('target_generation_instructions', '')
        print(f"DEBUG: Using instructions: {user_instructions}")
        
        # Get previous error message if any
        error_message = self.session.get('target_generation_error', '')
        
        # Check if we already have a generated result
        result = self.session.get('target_generation_result')
        print(f"DEBUG: Existing result found: {result is not None}")
        
        # Check if target has been prepared
        target_prepared = self.session.get('target_prepared', False)
        
        # If target is prepared, show the preview and proceed button
        if target_prepared:
            return self._show_prepared_target(df, problem_type)
        
        # Generate target code if we don't have a result yet
        if result is None:
            print("DEBUG: Generating new target code")
            self.view.display_spinner("Generating target column...")
        
        task_data = {
            'user_instructions': user_instructions,
            'df': df,
            'problem_type': problem_type,
            'error_message': error_message
        }
        task = Task("Generate Target", data=task_data)
        
        # Execute task
        result = self.target_agent.execute_task(task)
        
        # Store the result
        self.session.set('target_generation_result', result)
        
        # Display the generated code and explanation
        self.view.display_markdown("### Generated Code")
        self.view.display_markdown(f"```python\n{result['code']}\n```")
        
        self.view.display_markdown("### Explanation")
        self.view.display_markdown(result['explanation'])
        
        self.view.display_markdown("### Preview")
        self.view.display_markdown(result['preview'])
        
        # Create columns for buttons side by side
        self.view.display_markdown("### What would you like to do?")
        col1, col2 = self.view.create_columns([1, 1])
        
        # Use separate variables for each button
        with col1:
            approve_button = self.view.display_button("Approve & Prepare Target", key="approve_target")
            
        with col2:
            retry_button = self.view.display_button("Try Again", key="retry_target")
        
        # Add debug prints to see button states
        print(f"DEBUG: approve_button={approve_button}, retry_button={retry_button}")
        
        # Handle retry button first, with explicit check
        if retry_button:
            print("DEBUG: Retry button clicked, resetting state")
            # User wants to try again, reset initialization and clear previous results
            self.session.set('target_generation_initialized', False)
            self.session.set('target_generation_complete', False)
            self.session.set('target_generation_error', '')
            self.session.set('target_generation_result', None)
            self.session.set('target_generation_instructions', '')
            
            # Force a complete rerun of the app to clear the UI
            print("DEBUG: Forcing UI refresh with rerun()")
            self.view.rerun_script()
            
            # This return won't be reached due to rerun, but included for completeness
            return False
        # Only check approve button if retry wasn't clicked
        elif approve_button:
            print("DEBUG: Approve button clicked, proceeding to execution")
            # User approves the target, proceed with execution
        return self._execute_target_code(df, result, problem_type, user_instructions)
        
        # If no choice was made yet (neither approve nor retry), stay on this page
        print("DEBUG: No button clicked yet, staying on page")
        return False
    
    def _get_target_suggestions(self, df: pd.DataFrame, problem_type: str) -> list:
        """
        Get suggested target creation instructions based on the data and problem type.
        This will be replaced with an LLM-based suggestion in the future.
        
        Args:
            df: The dataframe containing the data
            problem_type: The type of problem (classification or regression)
            
        Returns:
            list: A list of suggested target creation instructions
        """
        # Get column names for reference
        columns = list(df.columns)
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # For now, use hardcoded suggestions based on problem type
        # In the future, this will be replaced with an LLM call
        
        if problem_type == 'classification':
            suggestions = [
                f"Create a binary target column where rows with {numeric_columns[0] if numeric_columns else 'value'} > {df[numeric_columns[0]].median() if numeric_columns else 0} are labeled as 1 and others as 0.",
                f"Create a target column based on the combination of {categorical_columns[0] if categorical_columns else 'category'} and {numeric_columns[0] if numeric_columns else 'value'}, where specific combinations are labeled as 1 and others as 0.",
                f"Create a multi-class target column by binning {numeric_columns[0] if numeric_columns else 'value'} into 'low', 'medium', and 'high' categories."
            ]
        else:  # regression
            suggestions = [
                f"Create a target column that is the ratio of {numeric_columns[0] if len(numeric_columns) > 0 else 'value1'} to {numeric_columns[1] if len(numeric_columns) > 1 else 'value2'}.",
                f"Create a target column that is the log transformation of {numeric_columns[0] if numeric_columns else 'value'}.",
                f"Create a target column that combines multiple features: {', '.join(numeric_columns[:3]) if len(numeric_columns) >= 3 else 'available numeric columns'} using a weighted sum."
            ]
            
        return suggestions
        
    def _execute_target_code(self, df: pd.DataFrame, result: Dict[str, Any], 
                            problem_type: str, user_instructions: str) -> bool:
        """Execute the generated target code and apply it to the full dataset"""
        self.view.display_spinner("Preparing target column...")
        
        try:
            # Create a copy of the dataframe
            df_copy = df.copy()
            print(f"DEBUG: Original dataframe shape before execution: {df_copy.shape}")
            
            # Execute the code with proper imports, making sure 'df' refers to our dataframe
            # and not any sample data the LLM might have created
            import numpy as np
            import pandas as pd
            exec_locals = {'df': df_copy, 'pd': pd, 'np': __import__('numpy')}
            exec(result['code'], {}, exec_locals)
            df_with_target = exec_locals['df']
            
            # Check if target column was created
            if 'target' not in df_with_target.columns:
                print("DEBUG: Target column not found after execution")
                raise ValueError("No 'target' column was created by the code")
                
            # Store the dataframe with target temporarily
            self.session.set('df_with_target', df_with_target)
            
            # Mark target as prepared but not yet complete
            self.session.set('target_prepared', True)
            self.view.show_message("✅ Target column prepared successfully", "success")
            
            # Force a refresh to show the prepared target view
            self.view.rerun_script()
            
            return False
            
        except Exception as e:
            # Execution failed, show error
            error_message = str(e)
            self.view.show_message(f"❌ Error executing code: {error_message}", "error")
            
            # Store the error
            self.session.set('target_generation_error', error_message)
            
            # Ask user if they want to try again
            self.view.display_markdown("Would you like to try again with new instructions?")
            if self.view.display_button("Try again", key="try_again_error"):
                print("DEBUG: Try again button clicked after error, resetting state")
                # Reset initialization and clear previous results
                self.session.set('target_generation_initialized', False)
                self.session.set('target_generation_complete', False)
                self.session.set('target_generation_error', '')
                self.session.set('target_generation_result', None)
                self.session.set('target_generation_instructions', '')
                
                # Force a complete rerun of the app to clear the UI
                print("DEBUG: Forcing UI refresh with rerun()")
                self.view.rerun_script()
                
                # This return won't be reached due to rerun, but included for completeness
                return False
                
        # Default return value if we somehow get here
        return False
                
    def _save_target_generation(self, instructions: str, code: str, explanation: str):
        """Save target generation details to database"""
        try:
            session_id = self.session.get('session_id')
            
            # Save to database
            self.db.save_target_generation(
                session_id=session_id,
                instructions=instructions,
                code=code,
                explanation=explanation
            )
            
        except Exception as e:
            logger.error(f"Error saving target generation: {str(e)}")
            self.view.show_message(f"Warning: Could not save target generation details: {str(e)}", "warning")
            
    def _show_state_summary(self):
        """Show a summary of the target generation state"""
        mappings = self.session.get('field_mappings', {})
        df = self.session.get('df')
        target_column = mappings.get('target', 'No target column')
        
        # CRITICAL: Verify the target column exists in the dataframe
        if df is not None and target_column != 'No target column' and target_column not in df.columns:
            print(f"ERROR: Target column '{target_column}' not found in dataframe in _show_state_summary")
            # Try to recover by getting the dataframe with target
            df_with_target = self.session.get('df_with_target')
            if df_with_target is not None and 'target' in df_with_target.columns:
                print("DEBUG: Recovering by using df_with_target")
                # Update the main dataframe with the one containing the target
                self.session.set('df', df_with_target)
                df = df_with_target
        
        self.view.display_markdown("## Target Preparation Summary")
        self.view.display_markdown(f"Target column: `{target_column}` has been successfully created.")
        
        # If we have a generated target, show its explanation
        result = self.session.get('target_generation_result', {})
        if result.get('explanation'):
            self.view.display_markdown(f"**Description:** {result['explanation']}")
            
        # Show sample of the dataframe with target
        if df is not None and target_column in df.columns:
            self.view.display_markdown("### Sample Data with Target Column")
            self.view.display_markdown("Here's a preview of your data with the target column:")
            self.view.display_dataframe(df.head(10))  # Show only a sample for display
            
            # Show target distribution
            self.view.display_markdown("### Target Distribution")
            problem_type = self.session.get('next_state', 'unknown')
            if problem_type in ['classification', 'forecasting']:
                # For classification, show value counts
                value_counts = df[target_column].value_counts()
                self.view.display_dataframe(value_counts.reset_index().rename(
                    columns={'index': 'Value', target_column: 'Count'}
                ))
            else:
                # For regression, show descriptive statistics
                self.view.display_markdown(str(df[target_column].describe()))
                
            # Add download button for the dataframe
            self.view.display_markdown("### Download Complete Dataset")
            self.view.display_markdown(f"The complete dataset contains **{len(df)} rows** with **{len(df.columns)} columns** including the target column.")
            
            # Debug info to verify we're exporting the full dataframe
            print(f"DEBUG: Exporting dataframe with shape: {df.shape}")
            
            # Convert the entire dataframe to CSV string
            csv_data = df.to_csv(index=False)
            
            # Create download button with the full CSV data
            self.view.create_download_button(
                "Download Data with Target Column", 
                csv_data, 
                "data_with_target.csv", 
                "text/csv"
            )
        else:
            print(f"ERROR: Cannot show target column preview - target column '{target_column}' not found in dataframe")
            self.view.show_message(f"⚠️ Warning: Target column '{target_column}' not found in dataframe", "warning")
            
    def _show_prepared_target(self, df: pd.DataFrame, problem_type: str) -> bool:
        """Show the prepared target and allow user to proceed or go back"""
        # Get the dataframe with target
        df_with_target = self.session.get('df_with_target')
        if df_with_target is None:
            self.view.show_message("❌ Prepared target not found", "error")
            return False
        
        # Debug print to verify target column exists
        print(f"DEBUG: Target column exists in df_with_target: {'target' in df_with_target.columns}")
        print(f"DEBUG: Columns in df_with_target: {list(df_with_target.columns)}")
        
        self.view.display_markdown("## Target Preview")
        self.view.display_markdown("Your target column has been prepared. Here's a preview:")
        
        # Show preview of the dataframe with target
        self.view.display_dataframe(df_with_target.head(10))
        
        # Show target distribution
        self.view.display_markdown("### Target Distribution")
        if problem_type in ['classification', 'forecasting']:
            # For classification, show value counts
            self.view.display_dataframe(df_with_target['target'].value_counts())
        else:
            # For regression, show histogram
            self.view.display_markdown(str(df_with_target['target'].describe()))
        
        # Create columns for buttons
        self.view.display_markdown("### What would you like to do?")
        col1, col2 = self.view.create_columns([1, 1])
        
        with col1:
            proceed_button = self.view.display_button("Proceed to Next Step", key="proceed_target")
        
        with col2:
            go_back_button = self.view.display_button("Go Back to Target Prep", key="go_back_target")
        
        if proceed_button:
            # User wants to proceed, save the target and mark as complete
            
            # CRITICAL: Make sure the target column exists in the dataframe
            if 'target' not in df_with_target.columns:
                self.view.show_message("❌ Target column not found in prepared data", "error")
                return False
            
            # Update the main dataframe with the one containing the target
            print(f"DEBUG: Saving dataframe with target. Shape: {df_with_target.shape}")
            print(f"DEBUG: Target column exists before saving: {'target' in df_with_target.columns}")
            self.session.set('df', df_with_target)
            
            # Verify the dataframe was saved correctly
            saved_df = self.session.get('df')
            print(f"DEBUG: After saving, target column exists: {'target' in saved_df.columns if saved_df is not None else 'No dataframe'}")
            print(f"DEBUG: After saving, columns in df: {list(saved_df.columns) if saved_df is not None else 'No dataframe'}")
            
            # Update field mappings to include the target
            mappings = self.session.get('field_mappings', {})
            mappings['target'] = 'target'
            self.session.set('field_mappings', mappings)
            print(f"DEBUG: Updated field_mappings: {mappings}")
            
            # Save the target generation details
            result = self.session.get('target_generation_result', {})
            user_instructions = self.session.get('target_generation_instructions', '')
            self._save_target_generation(user_instructions, result.get('code', ''), result.get('explanation', ''))
            
            # Mark as complete and proceed
            self.session.set('target_generation_complete', True)
            self.view.show_message("✅ Target column saved successfully", "success")
            
            # Show the target summary before proceeding
            self._show_state_summary()
            return True
        
        elif go_back_button:
            # User wants to go back to target prep
            self.session.set('target_prepared', False)
            self.session.set('target_generation_initialized', False)
            self.session.set('target_generation_complete', False)
            self.session.set('target_generation_error', '')
            self.session.set('target_generation_result', None)
            self.session.set('target_generation_instructions', '')
            self.session.set('df_with_target', None)
            
            # Force a refresh to go back to the initial target prep view
            self.view.rerun_script()
            
            return False
        
        # If no choice was made yet, stay on this page
        return False 