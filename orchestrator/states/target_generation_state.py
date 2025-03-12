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
        self.target_generation_attempts = []
        
    def execute(self) -> bool:
        """Execute the target generation state logic"""
        # Check if state is already complete
        if self.session.get('target_generation_complete', False):
            print("DEBUG: Target generation already complete, skipping")
            # Double check the value
            print(f"DEBUG: target_generation_complete = {self.session.get('target_generation_complete')}")
            self._show_state_summary()
            return True
            
        # Get the dataframe
        df = self.session.get('df')
        if df is None:
            self.view.show_message("❌ No dataframe found in session", "error")
            return False
            
        # Get problem type
        problem_type = self.session.get('next_state', 'unknown')
        if problem_type not in ['classification', 'regression', 'forecasting']:
            # Target generation only applies to these problem types
            self.session.set('target_generation_complete', True)
            return True
            
        # Check if target already exists in mappings
        mappings = self.session.get('field_mappings', {})
        if mappings.get('target') and mappings.get('target') in df.columns:
            # Target already exists, ask if user wants to keep it or create a new one
            logger.info(f"Found existing target column: {mappings.get('target')}")
            return self._handle_existing_target(df, mappings, problem_type)
            
        # No target exists, proceed with target generation
        return self._generate_target(df, problem_type)
        
    def _handle_existing_target(self, df: pd.DataFrame, mappings: Dict, problem_type: str) -> bool:
        """Handle case where target already exists"""
        target_column = mappings.get('target')
        
        # Display the existing target information
        self.view.display_markdown(f"## Target Column")
        self.view.display_markdown(f"An existing target column `{target_column}` was found in your data.")
        
        # Show target statistics
        self.view.display_markdown("### Target Statistics")
        if problem_type in ['classification', 'forecasting']:
            # For classification, show value counts as a table
            value_counts = df[target_column].value_counts()
            self.view.display_dataframe(value_counts.reset_index().rename(
                columns={'index': 'Value', target_column: 'Count'}
            ))
        else:
            # For regression, show descriptive statistics
            stats = df[target_column].describe().reset_index().rename(
                columns={'index': 'Statistic', target_column: 'Value'}
            )
            self.view.display_dataframe(stats)
            
        # Ask user if they want to keep the existing target or create a new one
        options = ["Keep existing target", "Create new target"]
        choice = self.view.select_option("What would you like to do?", options)
        
        if choice == "Keep existing target":
            self.view.show_message(f"✅ Using existing target column: {target_column}", "success")
            self.session.set('target_generation_complete', True)
            return True
        else:
            # User wants to create a new target, proceed with generation
            return self._generate_target(df, problem_type)
            
    def _generate_target(self, df: pd.DataFrame, problem_type: str) -> bool:
        """Generate a target column based on user instructions"""
        # Initialize target generation if not already done
        if not self.session.get('target_generation_initialized'):
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
            
            self.view.display_markdown("### Target Creation Instructions")
            self.view.display_markdown(
                "Now, please provide instructions for creating your target column:"
            )
            
            # Show examples based on problem type
            if problem_type == 'classification':
                example = (
                    "Example: Create a target column where customers with income > 60000 "
                    "AND age > 30 are labeled as 'high_value' (1) and others as 'standard' (0)"
                )
            elif problem_type == 'regression':
                example = (
                    "Example: Create a target column that represents the ratio of income to age, "
                    "multiplied by the square root of purchase_count"
                )
            else:  # forecasting
                example = (
                    "Example: Create a target column that is the purchase_count value, "
                    "but shifted forward by 1 month to represent future purchases"
                )
                
            self.view.display_markdown(f"*{example}*")
            
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
                self.view.show_message("Please provide instructions for target creation", "warning")
                return False
                
            # Button was clicked and instructions are provided
            # Store instructions and mark as initialized 
            self.view.show_message("Starting target generation...", "info")
            self.session.set('target_generation_instructions', user_instructions)
            self.session.set('target_generation_initialized', True)
            
        # Get stored instructions
        user_instructions = self.session.get('target_generation_instructions', '')
        
        # Get previous error message if any
        error_message = self.session.get('target_generation_error', '')
        
        # Generate target code
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
        
        # Execute the code and show preview
        return self._execute_target_code(df, result, problem_type, user_instructions)
        
    def _execute_target_code(self, df: pd.DataFrame, result: Dict[str, Any], 
                            problem_type: str, user_instructions: str) -> bool:
        """Execute the generated target code and show preview"""
        self.view.display_markdown("### Execution Result")
        
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
                
            # Create columns for buttons side by side
            self.view.display_markdown("### What would you like to do?")
            col1, col2 = self.view.create_columns([1, 1])
            
            with col1:
                approve_button = self.view.display_button("Approve and use this target", key="approve_target")
                
            with col2:
                retry_button = self.view.display_button("Try again with new instructions", key="retry_target")
                
            # Determine choice based on button clicks
            choice = None
            if approve_button:
                choice = "Approve and use this target"
            elif retry_button:
                choice = "Try again with new instructions"
            
            if choice == "Approve and use this target":
                # User approves the target
                # Update the dataframe, mappings, and mark as complete
                self.session.set('df', df_with_target)
                
                # Update field mappings to include the target
                mappings = self.session.get('field_mappings', {})
                mappings['target'] = 'target'
                self.session.set('field_mappings', mappings)
                print(f"DEBUG: Updated field_mappings: {mappings}")
                
                # CRITICAL: Save the updated dataframe with target column back to the session
                print(f"DEBUG: Saving dataframe with target column back to session. Shape: {df_with_target.shape}")
                print(f"DEBUG: Target column exists in df_with_target: {'target' in df_with_target.columns}")
                print(f"DEBUG: Columns in df_with_target: {list(df_with_target.columns)}")
                self.session.set('df', df_with_target)
                
                # Verify the dataframe was saved correctly
                print(f"DEBUG: After saving, target column exists in session df: {'target' in self.session.get('df', pd.DataFrame()).columns}")
                
                # Save the target generation details
                self._save_target_generation(user_instructions, result['code'], result['explanation'])
                
                # Mark as complete and proceed
                self.session.set('target_generation_complete', True)
                self.view.show_message("✅ Target column created successfully", "success")
                
                # Show the target summary before proceeding
                self._show_state_summary()
                return True
                
            elif choice == "Try again with new instructions":
                # User wants to try again, reset initialization
                self.session.set('target_generation_initialized', False)
                self.session.set('target_generation_complete', False)  # Ensure it's not marked complete
                
                # Store this attempt for reference
                attempts = self.session.get('target_generation_attempts', [])
                attempts.append({
                    'instructions': user_instructions,
                    'code': result['code'],
                    'error': None
                })
                self.session.set('target_generation_attempts', attempts)
                
                # Clear error
                self.session.set('target_generation_error', '')
                
                return False
                
            # If no choice was made yet (neither approve nor retry), stay on this page
            return False
            
        except Exception as e:
            # Execution failed, show error
            error_message = str(e)
            self.view.show_message(f"❌ Error executing code: {error_message}", "error")
            
            # Store the error
            self.session.set('target_generation_error', error_message)
            
            # Store this attempt for reference
            attempts = self.session.get('target_generation_attempts', [])
            attempts.append({
                'instructions': user_instructions,
                'code': result['code'],
                'error': error_message
            })
            self.session.set('target_generation_attempts', attempts)
            
            # Ask user if they want to try again
            options = ["Try again with new instructions", "View previous attempts"]
            choice = self.view.select_option("What would you like to do?", options)
            
            if choice == "Try again with new instructions":
                # Reset initialization to get new instructions
                self.session.set('target_generation_initialized', False)
                return False
            else:
                # Show previous attempts
                self._show_previous_attempts()
                return False
                
        # Default return value if we somehow get here
        return False
        
    def _show_previous_attempts(self):
        """Show previous target generation attempts"""
        attempts = self.session.get('target_generation_attempts', [])
        
        if not attempts:
            self.view.display_markdown("No previous attempts found.")
            return
            
        self.view.display_markdown("## Previous Attempts")
        
        for i, attempt in enumerate(attempts):
            self.view.display_markdown(f"### Attempt {i+1}")
            self.view.display_markdown(f"**Instructions:** {attempt['instructions']}")
            
            self.view.display_markdown("**Code:**")
            self.view.display_markdown(f"```python\n{attempt['code']}\n```")
            
            if attempt['error']:
                self.view.display_markdown(f"**Error:** {attempt['error']}")
            else:
                self.view.display_markdown("**Status:** Executed successfully but not approved")
                
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