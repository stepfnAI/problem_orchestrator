from sfn_blueprint import Task, SFNValidateAndRetryAgent
from regression_agent.agents.data_splitting_agent import SFNDataSplittingAgent
from orchestrator.config.model_config import DEFAULT_LLM_PROVIDER
import pandas as pd
import numpy as np

class DataSplitting:
    def __init__(self, session_manager, view, validation_window=3):
        self.session = session_manager
        self.view = view
        self.validation_window = validation_window
        self.splitting_agent = SFNDataSplittingAgent(validation_window=validation_window)
        
    def execute(self):
        """Orchestrates the data splitting step"""
        # Check if we already have split info
        existing_split_info = self.session.get('split_info')
        splitting_started = self.session.get('splitting_started', False)
        is_forecasting = self.session.get('is_forecasting', False)
        
        # If we have split info, show it and handle confirmation
        if existing_split_info:
            self._display_split_info(existing_split_info)
            if self.view.display_button("Proceed to Model Training", key="confirm_split"):
                self._save_step_summary(existing_split_info)
                self.session.set('step_4_complete', True)
                return True
            return False

        # If splitting hasn't started, show instructions input
        if not splitting_started:
            self.view.display_markdown("🤔 **Custom Splitting Instructions (Optional)**")
            user_instructions = self.view.text_area(
                "Add any specific instructions for data splitting (leave empty for default strategy):",
                help="Examples:\n- 'Use last 2 months for inference'\n- 'Split chronologically with 70-20-10 ratio'\n- 'Put all records after 2023 in inference set'",
                key="splitting_instructions"
            )

            if not self.view.display_button("Begin Splitting", key="start_split"):
                return False

            # Mark splitting as started and proceed
            self.session.set('splitting_started', True)
            
            try:
                # Get data and perform split
                df = self.session.get('df')
                mappings = self.session.get('field_mappings')
                date_col = mappings.get('date')
                
                with self.view.display_spinner('🤖 AI is determining optimal split...'):
                    task = Task("Split data", data={
                        'df': df, 
                        'date_column': date_col,
                        'validation_window': self.validation_window,
                        'field_mappings': mappings,
                        'user_instructions': user_instructions
                    })
                    validation_task = Task("Validate data splitting", data={
                        'df': df, 
                        'date_column': date_col,
                        'mappings': mappings
                    })
                    
                    validate_and_retry_agent = SFNValidateAndRetryAgent(
                        llm_provider=DEFAULT_LLM_PROVIDER,
                        for_agent='data_splitter'
                    )
                    
                    split_info, validation_message, is_valid = validate_and_retry_agent.complete(
                        agent_to_validate=self.splitting_agent,
                        task=task,
                        validation_task=validation_task,
                        method_name='execute_task',
                        get_validation_params='get_validation_params',
                        max_retries=2,
                        retry_delay=3.0
                    )
                    
                    if not is_valid:
                        self.view.show_message("❌ AI couldn't validate data splitting.", "error")
                        self.view.show_message(validation_message, "warning")
                        self.session.set('splitting_started', False)
                        return False
                
                # Aggregate data if forecasting
                if is_forecasting:
                    split_info = self._aggregate_forecasting_data(split_info)
                    if split_info is None:
                        self.view.show_message("❌ Error aggregating forecasting data.", "error")
                        return False
                
                # Save and display results outside spinner
                self.session.set('split_info', split_info)
                self._display_split_info(split_info)
                
                # Show confirmation button
                if self.view.display_button("Proceed to Model Training", key="confirm_split"):
                    self._save_step_summary(split_info)
                    self.session.set('step_4_complete', True)
                    return True
                
            except Exception as e:
                self.view.show_message(f"❌ Error during splitting: {str(e)}", "error")
                self.session.set('splitting_started', False)
                return False
        
        return False
        
    def _get_split_info(self):
        """Get split information"""
        # Check if we already have split info
        existing_split_info = self.session.get('split_info')
        splitting_started = self.session.get('splitting_started', False)
        
        # If we have split info, just display it without confirmation button
        if existing_split_info:
            self._display_split_info(existing_split_info)
            return existing_split_info  # Just return the info, no confirmation here

        # If splitting hasn't started, show instructions input
        if not splitting_started:
            self.view.display_markdown("🤔 **Custom Splitting Instructions (Optional)**")
            user_instructions = self.view.text_area(
                "Add any specific instructions for data splitting (leave empty for default strategy):",
                help="Examples:\n- 'Use last 2 months for inference'\n- 'Split chronologically with 70-20-10 ratio'\n- 'Put all records after 2023 in inference set'",
                key="splitting_instructions"
            )

            if not self.view.display_button("Begin Splitting", key="start_split"):
                return None

            # Mark splitting as started and proceed
            self.session.set('splitting_started', True)
            
            # Get data and perform split
            df = self.session.get('df')
            mappings = self.session.get('field_mappings')
            date_col = mappings.get('date')
            
            with self.view.display_spinner('🤖 AI is determining optimal split...'):
                task = Task("Split data", data={
                    'df': df, 
                    'date_column': date_col,
                    'validation_window': self.validation_window,
                    'field_mappings': mappings,
                    'user_instructions': user_instructions
                })
                validation_task = Task("Validate data splitting", data={
                    'df': df, 
                    'date_column': date_col,
                    'mappings': mappings
                })
                
                validate_and_retry_agent = SFNValidateAndRetryAgent(
                    llm_provider=DEFAULT_LLM_PROVIDER,
                    for_agent='data_splitter'
                )
                
                split_info, validation_message, is_valid = validate_and_retry_agent.complete(
                    agent_to_validate=self.splitting_agent,
                    task=task,
                    validation_task=validation_task,
                    method_name='execute_task',
                    get_validation_params='get_validation_params',
                    max_retries=2,
                    retry_delay=3.0
                )
                
                if not is_valid:
                    self.view.show_message("❌ AI couldn't validate data splitting.", "error")
                    self.view.show_message(validation_message, "warning")
                    self.session.set('splitting_started', False)
                    return None
                
                return split_info

        # If splitting started but no split info yet, show spinner
        self.view.display_spinner('Processing split...')
        return False
        
    def _display_split_info(self, split_info):
        """Display data split information"""
        self.view.display_subheader("Data Split Information")
        
        # Display AI's explanation if available
        if 'explanation' in split_info:
            self.view.display_markdown("🎯 **AI's Splitting Strategy:**")
            self.view.show_message(split_info['explanation'], "info")
            self.view.display_markdown("---")
        
        # Only display date ranges if date information is available
        if split_info['train_start'] is not None:
            self.view.display_markdown("**Training Period:**")
            self.view.display_markdown(f"- Start: {split_info['train_start']}")
            self.view.display_markdown(f"- End: {split_info['train_end']}")
            
            self.view.display_markdown("**Validation Period:**")
            self.view.display_markdown(f"- Start: {split_info['valid_start']}")
            self.view.display_markdown(f"- End: {split_info['valid_end']}")
            
            self.view.display_markdown("**Inference Period:**")
            self.view.display_markdown(f"- Month: {split_info['infer_month']}")
        
        # Display sample counts (always show this)
        self.view.display_markdown("\n**Sample Counts:**")
        self.view.display_markdown(f"- Training: **{split_info['train_samples']}**")
        self.view.display_markdown(f"- Validation: **{split_info['valid_samples']}**")
        self.view.display_markdown(f"- Inference: **{split_info['infer_samples']}**")
        
    def _save_step_summary(self, split_info):
        """Save step summary for display in completed steps"""
        summary = "✅ Data Splitting Complete:\n"
        # Add date range info only if available
        if split_info['train_start'] is not None:
            summary += f"- Period: {split_info['train_start']} to {split_info['train_end']}\n"
        summary += f"- Training samples: **{split_info['train_samples']}**\n"
        summary += f"- Validation samples: **{split_info['valid_samples']}**\n"
        summary += f"- Inference samples: **{split_info['infer_samples']}**"
        self.session.set('step_4_summary', summary)

    def _aggregate_forecasting_data(self, split_info):
        """Aggregate data by date for forecasting"""
        if not split_info:
            return None
        
        # Get date and target columns
        date_col = self.session.get('field_mappings', {}).get('date')
        target_col = self.session.get('field_mappings', {}).get('forecasting_field')
        
        if not date_col or not target_col:
            return split_info
        
        # Process each split if it exists
        for split in ['train_df', 'valid_df', 'infer_df']:
            if split in split_info and split_info[split] is not None:
                df = split_info[split]
                
                # Convert date column
                df[date_col] = pd.to_datetime(df[date_col])
                
                # Group by date and aggregate
                agg_df = df.groupby(date_col).agg({
                    target_col: 'sum',
                    # Add other numeric columns with mean aggregation
                    **{col: 'mean' for col in df.select_dtypes(include=[np.number]).columns 
                       if col != target_col}
                }).reset_index()
                
                # Update the split info
                split_info[split] = agg_df
                
        return split_info 