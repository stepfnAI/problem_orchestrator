from typing import Dict, Optional
import pandas as pd
from classification_agent.agents.data_splitting_agent import SFNDataSplittingAgent
import logging
from sfn_blueprint import Task

logger = logging.getLogger(__name__)

class Step6DataSplitting:
    def __init__(self, session_manager, view):
        """Initialize Step6DataSplitting with session manager and view"""
        self.session = session_manager
        self.view = view
        self.splitting_agent = SFNDataSplittingAgent(llm_provider='openai')

    def process_splitting(self, joined_df: pd.DataFrame) -> Optional[Dict]:
        """Process data splitting for the joined dataset"""
        # try:
        # Add debug print to verify input data
        print("DEBUG: Input dataframe shape:", joined_df.shape)

        # If splitting is already completed, return the stored results
        if self.session.get('splitting_completed'):
            return self.session.get('splitting_results')

        # Get field mappings from previous steps
        # field_mappings = self.session.get('mapping_info', {})
        field_mappings = {'target': 'target',
                          'date':'BillingDate',
                          'cust_id':'CustomerID',
                          "prod_id": "ProductID"}
    
        # Execute splitting directly without instructions
        splitting_task = Task("Split data", data={
            'df': joined_df,
            'field_mappings': field_mappings,
            'user_instructions': ''  # Empty instructions
        })
        
        with self.view.display_spinner('🤖 AI is performing data splitting...'):
            split_results = self.splitting_agent.execute_task(splitting_task)

        # Display splitting results
        self._display_splitting_results(split_results)

        # Store results in session
        self.session.set('splitting_results', split_results)
        self.session.set('splitting_completed', True)

        # Show completion message and proceed button
        self.view.show_message("✅ Data splitting completed successfully!", "success")
        
        if self.view.display_button("Proceed to Model Training", key="proceed_to_model_training"):
            self.session.set('current_step', 7)  # Set next step before returning
            return split_results

        return None

        # except Exception as e:
        #     logger.error(f"Error in data splitting: {str(e)}")
        #     self.view.show_message(f"❌ Error in data splitting: {str(e)}", "error")
        #     return None

    def _display_splitting_results(self, results: Dict):
        """Display the results of data splitting"""
        self.view.display_subheader("Splitting Results")
        
        # Display split sizes
        self.view.display_markdown("### Dataset Sizes")
        self.view.show_message(f"""
        - Training set: {results['train_samples']} samples
        - Validation set: {results['valid_samples']} samples
        - Inference set: {results['infer_samples']} samples
        """)

        # Display date ranges if available
        if results.get('train_start'):
            self.view.display_markdown("### Date Ranges")
            self.view.show_message(f"""
            Training: {results['train_start']} to {results['train_end']}
            Validation: {results['valid_start']} to {results['valid_end']}
            Inference Month: {results['infer_month']}
            """)

        # Display explanation
        self.view.display_markdown("### Splitting Strategy")
        self.view.show_message(results['explanation'])
