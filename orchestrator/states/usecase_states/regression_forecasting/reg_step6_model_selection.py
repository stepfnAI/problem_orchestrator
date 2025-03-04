from sfn_blueprint import Task, SFNValidateAndRetryAgent
from regression_agent.agents.model_selection_agent import SFNModelSelectionAgent
from orchestrator.utils.reg_model_manager import ModelManager
from orchestrator.config.model_config import DEFAULT_LLM_PROVIDER
from typing import Dict

class ModelSelection:
    def __init__(self, session_manager, view):
        self.session = session_manager
        self.view = view
        self.selection_agent = SFNModelSelectionAgent()
        self.model_manager = ModelManager()
        
    def execute(self):
        """Execute model selection step"""
        # try:
        # Get model results
        model_results = self.session.get('model_results')
        if not model_results:
            self.view.show_message("❌ No model results found. Please complete training first.", "error")
            return False
        
        # Get analysis type
        is_forecasting = self.session.get('is_forecasting', False)
        
        # If we don't have a recommendation yet, get one
        if not self.session.get('model_recommendation'):
            with self.view.display_spinner('🤖 AI is determining the best model...'):
                validate_and_retry_agent = SFNValidateAndRetryAgent(
                    llm_provider=DEFAULT_LLM_PROVIDER,
                    for_agent='model_selector'
                )
            
            # Format model results first
            selection_info = self._format_model_results(model_results, is_forecasting)

            task = Task("Select best model", data={
                'selection_info': selection_info,
                'custom_instructions': ""
            })

            result, validation_message, is_valid = validate_and_retry_agent.complete(
            agent_to_validate=self.selection_agent,
            task=task,
            validation_task=task,
            method_name='execute_task',
            get_validation_params='get_validation_params',
            max_retries=2,
            retry_delay=3.0
            )

            if not is_valid:
                self.view.show_message(f"❌ Model selection failed: {validation_message}", "error")
                return False
            
            self.session.set('model_recommendation', result)
            self.session.set('model_recommended', True)
        
        # Get the recommendation from session
        recommendation = self.session.get('model_recommendation')
        
        # Always display recommendation if we have it
        if recommendation:
            self._display_recommendation(recommendation)
        
        # If we haven't completed selection yet, show selection options
        if not self.session.get('step_6_complete'):
            # Get user confirmation
            selected_model = self._get_user_selection(model_results, recommendation['selected_model'])
            if selected_model:
                # Save the selected model info
                self.session.set('selected_model', model_results[selected_model])
                self.session.set('selection_mode', 'ai' if selected_model == recommendation['selected_model'] else 'manual')
                self._save_step_summary()
                self.session.set('step_6_complete', True)
                return True
            return False
        
        # If we've already completed, show summary
        summary = self.session.get('step_6_summary')
        if summary:
            self.view.show_message(summary, "success")
        
        return True
            
        # except Exception as e:
        #     self.view.show_message(f"Error in model selection: {str(e)}", "error")
        #     return False
            
    def _format_model_results(self, model_results: Dict, is_forecasting: bool) -> str:
        """Format model results for LLM consumption"""
        formatted_info = "Model Performance Summary:\n\n"
        
        for model_name, results in model_results.items():
            metrics = results.get('metrics', {})
            formatted_info += f"Model: {model_name}\n"
            
            if is_forecasting:
                formatted_info += f"- MAPE: {metrics.get('mape', 'N/A')}\n"
                formatted_info += f"- RMSE: {metrics.get('rmse', 'N/A')}\n"
                formatted_info += f"- MAE: {metrics.get('mae', 'N/A')}\n"
            else:
                formatted_info += f"- R² Score: {metrics.get('r2', 'N/A')}\n"
                formatted_info += f"- MSE: {metrics.get('mse', 'N/A')}\n"
                formatted_info += f"- MAE: {metrics.get('mae', 'N/A')}\n"
            
            # Add records info
            records = results.get('records_info', {})
            formatted_info += f"- Training Records: {records.get('used_train', 0)}/{records.get('total_train', 0)}\n"
            formatted_info += f"- Validation Records: {records.get('used_valid', 0)}/{records.get('total_valid', 0)}\n\n"
            
        return formatted_info
            
    def _display_recommendation(self, recommendation):
        """Display model recommendation and explanation"""
        self.view.display_subheader("🤖 Model Recommendation")
        
        # Show selected model
        self.view.show_message(f"**Recommended Model:** {recommendation['selected_model']}", "success")
        
        # Show explanation
        self.view.show_message("**Explanation:**\n" + recommendation['explanation'], "info")
        
        # Show comparison summary
        self.view.show_message("**Model Comparison:**\n" + recommendation['comparison_summary'], "info")

    def _get_user_selection(self, model_results, recommended_model):
        """Get user confirmation or alternative selection"""
        self.view.display_subheader("Model Selection")
        
        # First let user choose the selection mode
        selection_mode = self.view.display_radio(
            "How would you like to select the model?",
            options=["Proceed with AI Recommendation", "Select Model Manually"],
            key="selection_mode"
        )
        
        if selection_mode == "Proceed with AI Recommendation":
            if self.view.display_button("✅ Confirm AI Recommendation", key="confirm_ai"):
                return recommended_model
        else:
            # If manual selection, show all models
            selected = self.view.display_radio(
                "Select model to use:",
                options=list(model_results.keys()),
                key="model_selection"
            )
            
            if self.view.display_button("✅ Confirm Selection", key="confirm_manual"):
                if selected:
                    return selected
                else:
                    self.view.show_message("⚠️ Please select a model first", "warning")
                    return None
        
        return None

    def _save_step_summary(self):
        """Save step summary for display in completed steps"""
        selected_model = self.session.get('selected_model')
        selection_mode = self.session.get('selection_mode')
        
        if not selected_model:  # Safety check
            return
        
        summary = "✅ Model Selection Complete:\n"
        summary += f"- Selected Model: **{selected_model.get('model_name', 'Unknown')}**\n"
        
        if selection_mode == 'ai':
            summary += "- Using AI recommended model\n"
        else:
            summary += "- Using manually selected model\n"
        
        self.session.set('step_6_summary', summary)

    def _get_model_recommendation(self, model_results: Dict, is_forecasting: bool) -> Dict:
        """Get model recommendation from selection agent"""
        try:
            # Format model results for selection
            selection_info = self._format_model_results(model_results, is_forecasting)
            
            # Initialize selection agent with appropriate type
            selection_agent = SFNModelSelectionAgent(
                analysis_type="forecasting" if is_forecasting else "regression"
            )
            
            # Get model recommendation
            task_data = {
                'selection_info': selection_info,
                'custom_instructions': ""
            }
            
            recommendation = selection_agent.execute_task(
                Task("Select best model", data=task_data)
            )
            
            if not recommendation or 'selected_model' not in recommendation:
                return None
            
            return {
                'selected_model': recommendation['selected_model'],
                'explanation': recommendation.get('explanation', ''),
                'comparison_summary': recommendation.get('comparison_summary', ''),
                'model_rankings': recommendation.get('model_rankings', [])
            }
            
        except Exception as e:
            print(f"Error getting model recommendation: {str(e)}")
            return None 