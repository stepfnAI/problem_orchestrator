from classification_agent.agents.model_selection_agent import SFNModelSelectionAgent
from sfn_blueprint import Task
import pandas as pd

class Step8ModelSelectionOrchestrator:
    def __init__(self, session_manager, view):
        self.model_selection_agent = SFNModelSelectionAgent()
        self.session = session_manager
        self.view = view

    def orchestrate(self):
        if not self.session.get('model_recommendation_complete'):
            self._get_model_recommendation()
            return

        self._show_model_selection_ui()

    def _get_model_recommendation(self):
        self.view.display_subheader("AI Model Recommendation")
        
        # Get model results from step 7
        step7_output = self.session.get('step7_output')
        if not step7_output or 'model_results' not in step7_output:
            self.view.show_message("❌ Model training results not found. Please complete model training first.", "error")
            return

        with self.view.display_spinner("🤖 AI is analyzing model performance..."):
            # Pass the simplified model results structure
            task = Task("model_selection", data={
                'model_results': step7_output['model_results']  # Already in correct format: {model_name: {'name': name, 'metrics': metrics}}
            })
            
            recommendation = self.model_selection_agent.execute_task(task)
            
            # Store recommendation with metrics from original results
            recommended_model_name = recommendation.get('selected_model')
            if recommended_model_name:
                model_metrics = step7_output['model_results'].get(recommended_model_name, {}).get('metrics', {})
                recommendation['metrics'] = model_metrics
            
            self.session.set('model_recommendation', recommendation)
            self.session.set('model_recommendation_complete', True)
            self.view.rerun_script()

    def _show_model_selection_ui(self):
        recommendation = self.session.get('model_recommendation')
        if not recommendation:
            return

        # Get model results from step 7
        step7_output = self.session.get('step7_output')
        if not step7_output or 'model_results' not in step7_output:
            return

        # Display AI recommendation
        self.view.display_subheader("🤖 AI Model Recommendation")
        recommended_model = recommendation.get('selected_model')
        explanation = recommendation.get('explanation')
        
        # Get metrics for recommended model from step7_output
        model_metrics = step7_output['model_results'].get(recommended_model, {}).get('metrics', {})
        
        recommendation_msg = f"""
        **Recommended Model:** {recommended_model}
        
        **Model Metrics:**
        • ROC AUC: {model_metrics.get('roc_auc', 0):.3f}
        • Precision: {model_metrics.get('precision', 0):.3f}
        • Recall: {model_metrics.get('recall', 0):.3f}
        • F1 Score: {model_metrics.get('f1', 0):.3f}
        
        **Explanation:**
        {explanation}
        """
        self.view.show_message(recommendation_msg, "info")

        # Show selection options
        self.view.display_subheader("Make Your Selection")
        available_models = list(step7_output['model_results'].keys())
        
        selection_mode = self.view.display_radio(
            "How would you like to select the model?",
            options=["Use AI Recommended Model", "Select Model Manually"],
            key="model_selection_mode"
        )
        
        if selection_mode == "Use AI Recommended Model":
            if self.view.display_button("✅ Confirm AI Recommendation"):
                model_id = step7_output['model_results'][recommended_model].get('model_id')
                self.session.set('selected_model', recommended_model)
                self.session.set('selected_model_id', model_id)
                self.session.set('model_selection_complete', True)
                self.view.show_message(f"✅ Selected model: {recommended_model}", "success")
        else:
            selected_model = self.view.display_radio(
                "Choose a model:",
                options=available_models,
                key="manual_model_selection"
            )
            
            if self.view.display_button("✅ Confirm Selection"):
                if selected_model:
                    model_id = step7_output['model_results'][selected_model].get('model_id')
                    self.session.set('selected_model', selected_model)
                    self.session.set('selected_model_id', model_id)
                    self.session.set('model_selection_complete', True)
                    self.view.show_message(f"✅ Selected model: {selected_model}", "success")
                else:
                    self.view.show_message("⚠️ Please select a model first", "warning")

        # Show proceed button if model is selected
        if self.session.get('model_selection_complete'):
            if self.view.display_button("Proceed to Inference", key="proceed_to_inference"):
                self.session.set('current_step', 9)
                self.view.rerun_script()

    def get_summary(self):
        if not self.session.get('model_selection_complete'):
            return "Model selection not yet completed"
        
        selected_model = self.session.get('selected_model')
        was_ai_recommended = selected_model == self.session.get('model_recommendation', {}).get('recommended_model')
        
        return f"Selected model: {selected_model} ({'AI recommended' if was_ai_recommended else 'Manually selected'})" 