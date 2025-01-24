from classification_agent.agents.model_training_agent import SFNModelTrainingAgent
from sfn_blueprint import Task
import pandas as pd
import time
from datetime import datetime
from utils.model_manager import ModelManager

class Step7ModelTrainingOrchestrator:
    def __init__(self, session_manager, view):
        self.model_training_agent = SFNModelTrainingAgent()
        self.models = ['XGBoost', 'Random Forest', 'LightGBM', 'CatBoost']
        self.session = session_manager
        self.view = view
        self.model_manager = ModelManager()

    def orchestrate(self):
        
        if not self.session.get('target_column'):
            self._show_target_mapping_ui()
            return
            
        if not self.session.get('model_training_started'):
            self.session.set('model_training_started', False)
            self.session.set('current_model_index', 0)
            self.session.set('model_results', {})
            
        if not self.session.get('model_training_started'):
            if self.view.display_button("Start Model Training"):
                self.session.set('model_training_started', True)
                self.view.rerun_script()
                
        if self.session.get('model_training_started'):
            self._run_model_training()
            
    def _show_target_mapping_ui(self):
        self.view.display_markdown("Before proceeding with model training, please select your target column.")
        self.view.show_message("⚠️ Target column selection is mandatory for model training.", "warning")
        
        # Get columns from training dataframe
        split_results = self.session.get('splitting_results', {})
        train_df = split_results.get('train_df')
        
        if train_df is not None:
            columns = train_df.columns.tolist()
            
            # Create target column selector
            target_col = self.view.select_box(
                "Select Target Column",
                options=columns,
                key="target_column_selector"
            )
            
            if self.view.display_button("Confirm Target Column"):
                self.session.set('target_column', target_col)
                self.view.show_message(f"Target column set to: {target_col}", "success")
                self.view.rerun_script()
        else:
            self.view.show_message("No training data available. Please complete previous steps first.", "error")
            
    def _run_model_training(self):
        current_model_index = self.session.get('current_model_index', 0)
        if current_model_index >= len(self.models):
            self._show_final_results()
            return
            
        current_model = self.models[current_model_index]
        
        # Show progress for current model
        self.view.display_subheader(f"Training {current_model}")
        
        # Show previous results if any
        model_results = self.session.get('model_results', {})
        if model_results:
            self.view.display_subheader("Results So Far:")
            for model, results in model_results.items():
                self._display_model_results(model, results)
        
        try:
            task_data = {
                'df_train': self.session.get('train_df'),
                'df_valid': self.session.get('valid_df'),
                'target_column': self.session.get('target_column'),
                'model_name': current_model.lower()
            }
            
            task = Task("model_training", data=task_data)
            
            # Use spinner instead of progress bar
            with self.view.display_spinner(f"🤖 Training {current_model} model..."):
                results = self.model_training_agent.execute_task(task)
                
                # Save model object and metadata
                model_metadata = {
                    'metrics': results.get('metrics', {}),
                    'training_date': str(datetime.now()),
                    'target_column': self.session.get('target_column'),
                    'training_features': results.get('training_features', [])
                }
                model_id = self.model_manager.save_model(
                    model=results.get('model'),
                    model_name=current_model,
                    metadata=model_metadata
                )
                
                # Store model_id in results
                results['model_id'] = model_id
                
                model_results[current_model] = results
                self.session.set('model_results', model_results)
                
            self.view.show_message(f"✅ Completed training {current_model}!", "success")
            
            # Move to next model
            self.session.set('current_model_index', current_model_index + 1)
            time.sleep(1)
            self.view.rerun_script()
            
        except Exception as e:
            self.view.show_message(f"Error training {current_model}: {str(e)}", "error")
            
    def _show_final_results(self):
        self.view.show_message("🎉 Model Training Complete!", "success")
        
        self.view.display_subheader("Final Results")
        model_results = self.session.get('model_results', {})
        for model, results in model_results.items():
            self._display_model_results(model, results)
            
        # Store all model results in step7_output with simplified structure
        step7_output = {
            'model_results': {
                model_name: {
                    'name': model_name,
                    'metrics': results.get('metrics', {}),
                    'model_id': results.get('model_id')
                }
                for model_name, results in model_results.items()
            },
            'step7_validation': True
        }
        self.session.set('step7_output', step7_output)
        
        if self.view.display_button("Proceed to model selection"):
            self.session.set('current_step', 8)
            self.view.rerun_script()
            
    def _display_model_results(self, model_name, results):
        metrics = results.get('metrics', {})
        records_info = results.get('records_info', {})
        
        self.view.display_markdown(f"### {model_name}")
        
        metrics_message = f"""
        **Model Metrics:** \n
        • ROC AUC: {metrics.get('roc_auc', 0):.3f} \n
        • Precision: {metrics.get('precision', 0):.3f} \n
        • Recall: {metrics.get('recall', 0):.3f} \n
        • F1 Score: {metrics.get('f1', 0):.3f} \n
        """
        
        self.view.show_message(metrics_message, "info")
            
    def get_summary(self):
        if not self.session.get('model_results'):
            return "Model training not yet completed"
            
        best_model = None
        best_score = -1
        
        for model, results in self.session.get('model_results', {}).items():
            score = results.get('metrics', {}).get('roc_auc', 0)
            if score > best_score:
                best_score = score
                best_model = model
                
        return f"Model training completed. Best model: {best_model} (ROC AUC: {best_score:.3f})" 