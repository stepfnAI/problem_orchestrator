from sfn_blueprint import Task, SFNValidateAndRetryAgent
from orchestrator.agents.reg_model_training_agent import SFNModelTrainingAgent
from orchestrator.config.model_config import DEFAULT_LLM_PROVIDER
from orchestrator.utils.reg_model_manager import ModelManager
import pandas as pd
from typing import Dict

class ModelTraining:
    def __init__(self, session_manager, view):
        self.session = session_manager
        self.view = view
        self.training_agent = SFNModelTrainingAgent()
        self.model_pool = ['xgboost', 'lightgbm', 'catboost']
        self.model_manager = ModelManager()
        
    def execute(self):
        """Execute model training step"""
        try:
            # Get analysis type and data
            is_forecasting = self.session.get('is_forecasting', False)
            df = self.session.get('df')
            mappings = self.session.get('field_mappings')
            
            # Get appropriate target column based on analysis type
            target_column = mappings.get('forecasting_field' if is_forecasting else 'target')
            if not target_column:
                self.view.show_message(
                    "❌ Target column not found in mappings.",
                    "error"
                )
                return False
            # Get split info and data
            split_info = self.session.get('split_info')
            print("<><><><split_info",split_info)
            print("<><><><split_info['train_df']",split_info['train_df'].head(3))
            # # Get training sets
            # train_df = self.session.get('train_df')
            # valid_df = self.session.get('valid_df')
            # infer_df = self.session.get('infer_df')
            
            if not split_info:
                self.view.show_message("❌ Data split information not found.", "error")
                return False
            
            # Initialize appropriate training agent
            training_agent = SFNModelTrainingAgent(
                analysis_type="forecasting" if is_forecasting else "regression"
            )
            
            # Get appropriate models list
            if is_forecasting:
                models = ["Prophet","SARIMAX"] # ["Prophet", "SARIMAX", "XGBoost", "LightGBM"]
            else:
                models = ["XGBoost", "LightGBM", "CatBoost"]
            
            # Train each model
            results = {}
            for model_name in models:
                with self.view.display_spinner(f'Training {model_name} model...'):
                    task_data = {
                        'df_train': split_info['train_df'],
                        'df_valid': split_info['valid_df'],
                        'target_column': target_column,
                        'date_column': mappings.get('date'),
                        'model_name': model_name
                    }
                    
                    result = training_agent.execute_task(
                        Task(f"Train {model_name}", data=task_data)
                    )
                    
                    if result.get('model') is not None:
                        results[model_name] = result
                        indi_results = {model_name: result}
                        self._display_training_results(indi_results, is_forecasting)
                    else:
                        self.view.show_message(
                            f"⚠️ {model_name} training failed: {result.get('metrics', {}).get('error')}",
                            "warning"
                        )
            
            if not results:
                self.view.show_message("❌ All model training attempts failed.", "error")
                return False
            
            # Save results and display metrics
            self.session.set('model_results', results)
            self._display_training_results(results, is_forecasting)
            
            # Save step summary
            self._save_step_summary(results, is_forecasting)
            self.session.set('step_5_complete', True)
            return True
            
        except Exception as e:
            self.view.show_message(f"Error in model training: {str(e)}", "error")
            return False
            
    def _display_training_results(self, results: Dict, is_forecasting: bool):
        """Display training results with appropriate metrics"""
        self.view.display_subheader("Model Training Results")
        
        for model_name, result in results.items():
            metrics = result.get('metrics', {})
            
            # Display appropriate metrics based on analysis type
            if is_forecasting:
                metric_msg = (
                    f"**{model_name}**:\n"
                    f"- MAPE: {metrics.get('mape', 'N/A'):.4f}\n"
                    f"- RMSE: {metrics.get('rmse', 'N/A'):.4f}\n"
                    f"- MAE: {metrics.get('mae', 'N/A'):.4f}\n"
                )
            else:
                metric_msg = (
                    f"**{model_name}**:\n"
                    f"- R² Score: {metrics.get('r2', 'N/A'):.4f}\n"
                    f"- MSE: {metrics.get('mse', 'N/A'):.4f}\n"
                    f"- MAE: {metrics.get('mae', 'N/A'):.4f}\n"
                    f"- RMSLE: {metrics.get('rmsle', 'N/A'):.4f}\n"
                )
            
            self.view.show_message(metric_msg, "info")
            
    def _save_step_summary(self, results: Dict, is_forecasting: bool):
        """Save step summary with appropriate metrics"""
        summary = "✅ Model Training Complete\n\n"
        
        # Add analysis type
        summary += f"**Analysis Type:** {'Forecasting' if is_forecasting else 'Regression'}\n\n"
        
        # Add model results
        summary += "**Model Performance:**\n"
        for model_name, result in results.items():
            metrics = result.get('metrics', {})
            records = result.get('records_info', {})
            
            summary += f"\n{model_name}:\n"
            if is_forecasting:
                summary += f"- MAPE: {metrics.get('mape', 'N/A'):.4f}\n"
                summary += f"- RMSE: {metrics.get('rmse', 'N/A'):.4f}\n"
            else:
                summary += f"- R² Score: {metrics.get('r2', 'N/A'):.4f}\n"
                summary += f"- MSE: {metrics.get('mse', 'N/A'):.4f}\n"
                summary += f"- RMSLE: {metrics.get('rmsle', 'N/A'):.4f}\n"
            summary += f"- MAE: {metrics.get('mae', 'N/A'):.4f}\n"
            
            # Add records info
            summary += f"- Training Records: {records.get('used_train', 0)}/{records.get('total_train', 0)}\n"
            summary += f"- Validation Records: {records.get('used_valid', 0)}/{records.get('total_valid', 0)}\n"
        
        self.session.set('step_5_summary', summary)
        
    def _display_model_metrics(self, model_name: str, metrics: dict):
        """Display metrics for a trained model"""
        self.view.display_subheader(f"{model_name} Results")
        
        def format_metric(value):
            if value is None or value == 'N/A':
                return 'N/A'
            try:
                float_val = float(value)
                return f"{float_val:.3f}" if float_val != float('inf') else 'N/A'
            except (ValueError, TypeError):
                return str(value)
        
        metrics_text = "**Metrics:**\n"
        metrics_to_display = {
            'R² Score': metrics.get('r2'),
            'Mean Squared Error': metrics.get('mse'),
            'Mean Absolute Error': metrics.get('mae'),
            'Root Mean Squared Log Error': metrics.get('rmsle')
        }
        
        for metric_name, value in metrics_to_display.items():
            formatted_value = format_metric(value)
            metrics_text += f"- {metric_name}: {formatted_value}\n"
        
        self.view.show_message(metrics_text, "info") 