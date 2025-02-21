from sfn_blueprint import Task, SFNValidateAndRetryAgent
from clustering_agent.agents.clustering_strategy_selector import SFNClusterSelectionAgent
from clustering_agent.config.model_config import DEFAULT_LLM_PROVIDER
from typing import Dict
import pandas as pd
import numpy as np

class ClusteringModelSelection:
    def __init__(self, session_manager, view):
        self.session = session_manager
        self.view = view
        self.selection_agent = SFNClusterSelectionAgent()
        
    def execute(self):
        """Execute clustering model selection step"""
        self.view.display_header("Step 4: Model Selection")
        
        # Get cached response or run analysis
        response = self.session.get('model_selection_response')
        if response is None:
            # Get clustering results
            clustering_results = self._get_clustering_results()
            if not clustering_results:
                self.view.show_message("❌ No clustering results found", "error")
                return False
            
            self.view.display_subheader("Model Selection Results:")
            
            # Get model selection recommendation
            task = Task(
                "Select best clustering model",
                data={
                    'clustering_results': clustering_results,
                    'custom_instructions': self.session.get('custom_instructions', '')
                }
            )
            
            try:
                with self.view.display_spinner('🤖 Analyzing clustering results...'):
                    response = self.selection_agent.execute_task(task)
                
                if not response or not isinstance(response, dict):
                    self.view.show_message("❌ Invalid response from selection agent", "error")
                    return False
                
                # Cache the response
                self.session.set('model_selection_response', response)
                
            except Exception as e:
                self.view.show_message(f"Error in model selection: {str(e)}", "error")
                return False
            
        # Display results (always show this)
        self._display_recommendation(response)
        
        # Get user selection
        selected_model = self._get_user_selection(response)
        
        if selected_model:
            # Save selection
            self.session.set('selected_model', selected_model)
            self.session.set('selected_model_clusters', 
                           self.session.get(f'{selected_model}_clusters'))
            self.session.set('selected_model_metrics', 
                           self.session.get(f'{selected_model}_metrics'))
            self.session.set('selection_mode', 
                           'ai' if selected_model == response.get('selected_model') else 'manual')
            
            # Save summary and complete step
            self._save_step_summary(response)
            self.session.set('step_4_complete', True)
            
            # Show success message
            self.view.show_message(f"✅ {selected_model.upper()} selected as the final clustering model!", "success")
            return True
        
        return False
        
    def _get_clustering_results(self) -> Dict:
        """Get formatted clustering results for agent"""
        results = {}
        
        def convert_to_native(obj):
            """Convert numpy types to native Python types"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {int(k) if isinstance(k, np.integer) else k: convert_to_native(v) 
                       for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj
        
        for algorithm in ['kmeans', 'dbscan']:
            metrics = self.session.get(f'{algorithm}_metrics')
            clusters = self.session.get(f'{algorithm}_clusters')
            if metrics and clusters is not None:
                # Convert metrics and cluster distribution to native Python types
                converted_metrics = convert_to_native(metrics)
                
                # Convert cluster distribution keys to native integers
                cluster_dist = pd.Series(clusters).value_counts().to_dict()
                converted_dist = {
                    int(k) if isinstance(k, np.integer) else k: int(v) 
                    for k, v in cluster_dist.items()
                }
                
                results[algorithm] = {
                    'metrics': converted_metrics,
                    'cluster_distribution': converted_dist
                }
        return results
        
    def _display_recommendation(self, response: Dict):
        """Display model selection recommendation"""
        self.view.display_subheader("🤖 Model Recommendation")
        
        # Show recommended model
        recommended_model = response.get('selected_model', '').upper()
        self.view.show_message(f"Recommended Model: **{recommended_model}**", "success")
        
        # Show explanation
        explanation = response.get('explanation', '')
        if explanation:
            self.view.display_subheader("Explanation")
            self.view.show_message(explanation, "info")
            
        # Show comparison
        comparison = response.get('comparison_summary', '')
        if comparison:
            self.view.display_subheader("Model Comparison")
            self.view.show_message(comparison, "info")
            
        # Show rankings if available
        rankings = response.get('model_rankings', [])
        if rankings:
            self.view.display_subheader("Model Rankings")
            for rank in rankings:
                rank_msg = f"**{rank['model'].upper()}** (Rank {rank['rank']})\n"
                rank_msg += "Key Strengths:\n"
                for strength in rank['key_strengths']:
                    rank_msg += f"- {strength}\n"
                self.view.show_message(rank_msg, "info")
                
    def _get_user_selection(self, response: Dict) -> str:
        """Get user's model selection"""
        self.view.display_subheader("Model Selection")
        
        selection_mode = self.view.radio_select(
            "How would you like to select the clustering model?",
            options=[
                "Use AI Recommended Model",
                "Select Model Manually"
            ],
            key="model_selection_mode"
        )
        
        if selection_mode == "Use AI Recommended Model":
            if self.view.display_button("✅ Confirm AI Recommendation"):
                return response.get('selected_model')
        else:
            model = self.view.radio_select(
                "Select clustering model:",
                options=["kmeans", "dbscan"],
                key="manual_model_selection"
            )
            if self.view.display_button("✅ Confirm Selection"):
                return model
                
        return None
        
    def _save_step_summary(self, response: Dict):
        """Save step summary"""
        selected_model = self.session.get('selected_model', '').upper()
        selection_mode = self.session.get('selection_mode', '')
        
        summary = f"✅ Selected Model: **{selected_model}**\n\n"
        summary += f"Selection Mode: **{selection_mode}**\n\n"
        
        if response.get('explanation'):
            summary += f"Explanation:\n{response['explanation']}\n\n"
            
        if response.get('comparison_summary'):
            summary += f"Comparison Summary:\n{response['comparison_summary']}"
            
        self.session.set('step_4_summary', summary) 