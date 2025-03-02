import pandas as pd
import numpy as np
from orchestrator.agents.clustering_agent import SFNClusteringAgent
from sfn_blueprint import Task
import traceback
from typing import Dict, List, Tuple

class ClusteringSetup:
    def __init__(self, session_manager, view):
        self.session = session_manager
        self.view = view
        self.clustering_agent = SFNClusteringAgent()
        self.algorithms = {
            'kmeans': 'K-Means: Best for spherical clusters and when number of clusters is known',
            'dbscan': 'DBSCAN: Good for density-based clusters and detecting outliers'
        }
        self.max_retries = 3
        
    def execute(self):
        """Execute clustering step"""
        # Check if already completed
        if self.session.get('step_3_complete'):
            return True
            
        # If not started, show initial setup interface
        if not self.session.get('clustering_started'):
            if not self._show_setup_interface():
                return False
                
        # Run algorithms sequentially
        for algorithm in self.algorithms.keys():
            # Skip if this algorithm is already complete
            if self.session.get(f'{algorithm}_complete'):
                continue
            
            # Set current algorithm
            self.session.set('current_algorithm', algorithm)
            
            # Run clustering for current algorithm
            clustering_result = self._handle_clustering(algorithm)
            
            # If clustering failed, stop
            if not clustering_result and not self.session.get(f'{algorithm}_complete'):
                return False
            
        # Check if all algorithms are complete
        all_complete = all(
            self.session.get(f'{alg}_complete', False) 
            for alg in self.algorithms
        )
        
        if all_complete:
            return self._handle_completion()
        
        return False
        
    def _show_setup_interface(self):
        """Display initial clustering setup interface"""
        self.view.display_subheader("Available Clustering Algorithms")
        
        # Show algorithm descriptions
        algo_info = "Supported Algorithms:\n"
        for algo, description in self.algorithms.items():
            algo_info += f"- **{algo.upper()}**: {description}\n"
        self.view.show_message(algo_info, "info")
        
        # Custom instructions input
        self.view.display_subheader("Custom Instructions (Optional)")
        custom_instructions = self.view.text_area(
            "Add any specific instructions for clustering:",
            help="Examples:\n- 'Focus on usage patterns'\n- 'Prioritize revenue patterns'\n- 'Consider API usage as primary factor'",
            key="clustering_instructions"
        )
        
        # Begin clustering button
        if self.view.display_button("🚀 Begin Clustering Analysis"):
            self.session.set('custom_instructions', custom_instructions)
            self.session.set('clustering_started', True)
            return True
            
        return False
        
    def _handle_clustering(self, algorithm: str):
        """Handle clustering for current algorithm"""
        try:
            # Get data and feature info
            df = self.session.get('df')
            mappings = self.session.get('field_mappings', {})  # Get the mappings
            id_field = mappings.get('id')  # Get ID field from mappings
            
            # Prepare clustering task
            df_info = {
                'shape': df.shape,
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'numeric_features': self.session.get('feature_info', {}).get('numeric_features', []),
                'id_field': id_field  # Add ID field to df_info
            }
            
            custom_instructions = (
                f"IMPORTANT: Use train_df['{id_field}'] for the ID values in cluster_mapping. " +
                "Example cluster_mapping creation:\n" +
                "cluster_mapping = pd.DataFrame({{\n" +
                "    'id': train_df[id_field],\n" +
                "    'cluster': clusters\n" +
                "}})"
            )
            
            clustering_task = Task(
                "Perform clustering",
                data={
                    'df': df,
                    'df_info': df_info,
                    'algorithm': algorithm,
                    'custom_instructions': custom_instructions + "\n" + self.session.get('custom_instructions', '')
                }
            )
            
            # Execute clustering
            with self.view.display_spinner(f'🤖 Training {algorithm.upper()} model...'):
                # Get code from agent
                agent_response = self.clustering_agent.execute_task(clustering_task)
                
                if not agent_response or 'code' not in agent_response:
                    self.view.show_message("Failed to generate clustering code", "error")
                    return False
                    
                # Create execution environment
                from sklearn.cluster import KMeans, DBSCAN
                from sklearn.preprocessing import StandardScaler
                from sklearn.metrics import silhouette_score
                
                local_vars = {
                    'train_df': df,
                    'pd': pd,
                    'np': np,
                    'KMeans': KMeans,
                    'DBSCAN': DBSCAN,
                    'StandardScaler': StandardScaler,
                    'silhouette_score': silhouette_score
                }
                
                # Execute the code
                exec(agent_response['code'], globals(), local_vars)
                
                # Extract results
                result = {
                    'clusters': local_vars.get('clusters'),
                    'metrics': local_vars.get('metrics'),
                    'cluster_mapping': local_vars.get('cluster_mapping')
                }
                
                if result['clusters'] is None or result['metrics'] is None:
                    self.view.show_message("Clustering code execution failed to produce results", "error")
                    return False
                
                # Save results
                self.session.set(f'{algorithm}_clusters', result['clusters'])
                self.session.set(f'{algorithm}_metrics', result['metrics'])
                self.session.set(f'{algorithm}_cluster_mapping', result['cluster_mapping'])
                self.session.set(f'{algorithm}_complete', True)
                
                # Show results
                self._display_algorithm_results(algorithm, result)
                
                self.view.show_message(f"✅ {algorithm.upper()} clustering complete", "success")
                return True
            
        except Exception as e:
            self.view.show_message(f"Error in clustering: {str(e)}", "error")
            traceback.print_exc()
            return False
        
    def _display_algorithm_results(self, algorithm: str, result: Dict):
        """Display results for current algorithm"""
        self.view.display_subheader(f"{algorithm.upper()} Results")
        
        # Show metrics
        metrics_msg = f"📊 {algorithm.upper()} Metrics:\n"
        for metric, value in result['metrics'].items():
            metrics_msg += f"- {metric}: **{value}**\n"
        self.view.show_message(metrics_msg, "info")
        
        # Show cluster distribution
        cluster_counts = pd.Series(result['clusters']).value_counts().sort_index()
        dist_msg = "🔍 Cluster Distribution:\n"
        
        if algorithm == 'dbscan':
            # Special handling for DBSCAN
            n_noise = cluster_counts.get(-1, 0)
            if -1 in cluster_counts:
                cluster_counts = cluster_counts[cluster_counts.index != -1]
                dist_msg += f"- Core Clusters:\n"
                for cluster, count in cluster_counts.items():
                    dist_msg += f"  • Cluster {cluster}: {count} records\n"
                dist_msg += f"- Noise Points: {n_noise} records\n"
        else:
            # Regular handling for other algorithms
            for cluster, count in cluster_counts.items():
                dist_msg += f"- Cluster {cluster}: {count} records\n"
            
        self.view.show_message(dist_msg, "info")
        
    def _handle_algorithm_completion(self, current_algorithm: str) -> bool:
        """Handle completion of current algorithm"""
        # Get next algorithm
        algorithms = list(self.algorithms.keys())
        current_idx = algorithms.index(current_algorithm)
        
        if current_idx < len(algorithms) - 1:
            # Move to next algorithm
            next_algorithm = algorithms[current_idx + 1]
            self.session.set('current_algorithm', next_algorithm)
            # Return False to continue processing
            return False
        else:
            # All algorithms complete
            return self._handle_completion()
            
    def _handle_completion(self) -> bool:
        """Handle completion of all clustering algorithms"""
        self.view.display_subheader("Clustering Analysis Complete")
        self.view.show_message("✅ All clustering models trained successfully!", "success")
        
        # Save step summary first
        self._save_step_summary()
        
        # Show proceed button
        if self.view.display_button("▶️ Proceed to Model Selection"):
            self.session.set('step_3_complete', True)
            return True
            
        return False
        
    def _save_step_summary(self):
        """Save step summary for completed steps display"""
        summary = "✅ Clustering Analysis Complete\n\n"
        
        for algorithm in self.algorithms:
            metrics = self.session.get(f'{algorithm}_metrics', {})
            clusters = self.session.get(f'{algorithm}_clusters', [])
            
            if metrics and clusters is not None:
                summary += f"**{algorithm.upper()}**:\n"
                # Add metrics
                for metric, value in metrics.items():
                    if metric == 'cluster_sizes':
                        if algorithm == 'dbscan':
                            n_clusters = len([c for c in clusters if c != -1])
                            n_noise = len([c for c in clusters if c == -1])
                            summary += f"- Core Clusters: **{n_clusters}**\n"
                            summary += f"- Noise Points: **{n_noise}**\n"
                        else:
                            summary += f"- Number of Clusters: **{len(set(clusters))}**\n"
                    else:
                        summary += f"- {metric}: **{value}**\n"
                summary += "\n"
                
        self.session.set('step_3_summary', summary) 