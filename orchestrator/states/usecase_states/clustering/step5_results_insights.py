from sfn_blueprint import Task
import pandas as pd
from typing import Dict
import numpy as np

class ResultsAndInsights:
    def __init__(self, session_manager, view):
        self.session = session_manager
        self.view = view
        
    def execute(self):
        """Execute results and insights step"""
        # Get selected model and data
        selected_model = self.session.get('selected_model')
        model_clusters = self.session.get('selected_model_clusters')
        model_metrics = self.session.get('selected_model_metrics')
        df = self.session.get('df')
        
        if not all([selected_model, model_clusters is not None, model_metrics, df is not None]):
            self.view.show_message(
                "❌ Missing required data. Please complete model selection first.",
                "error"
            )
            return False
            
        # Check if results have been generated using a flag instead of the DataFrame
        if not self.session.get('step_5_started', False):
            # Add cluster assignments to original data
            results_df = df.copy()
            results_df['Cluster'] = model_clusters
            self.session.set('clustering_results_df', results_df)
            
            # Display results
            self._display_clustering_results(results_df, model_metrics)
            
            # Save summary
            self._save_step_summary(results_df, model_metrics)
            self.session.set('step_5_complete', True)
            self.session.set('step_5_started', True)
        
        # Always show results if available
        results_df = self.session.get('clustering_results_df')
        if isinstance(results_df, pd.DataFrame) and not results_df.empty:
            # Show summary
            summary = self.session.get('step_5_summary')
            if summary:
                self.view.show_message(summary, "success")
                
            # Display cluster statistics table
            self._display_cluster_statistics(selected_model)
            
            # Add download button
            try:
                result_df = self._prepare_download()
                if result_df is not None:
                    self.view.create_download_button(
                        label="📥 Download Results as CSV",
                        data=result_df.to_csv(index=False),
                        file_name=f"clustering_results_{selected_model}.csv",
                        mime_type='text/csv'
                    )
            except Exception as e:
                self.view.show_message(f"❌ Error preparing download: {str(e)}", "error")
            
            # Add reset button
            if self.view.display_button("🔄 Start New Analysis"):
                self.session.clear()
                self.view.rerun_script()
                return True
                
        return self.session.get('step_5_complete', False)
        
    def _display_clustering_results(self, results_df, metrics):
        """Display clustering results and insights"""
        # Get selected model from session
        selected_model = self.session.get('selected_model')
        if not selected_model:
            self.view.show_message("❌ Selected model not found", "error")
            return
        
        self.view.display_subheader("🎯 Clustering Results")
        
        # Display overall metrics
        metrics_msg = "**Clustering Performance Metrics:**\n"
        metrics_msg += f"- Silhouette Score: **{metrics['silhouette_score']:.3f}**\n"
        metrics_msg += f"- Within-cluster Sum of Squares: **{metrics['within_cluster_sum_squares']:.3f}**\n"
        
        # Display cluster size distribution
        metrics_msg += "\n**Cluster Size Distribution:**\n"
        for cluster_id, size in enumerate(metrics['cluster_sizes']):
            metrics_msg += f"- Cluster {cluster_id}: **{size}** samples\n"
        
        # Add DBSCAN-specific metrics if available
        if 'n_clusters' in metrics:
            metrics_msg += f"\nNumber of Clusters Found: **{metrics['n_clusters']}**\n"
        
        self.view.show_message(metrics_msg, "info")
        
        # Display cluster insights
        self._generate_cluster_insights(results_df, metrics)

    def _display_cluster_statistics(self, selected_model):
        """Display statistics for each cluster"""
        # Get required data
        cluster_mapping = self.session.get(f'{selected_model}_cluster_mapping')
        aggregated_df = self.session.get('df_aggregated')
        
        if cluster_mapping is None or aggregated_df is None:
            self.view.show_message("❌ Required data not found", "error")
            return
        
        # Get field mappings
        field_mappings = self.session.get('field_mappings', {})
        id_field = field_mappings.get('id')
        
        if not id_field:
            self.view.show_message("❌ ID field not found in mappings", "error")
            return
        
        # Check if 'id' column exists in cluster_mapping
        if 'id' not in cluster_mapping.columns:
            # If not, check if the id_field exists
            if id_field in cluster_mapping.columns:
                # Use the id_field column as 'id'
                cluster_mapping = cluster_mapping.rename(columns={id_field: 'id'})
            else:
                self.view.show_message(f"❌ Neither 'id' nor '{id_field}' column found in cluster mapping", "error")
                return
        
        # Merge cluster assignments with aggregated data
        results_df = aggregated_df.merge(
            cluster_mapping[['id', 'cluster']],
            left_on=id_field,
            right_on='id',
            how='left'
        )
        
        # Display cluster statistics
        self.view.display_subheader("📊 Cluster Statistics")
        
        # Get unique clusters
        clusters = results_df['cluster'].unique()
        clusters.sort()
        
        # Create statistics table
        stats_data = []
        
        # Get numeric columns for statistics
        numeric_cols = results_df.select_dtypes(include=['number']).columns.tolist()
        # Exclude id and cluster columns
        numeric_cols = [col for col in numeric_cols if col not in ['id', 'cluster']]
        
        # Calculate statistics for each cluster
        for cluster in clusters:
            cluster_data = results_df[results_df['cluster'] == cluster]
            cluster_size = len(cluster_data)
            cluster_pct = (cluster_size / len(results_df)) * 100
            
            row = {
                "Cluster": f"Cluster {cluster}",
                "Size": f"{cluster_size} samples ({cluster_pct:.1f}%)"
            }
            
            # Add mean and std for each numeric column
            for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
                if col in cluster_data.columns:
                    mean = cluster_data[col].mean()
                    std = cluster_data[col].std()
                    row[col] = f"{mean:.2f} (±{std:.2f})"
            
            stats_data.append(row)
        
        # Display statistics table
        if stats_data:
            self.view.display_table(stats_data)
            self.view.show_message("Note: Values shown as Mean (±Standard Deviation) for each metric", "info")
        else:
            self.view.show_message("No cluster statistics available", "warning")

    def _generate_cluster_insights(self, results_df, metrics):
        """Generate insights about the clusters"""
        self.view.display_subheader("💡 Key Insights")
        
        # Calculate basic insights
        n_clusters = len(metrics['cluster_sizes'])
        largest_cluster = max(metrics['cluster_sizes'])
        smallest_cluster = min(metrics['cluster_sizes'])
        
        insights_msg = "**Cluster Analysis:**\n\n"
        insights_msg += f"- Total number of clusters: **{n_clusters}**\n"
        insights_msg += f"- Largest cluster size: **{largest_cluster}** samples\n"
        insights_msg += f"- Smallest cluster size: **{smallest_cluster}** samples\n"
        insights_msg += f"- Overall clustering quality (Silhouette Score): **{metrics['silhouette_score']:.3f}**\n"
        
        self.view.show_message(insights_msg, "success")

    def _save_step_summary(self, results_df: pd.DataFrame, metrics: Dict):
        """Save step summary for display in completed steps"""
        # Calculate cluster percentages
        total_samples = len(results_df)
        cluster_percentages = {
            i: (size / total_samples) * 100 
            for i, size in enumerate(metrics['cluster_sizes'])
        }
        
        # Format summary
        summary = "✅ Step 5 Complete\n\n"
        summary += "**Clustering Results Summary:**\n"
        summary += f"- Model: {self.session.get('selected_model', 'Unknown').upper()}\n"
        summary += f"- Silhouette Score: {metrics['silhouette_score']:.3f}\n"
        summary += f"- Within-cluster Sum of Squares: {metrics['within_cluster_sum_squares']:.3f}\n\n"
        
        # Add cluster distribution
        summary += "**Cluster Distribution:**\n"
        for cluster_id, size in enumerate(metrics['cluster_sizes']):
            percentage = cluster_percentages[cluster_id]
            summary += f"- Cluster {cluster_id}: {size} samples ({percentage:.1f}%)\n"
        
        # Add DBSCAN-specific info if available
        if 'n_clusters' in metrics:
            summary += f"\nNumber of Clusters Found: {metrics['n_clusters']}\n"
        
        # Save summary
        self.session.set('step_5_summary', summary)

    def _prepare_download(self):
        """Prepare data for download"""
        try:
            # Check if download data already exists
            download_data = self.session.get('download_data')
            if download_data is not None:
                return download_data
            
            # Get the selected model
            selected_model = self.session.get('selected_model', 'kmeans')
            
            # Get required data
            cluster_mapping = self.session.get(f'{selected_model}_cluster_mapping')
            aggregated_df = self.session.get('df_aggregated')
            
            if cluster_mapping is None or aggregated_df is None:
                self.view.show_message("❌ Required data not found", "error")
                return None
            
            # Get ID field from mappings
            field_mappings = self.session.get('field_mappings', {})
            id_field = field_mappings.get('id')
            
            if not id_field:
                self.view.show_message("❌ ID field not found in mappings", "error")
                return None
            
            # Merge cluster assignments with original data
            result = pd.merge(
                aggregated_df,
                cluster_mapping,
                left_on=id_field,
                right_on='id',
                how='left'
            )
            
            # Set download data in session
            self.session.set('download_data', result)
            
            return result
        
        except Exception as e:
            self.view.show_message(f"❌ Error preparing download: {str(e)}", "error")
            return None 