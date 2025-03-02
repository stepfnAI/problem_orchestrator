import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Fix the import paths
from orchestrator.states.usecase_states.clustering.step3_clustering_setup import ClusteringSetup
from orchestrator.states.usecase_states.clustering.step4_model_selection import ClusteringModelSelection
from orchestrator.states.usecase_states.clustering.step5_results_insights import ResultsAndInsights
from orchestrator.utils.streamlit_views import StreamlitView
from sfn_blueprint import SFNSessionManager
import logging
import numpy as np

logger = logging.getLogger(__name__)

class ClusteringOrchestrator:
    def __init__(self, session_manager, view):
        self.view = view
        self.session = session_manager
        self.orchestrators = {
            3: ClusteringSetup(self.session, self.view),
            4: ClusteringModelSelection(self.session, self.view),
            5: ResultsAndInsights(self.session, self.view)
        }
        self.step_titles = {
            3: "Clustering Setup",
            4: "Clustering Model Selection",
            5: "Results & Insights"
        }
        
    def run(self):
        """Main clustering flow"""
        logger.info("Starting Clustering Orchestrator")
        
        # Load necessary data from database if not already in session
        if self.session.get('df') is None or self.session.get('field_mappings') is None:
            if not self._load_data_from_db():
                self.view.show_message("❌ Failed to load required data from database", "error")
                return False
        
        # Get current step (starting from step 3 since steps 1-2 are already done)
        current_step = self.session.get('clustering_current_step', 3)
        
        # Display completed steps
        self._display_completed_steps(current_step)
        
        # Display current step header
        if current_step in self.step_titles:
            self.view.display_header(f"Step {current_step}: {self.step_titles[current_step]}")
            self.view.display_markdown("---")
            
            # Execute current step
            if current_step in self.orchestrators:
                logger.info(f"Executing clustering step {current_step}: {self.step_titles[current_step]}")
                result = self.orchestrators[current_step].execute()
                
                # Only advance if not the final step and step is complete
                if result and current_step < max(self.orchestrators.keys()):
                    self._advance_step()
                # For final step, save state summary and results to database
                elif result and current_step == max(self.orchestrators.keys()):
                    self._save_clustering_results()
                    return True
        else:
            self.view.show_message("❌ Invalid step. Resetting application.", "error")
            self.session.set('clustering_current_step', 3)
            self.view.rerun_script()
    
    def _display_completed_steps(self, current_step):
        """Display summary of completed steps"""
        # First display the summaries from the main workflow (steps 1-2)
        self.view.display_header("Previous Steps")
        
        # Display onboarding summary
        onboarding_summary = self.session.get("step_1_summary")
        if onboarding_summary:
            self.view.display_subheader("Step 1: Data Onboarding")
            self.view.show_message(onboarding_summary, "success")
            
        # Display mapping summary
        mapping_summary = self.session.get("step_2_summary")
        if mapping_summary:
            self.view.display_subheader("Step 2: Column Mapping")
            self.view.show_message(mapping_summary, "success")
            
        # Display aggregation summary
        aggregation_summary = self.session.get("step_3_summary")
        if aggregation_summary:
            self.view.display_subheader("Step 3: Data Aggregation")
            self.view.show_message(aggregation_summary, "success")
            
        # Display join summary
        join_summary = self.session.get("step_4_summary")
        if join_summary:
            self.view.display_subheader("Step 4: Table Joining")
            self.view.show_message(join_summary, "success")
            
        self.view.display_markdown("---")
        
        # Now display clustering-specific completed steps
        if current_step <= 3:
            return
            
        for step in range(3, current_step):
            if self.session.get(f'step_{step}_complete'):
                self.view.display_header(f"Step {step}: {self.step_titles[step]}")
                self._display_step_summary(step)
                self.view.display_markdown("---")
    
    def _display_step_summary(self, step):
        """Display summary for a completed step"""
        summary = self.session.get(f'step_{step}_summary')
        if summary:
            self.view.show_message(summary, "success")
    
    def _advance_step(self):
        """Advance to the next step"""
        current_step = self.session.get('clustering_current_step', 3)
        self.session.set('clustering_current_step', current_step + 1)
        self.view.rerun_script()

    def _load_data_from_db(self):
        """Load necessary data from database and set in session"""
        try:
            from orchestrator.storage.db_connector import DatabaseConnector
            import sqlite3
            import pandas as pd
            import json
            
            db = DatabaseConnector()
            session_id = self.session.get('session_id')
            
            # 1. Get the final table name from join summary
            with sqlite3.connect(db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT final_table_name FROM join_summary WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
                    (session_id,)
                )
                result = cursor.fetchone()
                
                if not result:
                    self.view.show_message("❌ No join summary found in database", "error")
                    return False
                    
                final_table_name = result[0]
                logger.info(f"Found final table name: {final_table_name}")
                
                # 2. Load the final table data
                df = pd.read_sql(f"SELECT * FROM {final_table_name}", conn)
                
                if df.empty:
                    self.view.show_message("❌ Final table is empty", "error")
                    return False
                    
                logger.info(f"Loaded final table with shape: {df.shape}")
                
                # 3. Get field mappings from mapping summary
                cursor.execute(
                    "SELECT mappings FROM mappings_summary WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
                    (session_id,)
                )
                result = cursor.fetchone()
                
                if not result:
                    self.view.show_message("❌ No mapping summary found in database", "error")
                    return False
                    
                field_mappings = json.loads(result[0])
                logger.info(f"Loaded field mappings: {field_mappings}")
                
                # Ensure field_mappings has an 'id' field
                if 'id' not in field_mappings:
                    # Try to find a suitable ID column in the dataframe
                    potential_id_columns = ['id', 'ID', 'Id', 'user_id', 'customer_id', 'product_id']
                    
                    for col in potential_id_columns:
                        if col in df.columns:
                            field_mappings['id'] = col
                            logger.info(f"Using {col} as ID field")
                            break
                    
                    # If no suitable ID column found, use the first column
                    if 'id' not in field_mappings and len(df.columns) > 0:
                        field_mappings['id'] = df.columns[0]
                        logger.info(f"Using {df.columns[0]} as ID field (first column)")
                
                # 4. Store in session
                self.session.set('df', df)
                self.session.set('field_mappings', field_mappings)
                
                # 5. Also store the final table name
                self.session.set('final_table_name', final_table_name)
                
                # 6. Set df_aggregated (needed by step5_results_insights.py)
                self.session.set('df_aggregated', df)
                
                # 7. Also set raw_df which might be needed
                self.session.set('raw_df', df)
                
                # 8. Set up initial cluster mappings for each algorithm
                # This is needed by step5_results_insights.py
                for algorithm in ['kmeans', 'dbscan']:
                    if self.session.get(f'{algorithm}_cluster_mapping') is None:
                        # Create an empty DataFrame with id and cluster columns
                        cluster_mapping = pd.DataFrame({
                            'id': df[field_mappings['id']],
                            'cluster': [0] * len(df)  # Default cluster is 0
                        })
                        self.session.set(f'{algorithm}_cluster_mapping', cluster_mapping)
                
                # 9. Set selected_model if not already set
                if self.session.get('selected_model') is None:
                    self.session.set('selected_model', 'kmeans')  # Default to kmeans
                
                return True
                
        except Exception as e:
            logger.error(f"Error loading data from database: {str(e)}")
            self.view.show_message(f"❌ Error loading data: {str(e)}", "error")
            return False

    def _save_clustering_results(self):
        """Save clustering results to database and create state summary"""
        try:
            from orchestrator.storage.db_connector import DatabaseConnector
            import sqlite3
            import pandas as pd
            import json
            import numpy as np
            
            # Custom JSON encoder to handle NumPy types
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super(NumpyEncoder, self).default(obj)
            
            db = DatabaseConnector()
            session_id = self.session.get('session_id')
            selected_model = self.session.get('selected_model')
            
            if not selected_model:
                self.view.show_message("❌ No clustering model selected", "error")
                return False
            
            # Get the cluster mapping for the selected model
            cluster_mapping = self.session.get(f'{selected_model}_cluster_mapping')
            if cluster_mapping is None:
                self.view.show_message("❌ No cluster mapping found for selected model", "error")
                return False
            
            # Get the original data
            df = self.session.get('df')
            if df is None:
                self.view.show_message("❌ No data found", "error")
                return False
            
            # Get field mappings
            field_mappings = self.session.get('field_mappings')
            if field_mappings is None:
                self.view.show_message("❌ No field mappings found", "error")
                return False
            
            # Create a result dataframe with original data and cluster assignments
            result_df = df.copy()
            
            # Merge with cluster mapping
            id_field = field_mappings.get('id')
            if id_field not in result_df.columns:
                self.view.show_message(f"❌ ID field '{id_field}' not found in data", "error")
                return False
            
            # Ensure cluster_mapping has the right column name for merging
            cluster_mapping_copy = cluster_mapping.copy()
            if 'id' in cluster_mapping_copy.columns and id_field != 'id':
                cluster_mapping_copy = cluster_mapping_copy.rename(columns={'id': id_field})
            
            # Merge on ID field
            result_df = pd.merge(
                result_df,
                cluster_mapping_copy,
                on=id_field,
                how='left'
            )
            
            # Rename cluster column to be more descriptive
            result_df = result_df.rename(columns={'cluster': f'{selected_model}_cluster'})
            
            # Make sure the result_df has both the original ID field and 'id' if they're different
            if id_field != 'id' and 'id' not in result_df.columns:
                result_df['id'] = result_df[id_field]

            # Save to database
            with sqlite3.connect(db.db_path) as conn:
                # Create a standardized table name
                clustering_table_name = f"clustering_results_{session_id[:8]}"
                
                # Save to database
                result_df.to_sql(clustering_table_name, conn, if_exists='replace', index=False)
                logger.info(f"Saved clustering results to database: {clustering_table_name}")
                
                # Create clustering summary table if it doesn't exist
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS clustering_summary (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        selected_model TEXT NOT NULL,
                        num_clusters INTEGER NOT NULL,
                        result_table_name TEXT NOT NULL,
                        metrics TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Get number of clusters
                num_clusters = len(result_df[f'{selected_model}_cluster'].unique())
                
                # Get metrics if available
                metrics = self.session.get(f'{selected_model}_metrics', {})
                
                # Convert metrics to a serializable format
                serializable_metrics = {}
                for k, v in metrics.items():
                    if isinstance(v, (np.integer, np.floating, np.ndarray)):
                        if isinstance(v, np.ndarray):
                            serializable_metrics[k] = v.tolist()
                        else:
                            serializable_metrics[k] = float(v)
                    else:
                        serializable_metrics[k] = v
                
                # Insert or update summary
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO clustering_summary 
                    (session_id, selected_model, num_clusters, result_table_name, metrics)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        selected_model,
                        num_clusters,
                        clustering_table_name,
                        json.dumps(serializable_metrics, cls=NumpyEncoder)
                    )
                )
                
                conn.commit()
                
            # Create a summary for display
            summary = f"### Clustering Results\n\n"
            summary += f"**Selected Model:** {selected_model.upper()}\n"
            summary += f"**Number of Clusters:** {num_clusters}\n\n"
            
            if metrics:
                summary += "**Performance Metrics:**\n"
                for metric, value in serializable_metrics.items():
                    summary += f"- {metric}: {value}\n"
            
            # Store the summary in the session
            self.session.set("clustering_summary", summary)
            
            # Also store the result dataframe for download
            self.session.set("clustering_result_df", result_df)
            
            # Set both download_data and the specific model's cluster_mapping
            self.session.set('download_data', result_df)
            self.session.set(f'{selected_model}_cluster_mapping', cluster_mapping_copy)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving clustering results: {str(e)}")
            self.view.show_message(f"❌ Error saving clustering results: {str(e)}", "error")
            return False

if __name__ == "__main__":
    app = ClusteringOrchestrator()
    app.run() 