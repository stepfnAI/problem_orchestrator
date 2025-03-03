import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Fix the import paths
from orchestrator.states.usecase_states.recommendation.step3_feature_suggestion import FeatureSuggestion
from orchestrator.states.usecase_states.recommendation.step4_similarity_calculation import SimilarityCalculation
from orchestrator.states.usecase_states.recommendation.step5_recommendation_generation import RecommendationGeneration
from orchestrator.utils.streamlit_views import StreamlitView
from sfn_blueprint import SFNSessionManager
import logging
import numpy as np
import pandas as pd
import json
import sqlite3

logger = logging.getLogger(__name__)

class RecommendationOrchestrator:
    def __init__(self, session_manager, view):
        self.view = view
        self.session = session_manager
        self.orchestrators = {
            3: FeatureSuggestion(self.session, self.view),
            4: SimilarityCalculation(self.session, self.view),
            5: RecommendationGeneration(self.session, self.view)
        }
        self.step_titles = {
            3: "Feature Selection",
            4: "Similarity Calculation",
            5: "Recommendation Generation"
        }
        
    def run(self):
        """Main recommendation flow"""
        logger.info("Starting Recommendation Orchestrator")
        
        # Load necessary data from database if not already in session
        if self.session.get('df') is None or self.session.get('field_mappings') is None:
            if not self._load_data_from_db():
                self.view.show_message("❌ Failed to load required data from database", "error")
                return False
        
        # Get current step (starting from step 3 since steps 1-2 are already done)
        current_step = self.session.get('recommendation_current_step', 3)
        
        # Display completed steps
        self._display_completed_steps(current_step)
        
        # Display current step header
        if current_step in self.step_titles:
            self.view.display_header(f"Step {current_step}: {self.step_titles[current_step]}")
            self.view.display_markdown("---")
            
            # Execute current step
            if current_step in self.orchestrators:
                logger.info(f"Executing recommendation step {current_step}: {self.step_titles[current_step]}")
                result = self.orchestrators[current_step].execute()
                
                # Only advance if not the final step and step is complete
                if result and current_step < max(self.orchestrators.keys()):
                    self._advance_step()
                # For final step, save state summary and results to database
                elif result and current_step == max(self.orchestrators.keys()):
                    self._save_recommendation_results()
                    return True
        else:
            self.view.show_message("❌ Invalid step. Resetting application.", "error")
            self.session.set('recommendation_current_step', 3)
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
        
        # Now display recommendation-specific completed steps
        if current_step <= 3:
            return
            
        for step in range(3, current_step):
            if self.session.get(f'recommendation_step_{step}_complete'):
                self.view.display_header(f"Step {step}: {self.step_titles[step]}")
                self._display_step_summary(step)
                self.view.display_markdown("---")
    
    def _display_step_summary(self, step):
        """Display summary for a completed step"""
        summary = self.session.get(f'recommendation_step_{step}_summary')
        if summary:
            self.view.show_message(summary, "success")
    
    def _advance_step(self):
        """Advance to the next step"""
        current_step = self.session.get('recommendation_current_step', 3)
        self.session.set('recommendation_current_step', current_step + 1)
        self.view.rerun_script()

    def _load_data_from_db(self):
        """Load necessary data from database and set in session"""
        try:
            from orchestrator.storage.db_connector import DatabaseConnector
            import sqlite3
            import pandas as pd
            import json
            
            db = DatabaseConnector()
            self.db = db  # Store for later use
            session_id = self.session.get('session_id')
            
            # 1. Get the final table name from join summary
            with sqlite3.connect(db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT final_table_name, final_mappings FROM join_summary WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
                    (session_id,)
                )
                result = cursor.fetchone()
                
                if not result:
                    self.view.show_message("❌ No join summary found in database", "error")
                    return False
                    
                final_table_name = result[0]
                final_mappings_json = result[1]
                print(f">><<Found final mappings in recommendation orchestrator: {final_mappings_json}")
                logger.info(f"Found final table name: {final_table_name}")
                logger.info(f"Found final mappings: {final_mappings_json}")
                
                # 2. Load the final table data
                df = pd.read_sql(f"SELECT * FROM {final_table_name}", conn)
                
                if df.empty:
                    self.view.show_message("❌ Final table is empty", "error")
                    return False
                    
                logger.info(f"Loaded final table with shape: {df.shape}")
                
                # Store in session before fetching mappings
                self.session.set('df', df)
                self.session.set('final_table_name', final_table_name)
                
                # 3. Load the final mappings if available
                if final_mappings_json:
                    try:
                        field_mappings = json.loads(final_mappings_json)
                        logger.info(f"Loaded field mappings from join summary: {field_mappings}")
                        self.session.set('field_mappings', field_mappings)
                    except Exception as e:
                        logger.error(f"Error parsing final mappings: {str(e)}")
                        # We'll try to get mappings from mapping summary as fallback
                        field_mappings = None
                else:
                    field_mappings = None
                    
                # 4. If we couldn't get mappings from join summary, try mapping summary as fallback
                if not field_mappings:
                    # Try to get mappings from mapping summary
                    cursor.execute(
                        "SELECT mappings FROM mappings_summary WHERE session_id = ? AND table_name = '_final_mappings' ORDER BY created_at DESC LIMIT 1",
                        (session_id,)
                    )
                    mapping_result = cursor.fetchone()
                    
                    if mapping_result:
                        try:
                            field_mappings = json.loads(mapping_result[0])
                            logger.info(f"Loaded field mappings from mapping summary: {field_mappings}")
                            self.session.set('field_mappings', field_mappings)
                        except Exception as e:
                            logger.error(f"Error parsing mappings from mapping summary: {str(e)}")
                            field_mappings = None
                
                # 5. If we still don't have mappings, try to get them from the original mapping state
                if not field_mappings:
                    field_mappings = self._fetch_mappings_from_db()
                    
                    if not field_mappings:
                        self.view.show_message("❌ Could not retrieve field mappings", "error")
                        return False
                
                # 6. Initialize feature metadata for recommendation
                self._initialize_feature_metadata(df)
                
                return True
                
        except Exception as e:
            logger.error(f"Error loading data from database: {str(e)}")
            self.view.show_message(f"❌ Error loading data: {str(e)}", "error")
            return False

    def _initialize_feature_metadata(self, df):
        """Initialize feature metadata for recommendation"""
        try:
            # Create feature metadata dictionary
            feature_metadata = {}
            
            for col in df.columns:
                is_numeric = pd.api.types.is_numeric_dtype(df[col])
                unique_count = df[col].nunique()
                
                feature_metadata[col] = {
                    'is_numeric': is_numeric,
                    'unique_count': unique_count,
                    'can_use': True,  # Default to True, can be modified by user
                    'dtype': str(df[col].dtype)
                }
                
            # Store in session
            self.session.set('feature_metadata', feature_metadata)
            
            # Initialize empty recommended features
            self.session.set('recommended_features', [])
            # Initialize approach
            self.session.set('recommendation_approach', self._get_recommendation_approach())
            logger.info(f"Initialized feature metadata for {len(feature_metadata)} columns")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing feature metadata: {str(e)}")
            return False

    def _get_recommendation_approach(self):
        """Fetch recommendation approach from onboarding summary"""
        try:
            # Try to get from session first (if already loaded)
            approach = self.session.get('recommendation_approach')
            if approach:
                print(f">><<Using recommendation approach from session: {approach}")
                return approach
            
            # If not in session, fetch from onboarding summary
            session_id = self.session.get('session_id')
            summary = self.db.fetch_state_summary('onboarding', session_id)
            print(f">><<Summary: {summary}")
            if summary and 'recommendation_approach' in summary and summary['recommendation_approach']:
                approach = summary['recommendation_approach']
                print(f">><<Loaded recommendation approach from onboarding summary: {approach}")
                return approach
            
            # Default to item_based if not found
            print(">><<No recommendation approach found, defaulting to item_based")
            return 'item_based'
            
        except Exception as e:
            print(f">><<Error fetching recommendation approach: {str(e)}")
            return 'item_based'  # Default to item_based on error

    def _save_recommendation_results(self):
        """Save recommendation results to database and create state summary"""
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
            
            # Get recommendations
            recommendations = self.session.get('recommendations', [])
            
            if not recommendations:
                self.view.show_message("❌ No recommendations generated", "error")
                return False
            
            # Get approach
            approach = self.session.get('recommendation_approach', 'user_based')
            
            # Get the original data
            df = self.session.get('df')
            if df is None:
                self.view.show_message("❌ No data found", "error")
                return False
            
            # Create a recommendations dataframe
            reco_df = pd.DataFrame(recommendations)
            
            # Save to database
            with sqlite3.connect(db.db_path) as conn:
                # Create a standardized table name
                reco_table_name = f"recommendation_results_{session_id[:8]}"
                
                # Save to database
                reco_df.to_sql(reco_table_name, conn, if_exists='replace', index=False)
                logger.info(f"Saved recommendation results to database: {reco_table_name}")
                
                # Create recommendation summary table if it doesn't exist
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS recommendation_summary (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        approach TEXT NOT NULL,
                        num_recommendations INTEGER NOT NULL,
                        result_table_name TEXT NOT NULL,
                        features_used TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Get features used
                features_used = self.session.get('selected_features', [])
                
                # Insert or update summary
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO recommendation_summary 
                    (session_id, approach, num_recommendations, result_table_name, features_used)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        approach,
                        len(recommendations),
                        reco_table_name,
                        json.dumps(features_used, cls=NumpyEncoder)
                    )
                )
                
                conn.commit()
                
            # Create a summary for display
            summary = f"### Recommendation Results\n\n"
            summary += f"**Approach:** {approach.replace('_', ' ').title()}\n"
            summary += f"**Number of Recommendations:** {len(recommendations)}\n\n"
            
            if features_used:
                summary += "**Features Used:**\n"
                for feature in features_used:
                    summary += f"- {feature}\n"
            
            # Store the summary in the session
            self.session.set("recommendation_summary", summary)
            
            # Also store the result dataframe for download
            self.session.set("recommendation_result_df", reco_df)
            self.session.set("download_data", reco_df)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving recommendation results: {str(e)}")
            self.view.show_message(f"❌ Error saving recommendation results: {str(e)}", "error")
            return False

    def _fetch_mappings_from_db(self):
        """Fetch and process field mappings from database
        
        Returns:
            dict: Processed field mappings with all required fields
        """
        try:
            from orchestrator.storage.db_connector import DatabaseConnector
            import sqlite3
            import json
            
            db = DatabaseConnector()
            session_id = self.session.get('session_id')
            
            with sqlite3.connect(db.db_path) as conn:
                cursor = conn.cursor()
                
                # First try to get table-specific mappings
                cursor.execute(
                    "SELECT table_name, mappings FROM mappings_summary WHERE session_id = ? AND table_name != '_state_summary'",
                    (session_id,)
                )
                results = cursor.fetchall()
                
                # If no table-specific mappings, try to get from state summary
                if not results:
                    cursor.execute(
                        "SELECT mappings FROM mappings_summary WHERE session_id = ? AND table_name = '_state_summary'",
                        (session_id,)
                    )
                    state_result = cursor.fetchone()
                    
                    if state_result:
                        # Parse state summary mappings
                        state_mappings = json.loads(state_result[0])
                        logger.info(f">><<Found mappings in state summary: {state_mappings}")
                        
                        # Extract any field mappings from state summary
                        field_mappings = {}
                        # Look for any key that might be a field mapping
                        for key, value in state_mappings.items():
                            if key not in ['tables_mapped', 'mandatory_columns_mapped', 'prediction_level', 
                                          'has_product_mapping', 'problem_type', 'completion_time']:
                                field_mappings[key] = value
                    else:
                        # No mappings found at all
                        logger.error("No mappings found in database")
                        return None
                else:
                    # Combine mappings from all tables
                    field_mappings = {}
                    for table_name, mappings_json in results:
                        table_mappings = json.loads(mappings_json)
                        field_mappings.update(table_mappings)
                
                logger.info(f">><<Raw field mappings: {field_mappings}")
                
                # Get the dataframe to check columns
                df = self.session.get('df')
                if df is None:
                    logger.error("DataFrame not loaded yet")
                    return field_mappings  # Return what we have so far
                
                # Ensure field_mappings has required fields for recommendation
                approach = self.session.get('recommendation_approach')
                logger.info(f">><<Current recommendation approach: {approach}")
                
                # Define required fields based on approach
                required_fields = ['product_id']
                if approach == 'user_based':
                    required_fields.append('id')  # User ID field
                
                # Special case: If we have 'cust_id' but not 'id' for user-based, use cust_id as id
                if approach == 'user_based' and 'cust_id' in field_mappings:
                    field_mappings['id'] = field_mappings['cust_id']
                    logger.info(f">><<Using cust_id mapping for id: {field_mappings['cust_id']}")
                
                logger.info(f">><<Final field mappings: {field_mappings}")
                return field_mappings
                
        except Exception as e:
            logger.error(f"Error fetching mappings: {str(e)}")
            return None

if __name__ == "__main__":
    app = RecommendationOrchestrator()
    app.run() 