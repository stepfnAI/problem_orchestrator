from sfn_blueprint import Task, SFNValidateAndRetryAgent
from clustering_agent.agents.mapping_agent import SFNMappingAgent
from clustering_agent.agents.aggregation_agent import SFNAggregationAgent
from clustering_agent.utils.preprocessing_utils import (
    perform_imputation,
    perform_encoding,
    perform_scaling
)
from clustering_agent.config.model_config import DEFAULT_LLM_PROVIDER
import pandas as pd
from typing import Dict, List, Tuple

class MappingAndPreprocessing:
    def __init__(self, session_manager, view):
        self.session = session_manager
        self.view = view
        self.mapping_agent = SFNMappingAgent()
        self.aggregation_agent = SFNAggregationAgent()
        self.feature_info = {
            'numeric_features': [],
            'categorical_features': [],
            'skipped_features': []
        }
        
    def execute(self):
        """Execute the mapping and preprocessing step"""
        # Check if data is available
        df = self.session.get('df')
        if df is None:
            self.view.show_message("❌ No data found. Please upload data first.", "error")
            return False
            
        # Get step status
        mapping_complete = self.session.get('mapping_complete', False)
        mapping_confirmed = self.session.get('mapping_confirmed', False)
        preprocessing_complete = self.session.get('preprocessing_complete', False)
        
        # First handle mapping if not confirmed
        if not mapping_confirmed:
            mapping_result = self._handle_mapping()
            if mapping_result:
                # Continue to preprocessing instead of returning False
                return self.execute()
            return False
            
        # If mapping confirmed but preprocessing not done
        if mapping_confirmed and not preprocessing_complete:
            if not self._handle_preprocessing():
                return False
                
        # If preprocessing is complete, mark step as complete
        if preprocessing_complete and not mapping_complete:
            self.session.set('mapping_complete', True)
            self.session.set('step_2_complete', True)
            self._save_step_summary()
            return True
            
        return mapping_complete
        
    def _handle_mapping(self):
        """Handle field mapping with AI suggestions and manual options"""
        try:
            df = self.session.get('df')
            mapping_confirmed = self.session.get('mapping_confirmed', False)
            
            # Get AI suggestions for mapping if not already confirmed
            if not mapping_confirmed and not self.session.get('suggested_mappings'):
                with self.view.display_spinner('🤖 AI is mapping critical fields...'):
                    mapping_task = Task("Map columns", data=df)
                    validation_task = Task("Validate field mapping", data=df)
                    
                    validate_and_retry_agent = SFNValidateAndRetryAgent(
                        llm_provider=DEFAULT_LLM_PROVIDER,
                        for_agent='field_mapper'
                    )
                    
                    mappings, validation_message, is_valid = validate_and_retry_agent.complete(
                        agent_to_validate=self.mapping_agent,
                        task=mapping_task,
                        validation_task=validation_task,
                        method_name='execute_task',
                        get_validation_params='get_validation_params',
                        max_retries=2,
                        retry_delay=3.0
                    )
                    
                    if is_valid:
                        self.session.set('suggested_mappings', mappings)
                    else:
                        self.view.show_message("❌ AI couldn't generate valid field mappings.", "error")
                        return False
                        
            # Display mapping interface
            return self._display_mapping_interface()
            
        except Exception as e:
            self.view.show_message(f"Error in mapping: {str(e)}", "error")
            return False
            
    def _display_mapping_interface(self):
        """Display interface for verifying and modifying field mappings"""
        self.view.display_subheader("AI Suggested Critical Field Mappings")
        
        suggested_mappings = self.session.get('suggested_mappings', {})
        current_mappings = self.session.get('field_mappings', {})
        df = self.session.get('df')
        all_columns = list(df.columns)
        
        # Check for mandatory ID field
        if not suggested_mappings.get('id'):
            self.view.show_message(
                "⚠️ AI couldn't find ID field mapping. Please map it manually.",
                "warning"
            )
            return self._handle_manual_mapping(all_columns, current_mappings)
            
        # Display AI suggestions
        message = "🎯 AI Suggested Mappings:\n"
        for field, mapped_col in suggested_mappings.items():
            message += f"- {field}:  **{mapped_col or 'Not Found'}**\n"
            
        self.view.show_message(message, "info")
        self.view.display_markdown("---")
        
        # Show options to proceed
        action = self.view.radio_select(
            "How would you like to proceed?",
            options=[
                "Use AI Recommended Mappings",
                "Select Columns Manually"
            ],
            key="mapping_choice"
        )
        
        if action == "Use AI Recommended Mappings":
            if self.view.display_button("Confirm Mappings"):
                self.session.set('field_mappings', suggested_mappings)
                self.session.set('mapping_confirmed', True)
                return True
        else:
            return self._handle_manual_mapping(all_columns, current_mappings)
            
        return False
        
    def _handle_manual_mapping(self, all_columns: List[str], current_mappings: Dict):
        """Handle manual column mapping selection"""
        modified_mappings = {}
        suggested_mappings = self.session.get('suggested_mappings', {})
        
        # Required ID field
        self.view.display_subheader("Required Fields")
        modified_mappings['id'] = self.view.select_box(
            "Select ID column (required)",
            options=[""] + all_columns,
            # Use suggested mapping as default if available, otherwise use current mapping
            default=suggested_mappings.get('id') or current_mappings.get('id', "")
        )
        
        # Optional fields
        self.view.display_subheader("Optional Fields")
        optional_fields = ['product', 'revenue', 'date']
        for field in optional_fields:
            value = self.view.select_box(
                f"Select {field} column (optional)",
                options=[""] + all_columns,
                # Use suggested mapping as default if available, otherwise use current mapping
                default=suggested_mappings.get(field) or current_mappings.get(field, "")
            )
            if value:
                modified_mappings[field] = value
                
        if self.view.display_button("Confirm Mappings"):
            if not modified_mappings.get('id'):
                self.view.show_message("❌ ID field is mandatory.", "error")
                return False
                
            self.session.set('field_mappings', modified_mappings)
            self.session.set('mapping_confirmed', True)
            return True
            
        return False
        
    def _handle_preprocessing(self):
        """Handle all preprocessing steps sequentially"""
        try:
            df = self.session.get('df')
            mappings = self.session.get('field_mappings')
            
            # Show preprocessing progress
            self.view.display_subheader("Data Preprocessing")
            
            # 1. Handle aggregation if not complete
            if not self.session.get('aggregation_complete'):
                with self.view.display_spinner('Aggregating data...'):
                    if not self._handle_aggregation(df, mappings):
                        return False
                    df = self.session.get('df')  # Get updated df after aggregation
            
            # 2. Handle imputation if not complete
            if not self.session.get('imputation_complete'):
                with self.view.display_spinner('Handling missing values...'):
                    # Get missing value info
                    missing_info = df.isnull().sum()
                    missing_cols = missing_info[missing_info > 0]
                    
                    if len(missing_cols) > 0:
                        # Display missing value information
                        missing_msg = "Missing Values Found:\n"
                        for col, count in missing_cols.items():
                            percentage = (count / len(df)) * 100
                            missing_msg += f"- {col}: {count} ({percentage:.1f}%)\n"
                        self.view.show_message(missing_msg, "info")
                        
                        # Perform imputation
                        imputed_df = perform_imputation(df)
                        self.session.set('df', imputed_df)
                    
                    self.session.set('imputation_complete', True)
                    self.view.show_message("✅ Missing values handled", "success")
                    df = self.session.get('df')
            
            # 3. Handle encoding if not complete
            if not self.session.get('encoding_complete'):
                with self.view.display_spinner('Encoding categorical features...'):
                    mapped_cols = set(mappings.values())
                    categorical_cols = []
                    skipped_cols = []
                    
                    for col in df.columns:
                        if col not in mapped_cols and pd.api.types.is_object_dtype(df[col]):
                            if df[col].nunique() <= 20:
                                categorical_cols.append(col)
                            else:
                                skipped_cols.append(col)
                    
                    if categorical_cols or skipped_cols:
                        if categorical_cols:
                            encoded_df, encoded_features = perform_encoding(df, categorical_cols)
                            self.feature_info['categorical_features'] = encoded_features
                            self.session.set('df', encoded_df)
                            self.view.show_message(
                                f"✅ Encoded {len(categorical_cols)} categorical features",
                                "success"
                            )
                        if skipped_cols:
                            self.feature_info['skipped_features'].extend(skipped_cols)
                            self.view.show_message(
                                f"ℹ️ Skipped {len(skipped_cols)} high-cardinality columns",
                                "info"
                            )
                    else:
                        self.view.show_message("ℹ️ No categorical features to encode", "info")
                    
                    self.session.set('encoding_complete', True)
                    df = self.session.get('df')
            
            # 4. Handle scaling if not complete
            if not self.session.get('scaling_complete'):
                with self.view.display_spinner('Scaling numeric features...'):
                    mapped_cols = set(mappings.values())
                    numeric_cols = [
                        col for col in df.columns 
                        if col not in mapped_cols and pd.api.types.is_numeric_dtype(df[col])
                    ]
                    
                    if numeric_cols:
                        scaled_df = perform_scaling(df, numeric_cols)
                        self.feature_info['numeric_features'] = numeric_cols
                        self.session.set('df', scaled_df)
                        self.view.show_message(
                            f"✅ Scaled {len(numeric_cols)} numeric features",
                            "success"
                        )
                    
                    self.session.set('scaling_complete', True)
            
            # Check if all preprocessing is complete
            if all([
                self.session.get('aggregation_complete'),
                self.session.get('imputation_complete'),
                self.session.get('encoding_complete'),
                self.session.get('scaling_complete')
            ]):
                self.session.set('preprocessing_complete', True)
                self.view.show_message("✅ All preprocessing steps completed!", "success")
                
                # Add proceed button after preprocessing is complete
                if self.view.display_button("▶️ Proceed to Clustering Analysis"):
                    self.session.set('step_2_complete', True)
                    self._save_step_summary()
                    return True
            
            return False
            
        except Exception as e:
            self.view.show_message(f"Error in preprocessing: {str(e)}", "error")
            return False
            
    def _handle_aggregation(self, df: pd.DataFrame, mappings: Dict) -> bool:
        """Handle data aggregation if needed"""
        # Get ID field from mappings
        id_field = mappings.get('id')
        if not id_field:
            self.view.show_message("❌ ID field not found in mappings", "error")
            return False
        
        # Check if aggregation is needed
        records_per_id = df.groupby(id_field).size()
        max_records = records_per_id.max()
        
        if max_records == 1:
            # Store both raw and aggregated (same in this case)
            self.session.set('df_raw', df.copy())
            self.session.set('df_aggregated', df.copy())
            self.session.set('aggregation_complete', True)
            return True
        
        # Show aggregation info
        self.view.show_message(
            f"Found multiple records per ID (max: {max_records}). Data will be aggregated.",
            "info"
        )
        
        # Get aggregation rules from agent
        agg_task = Task("Get aggregation methods", data={
            'df': df,
            'id_field': id_field
        })
        
        try:
            with self.view.display_spinner('🤖 AI is determining aggregation methods...'):
                agg_rules = self.aggregation_agent.execute_task(agg_task)
            
            if not agg_rules:
                self.view.show_message("❌ Failed to get aggregation rules", "error")
                return False
            
            # Store raw data before aggregation
            self.session.set('df_raw', df.copy())
            
            # Apply aggregation
            aggregated_df = df.groupby(id_field).agg(agg_rules).reset_index()
            
            # Save aggregated data separately
            self.session.set('df_aggregated', aggregated_df)
            # Save to df for further processing
            self.session.set('df', aggregated_df)
            self.session.set('aggregation_complete', True)
            
            # Show success message
            self.view.show_message(
                f"✅ Data aggregated successfully. New shape: {aggregated_df.shape}",
                "success"
            )
            return True
            
        except Exception as e:
            self.view.show_message(f"Error in aggregation: {str(e)}", "error")
            return False
        
    def _save_step_summary(self):
        """Save step summary for display in completed steps"""
        mappings = self.session.get('field_mappings')
        df = self.session.get('df')
        
        summary = "✅ Step 2 Complete\n\n"
        
        # Add field mappings
        summary += "Field Mappings:\n"
        for field, col in mappings.items():
            if col:
                summary += f"- {field}: **{col}**\n"
                
        # Add preprocessing info
        summary += "\nPreprocessing Summary:\n"
        if self.session.get('aggregation_complete'):
            summary += "- ✓ Data Aggregated\n"
        if self.session.get('imputation_complete'):
            summary += "- ✓ Missing Values Imputed\n"
            
        # Add feature information
        if self.feature_info['numeric_features']:
            summary += f"- Numeric Features: {len(self.feature_info['numeric_features'])}\n"
        if self.feature_info['categorical_features']:
            summary += f"- Encoded Categories: {len(self.feature_info['categorical_features'])}\n"
        if self.feature_info['skipped_features']:
            summary += f"- Skipped Features: {len(self.feature_info['skipped_features'])}\n"
            
        self.session.set('step_2_summary', summary) 