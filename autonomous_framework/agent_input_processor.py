from typing import Dict, Any, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class AgentInputProcessor:
    """
    Processes and standardizes inputs for different agents to ensure compatibility
    and proper data handling before agent execution.
    """
    
    def __init__(self, context_manager):
        """Initialize the input processor
        
        Args:
            context_manager: The context manager instance
        """
        self.context_manager = context_manager
        
        # Register input processors for different agent types
        self.processors = {
            "data_analyzer": self.process_data_analysis_input,
            "field_mapper": self.process_mapping_input,
            "target_generator": self.process_target_generation_input,
            "aggregation_advisor": self.process_aggregation_input,
            # Add more as needed
        }
    
    def process(self, agent_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process inputs for a specific agent
        
        Args:
            agent_name: Name of the agent
            inputs: Raw inputs for the agent
            
        Returns:
            Processed inputs suitable for the agent
        """
        # Get the appropriate processor for this agent
        processor = self.processors.get(agent_name, self.process_default_input)
        
        # Process the inputs
        try:
            processed_inputs = processor(inputs)
            logger.info(f"Processed inputs for {agent_name}")
            return processed_inputs
        except Exception as e:
            logger.error(f"Error processing inputs for {agent_name}: {str(e)}")
            # Return original inputs if processing fails
            return inputs
    
    def process_default_input(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Default input processor for agents without specific processors
        
        Args:
            inputs: Raw inputs
            
        Returns:
            Processed inputs
        """
        # Handle common input transformations
        processed_inputs = inputs.copy()
        
        # If 'df' is a string (table name), try to convert to actual DataFrame
        if 'df' in processed_inputs and isinstance(processed_inputs['df'], str):
            table_name = processed_inputs['df']
            
            # First check if this is a table in memory
            if table_name in self.context_manager.memory["tables"]:
                logger.info(f"Found table '{table_name}' in memory")
                # For some agents, we might want to keep the table name instead of the DataFrame
                # So we'll leave it as is and let the specific agent processor decide
            else:
                # Try to get the table from context
                try:
                    df = self.context_manager.get_table(table_name)
                    if df is not None:
                        processed_inputs['df'] = df
                        logger.info(f"Converted table name '{table_name}' to DataFrame")
                    else:
                        logger.warning(f"Table '{table_name}' not found in context")
                except Exception as e:
                    logger.error(f"Error getting table '{table_name}': {str(e)}")
        
        return processed_inputs
    
    def process_data_analysis_input(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process inputs for data analysis agent
        
        Args:
            inputs: Raw inputs
            
        Returns:
            Processed inputs
        """
        processed_inputs = inputs.copy()  # Start fresh, don't use process_default_input
        
        # If 'df' is a string (table name), convert to actual DataFrame
        if 'df' in processed_inputs and isinstance(processed_inputs['df'], str):
            table_name = processed_inputs['df']
            logger.info(f"Attempting to get DataFrame for table '{table_name}'")
            
            # Try to get the actual DataFrame
            df = self.context_manager.get_table(table_name)
            if df is not None and isinstance(df, pd.DataFrame):
                processed_inputs['df'] = df
                logger.info(f"Successfully retrieved DataFrame with shape {df.shape}")
            else:
                logger.warning(f"Could not get DataFrame for '{table_name}', using current dataframe")
                # Fall back to current dataframe
                current_df = self.context_manager.get_current_dataframe()
                if current_df is not None:
                    processed_inputs['df'] = current_df
                    logger.info(f"Using current DataFrame with shape {current_df.shape}")
        
        # If still no dataframe, try to get the current one
        if 'df' not in processed_inputs or processed_inputs['df'] is None or (isinstance(processed_inputs['df'], str)):
            current_df = self.context_manager.get_current_dataframe()
            if current_df is not None:
                processed_inputs['df'] = current_df
                logger.info(f"Using current DataFrame with shape {current_df.shape}")
            else:
                # Last resort: try to get any available dataframe
                tables = list(self.context_manager.memory["tables"].keys())
                if tables:
                    for table_name in tables:
                        df = self.context_manager.get_table(table_name)
                        if isinstance(df, pd.DataFrame):
                            processed_inputs['df'] = df
                            logger.info(f"Using DataFrame '{table_name}' with shape {df.shape}")
                            break
        
        # Final check to ensure we have a DataFrame
        if 'df' not in processed_inputs or not isinstance(processed_inputs['df'], pd.DataFrame):
            logger.error("Failed to get a valid DataFrame for data_analyzer")
        
        return processed_inputs
    
    def process_mapping_input(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process inputs for field mapping agent
        
        Args:
            inputs: Raw inputs
            
        Returns:
            Processed inputs
        """
        processed_inputs = inputs.copy()  # Start fresh
        
        # Ensure we have a DataFrame to work with
        if 'df' in processed_inputs and isinstance(processed_inputs['df'], str):
            table_name = processed_inputs['df']
            logger.info(f"Attempting to get DataFrame for table '{table_name}'")
            
            # Try to get the actual DataFrame
            df = self.context_manager.get_table(table_name)
            if df is not None and isinstance(df, pd.DataFrame):
                processed_inputs['df'] = df
                logger.info(f"Successfully retrieved DataFrame with shape {df.shape} for field_mapper")
            else:
                logger.warning(f"Could not get DataFrame for '{table_name}', using current dataframe")
                # Fall back to current dataframe
                current_df = self.context_manager.get_current_dataframe()
                if current_df is not None:
                    processed_inputs['df'] = current_df
                    logger.info(f"Using current DataFrame with shape {current_df.shape} for field_mapper")
        
        # If still no dataframe, try to get the current one
        if 'df' not in processed_inputs or processed_inputs['df'] is None or isinstance(processed_inputs['df'], str):
            current_df = self.context_manager.get_current_dataframe()
            if current_df is not None:
                processed_inputs['df'] = current_df
                logger.info(f"Using current DataFrame with shape {current_df.shape} for field_mapper")
            else:
                # Last resort: try to get any available dataframe
                tables = list(self.context_manager.memory["tables"].keys())
                if tables:
                    for table_name in tables:
                        df = self.context_manager.get_table(table_name)
                        if isinstance(df, pd.DataFrame):
                            processed_inputs['df'] = df
                            logger.info(f"Using DataFrame '{table_name}' with shape {df.shape} for field_mapper")
                            break
        
        # Ensure we have a problem type
        if 'problem_type' not in processed_inputs:
            # Try to infer from goal
            goal = self.context_manager.get_current_goal()
            if 'churn' in goal.lower():
                processed_inputs['problem_type'] = 'churn_prediction'
            elif 'recommend' in goal.lower() or 'suggestion' in goal.lower():
                processed_inputs['problem_type'] = 'recommendation'
            else:
                processed_inputs['problem_type'] = 'general'
        
        # Final check to ensure we have a DataFrame
        if 'df' not in processed_inputs or not isinstance(processed_inputs['df'], pd.DataFrame):
            logger.error("Failed to get a valid DataFrame for field_mapper")
        
        return processed_inputs
    
    def process_target_generation_input(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process inputs for target generation agent
        
        Args:
            inputs: Raw inputs
            
        Returns:
            Processed inputs
        """
        processed_inputs = inputs.copy()  # Start fresh
        
        # Ensure we have a DataFrame to work with
        if 'df' in processed_inputs and isinstance(processed_inputs['df'], str):
            table_name = processed_inputs['df']
            logger.info(f"Attempting to get DataFrame for table '{table_name}'")
            
            # Try to get the actual DataFrame
            df = self.context_manager.get_table(table_name)
            if df is not None and isinstance(df, pd.DataFrame):
                processed_inputs['df'] = df
                logger.info(f"Successfully retrieved DataFrame with shape {df.shape} for target_generator")
            else:
                logger.warning(f"Could not get DataFrame for '{table_name}', using current dataframe")
                # Fall back to current dataframe
                current_df = self.context_manager.get_current_dataframe()
                if current_df is not None:
                    processed_inputs['df'] = current_df
                    logger.info(f"Using current DataFrame with shape {current_df.shape} for target_generator")
        
        # If still no dataframe, try to get the current one
        if 'df' not in processed_inputs or processed_inputs['df'] is None or isinstance(processed_inputs['df'], str):
            current_df = self.context_manager.get_current_dataframe()
            if current_df is not None:
                processed_inputs['df'] = current_df
                logger.info(f"Using current DataFrame with shape {current_df.shape} for target_generator")
            else:
                # Last resort: try to get any available dataframe
                tables = list(self.context_manager.memory["tables"].keys())
                if tables:
                    for table_name in tables:
                        df = self.context_manager.get_table(table_name)
                        if isinstance(df, pd.DataFrame):
                            processed_inputs['df'] = df
                            logger.info(f"Using DataFrame '{table_name}' with shape {df.shape} for target_generator")
                            break
        
        # Ensure we have a target column name
        if 'target_column' not in processed_inputs:
            processed_inputs['target_column'] = 'target'
        
        # Ensure we have a problem type
        if 'problem_type' not in processed_inputs:
            goal = self.context_manager.get_current_goal()
            if 'churn' in goal.lower():
                processed_inputs['problem_type'] = 'churn_prediction'
            else:
                processed_inputs['problem_type'] = 'general'
        
        # Final check to ensure we have a DataFrame
        if 'df' not in processed_inputs or not isinstance(processed_inputs['df'], pd.DataFrame):
            logger.error("Failed to get a valid DataFrame for target_generator")
        
        return processed_inputs
    
    def process_aggregation_input(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process inputs for aggregation advisor agent
        
        Args:
            inputs: Raw inputs
            
        Returns:
            Processed inputs
        """
        processed_inputs = inputs.copy()  # Start fresh
        
        # Convert 'table' to 'df' if present
        if 'table' in processed_inputs and isinstance(processed_inputs['table'], str):
            table_name = processed_inputs['table']
            logger.info(f"Attempting to get DataFrame for table '{table_name}'")
            
            # Try to get the actual DataFrame
            df = self.context_manager.get_table(table_name)
            if df is not None and isinstance(df, pd.DataFrame):
                # Store the DataFrame as 'df' instead of 'table'
                processed_inputs['df'] = df
                # Remove the table name to avoid confusion
                del processed_inputs['table']
                logger.info(f"Successfully retrieved DataFrame with shape {df.shape} for aggregation_advisor")
            else:
                logger.warning(f"Could not get DataFrame for '{table_name}', using current dataframe")
                # Fall back to current dataframe
                current_df = self.context_manager.get_current_dataframe()
                if current_df is not None:
                    processed_inputs['df'] = current_df
                    # Remove the table name to avoid confusion
                    if 'table' in processed_inputs:
                        del processed_inputs['table']
                    logger.info(f"Using current DataFrame with shape {current_df.shape} for aggregation_advisor")
        
        # If still no dataframe, try to get the current one
        if 'df' not in processed_inputs or processed_inputs['df'] is None:
            current_df = self.context_manager.get_current_dataframe()
            if current_df is not None:
                processed_inputs['df'] = current_df
                logger.info(f"Using current DataFrame with shape {current_df.shape} for aggregation_advisor")
            else:
                # Last resort: try to get any available dataframe
                tables = list(self.context_manager.memory["tables"].keys())
                if tables:
                    for table_name in tables:
                        df = self.context_manager.get_table(table_name)
                        if isinstance(df, pd.DataFrame):
                            processed_inputs['df'] = df
                            logger.info(f"Using DataFrame '{table_name}' with shape {df.shape} for aggregation_advisor")
                            break
        
        # Add default groupby_columns if not present
        if 'df' in processed_inputs and isinstance(processed_inputs['df'], pd.DataFrame):
            df = processed_inputs['df']
            if 'groupby_columns' not in processed_inputs:
                # Try to infer groupby columns
                groupby_cols = self._infer_groupby_columns(df)
                processed_inputs['groupby_columns'] = groupby_cols
                logger.info(f"Inferred groupby columns: {groupby_cols}")
        
        # Final check to ensure we have a DataFrame
        if 'df' not in processed_inputs or not isinstance(processed_inputs['df'], pd.DataFrame):
            logger.error("Failed to get a valid DataFrame for aggregation_advisor")
            # Provide a simple empty DataFrame as a last resort to avoid NoneType errors
            processed_inputs['df'] = pd.DataFrame({'dummy': [1, 2, 3]})
            logger.warning("Using dummy DataFrame as fallback")
        
        return processed_inputs
    
    def _infer_groupby_columns(self, df: pd.DataFrame) -> list:
        """Infer columns that might be good for groupby operations
        
        Args:
            df: DataFrame
            
        Returns:
            List of column names
        """
        groupby_cols = []
        
        # Look for ID columns
        id_patterns = ['id', 'key', 'code', 'customer', 'user']
        for col in df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in id_patterns):
                # Check if column has reasonable cardinality
                if 0.01 < df[col].nunique() / len(df) < 0.9:
                    groupby_cols.append(col)
        
        # If no ID columns found, look for categorical columns with reasonable cardinality
        if not groupby_cols:
            for col in df.columns:
                if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                    if 0.01 < df[col].nunique() / len(df) < 0.5:
                        groupby_cols.append(col)
        
        # If still no columns found, use the first column as a last resort
        if not groupby_cols and len(df.columns) > 0:
            groupby_cols = [df.columns[0]]
        
        return groupby_cols 