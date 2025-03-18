from typing import Dict, Any, Optional, Union
import pandas as pd
import numpy as np
import logging
import json

logger = logging.getLogger(__name__)

class AgentOutputProcessor:
    """
    Processes outputs from different types of agents and applies transformations
    when necessary (executing code, applying aggregations, etc.)
    """
    
    def __init__(self, context_manager):
        self.context_manager = context_manager
        # Register processors for different agent types
        self.processors = {
            "aggregation_advisor": self.process_aggregation_advice,
            "target_generator": self.process_target_generation_output,
            "field_mapper": self.process_direct_mapping,
            # Add more as needed
        }
    
    def process(self, agent_name: str, agent_output: Dict[str, Any], 
                input_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Process output from any agent based on its type
        
        Args:
            agent_name: Name of the agent that produced the output
            agent_output: The raw output from the agent
            input_data: Optional DataFrame to apply transformations to
            
        Returns:
            Processed output, which may include transformed data
        """
        logger.info(f"Processing output from {agent_name}")
        
        if agent_name in self.processors:
            try:
                return self.processors[agent_name](agent_output, input_data)
            except Exception as e:
                logger.error(f"Error processing output from {agent_name}: {str(e)}")
                return {
                    "raw_output": agent_output,
                    "processing_error": str(e)
                }
        
        # Default: return as-is for unknown agent types
        logger.info(f"No specific processor for {agent_name}, returning raw output")
        return agent_output
    
    def process_aggregation_advice(self, advice: Dict[str, Any], 
                                  data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Process aggregation advice and apply if data is provided
        
        Args:
            advice: Aggregation advice from aggregation_advisor
            data: Optional DataFrame to apply aggregations to
            
        Returns:
            Dict containing raw advice and optionally aggregated data
        """
        # Store the raw advice
        processed_result = {"raw_advice": advice}
        
        if data is not None:
            # Apply the aggregation if data is provided
            try:
                aggregated_data = self._apply_aggregation(advice, data)
                processed_result["aggregated_data"] = aggregated_data
                processed_result["success"] = True
            except Exception as e:
                processed_result["success"] = False
                processed_result["error"] = str(e)
        
        return processed_result
    
    def process_target_generation_output(self, output, input_data=None):
        """Process output from target generator agent
        
        Args:
            output: Raw output from agent
            input_data: Optional DataFrame to apply transformations to
            
        Returns:
            Processed output
        """
        logger.info("Processing output from target_generator")
        print(f">>><<<1111 target output: {output}")
        # If output is already a dict, use it
        if isinstance(output, dict):
            processed_output = output.copy()
        else:
            # Try to parse JSON
            try:
                processed_output = json.loads(output)
            except:
                # If not JSON, return as is
                return output
        
        # Execute code if present
        if 'code' in processed_output:
            try:
                # Get the current DataFrame
                df = self.context_manager.get_current_dataframe()
                if df is None:
                    # Try to get any DataFrame
                    tables = list(self.context_manager.memory["tables"].keys())
                    if tables:
                        df = self.context_manager.get_table(tables[0])
                
                if df is not None:
                    # Create a copy to avoid modifying the original
                    df_copy = df.copy()
                    
                    # Create a local namespace with the DataFrame
                    local_vars = {'df': df_copy, 'pd': pd, 'np': np}
                    
                    # Execute the code
                    exec(processed_output['code'], globals(), local_vars)
                    
                    # Get the modified DataFrame
                    modified_df = local_vars['df']
                    
                    # Store the result in a new table
                    table_name = "df_with_target"
                    self.context_manager.add_table(table_name, modified_df)
                    try:
                        print(f">>><<<1111 modified_df: {modified_df.target.value_counts()}")
                    except:
                        pass
                    # Add the table name to the processed output
                    processed_output['result_table'] = table_name
                    
                    # Add a preview of the target column
                    if 'target' in modified_df.columns:
                        target_counts = modified_df['target'].value_counts().to_dict()
                        processed_output['target_counts'] = {str(k): int(v) for k, v in target_counts.items()}
                        
                        # Convert DataFrame to dict for JSON serialization
                        processed_output['result_preview'] = modified_df.head(5).to_dict(orient='records')
                    
                    logger.info(f"Successfully executed target generation code and created table '{table_name}'")
                else:
                    logger.error("No DataFrame available for target generation")
                    processed_output['error'] = "No DataFrame available for target generation"
            except Exception as e:
                logger.error(f"Error executing code: {str(e)}")
                processed_output['error'] = str(e)
        
        return processed_output
    
    def process_direct_mapping(self, mapping: Dict[str, Any], 
                              data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Process direct mapping output
        
        Args:
            mapping: Mapping output from field_mapper
            data: Optional DataFrame (not used for direct mappings)
            
        Returns:
            Processed mapping that can be used directly
        """
        # For direct mappings, we might want to validate or transform them
        # but generally they can be used as-is
        return {
            "mappings": mapping,
            "success": True
        }
    
    def _apply_aggregation(self, advice: Dict[str, Any], data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply aggregation advice to data
        
        Args:
            advice: Aggregation advice containing columns and methods
            data: DataFrame to apply aggregations to
            
        Returns:
            Aggregated DataFrame
        """
        if not advice:
            return data
        
        # Create a copy of the data to avoid modifying the original
        df_copy = data.copy()
        
        # Determine groupby columns (typically customer_id or similar)
        # This would ideally come from the context or be inferred
        groupby_cols = self._determine_groupby_columns(df_copy)
        
        if not groupby_cols:
            logger.warning("No groupby columns determined, returning original data")
            return df_copy
        
        # Create aggregation dictionary
        agg_dict = {}
        
        # Process each column's aggregation methods
        for column, agg_methods in advice.items():
            if column in df_copy.columns:
                # Extract method names from the advice
                methods = [item["method"].lower() for item in agg_methods]
                
                # Map method names to pandas aggregation functions
                pandas_methods = []
                for method in methods:
                    if method == "unique count":
                        pandas_methods.append("nunique")
                    elif method == "mode":
                        # Mode is handled separately
                        pandas_methods.append("mode")
                    elif method == "last value":
                        pandas_methods.append("last")
                    else:
                        # For min, max, sum, mean, median, etc.
                        pandas_methods.append(method)
                
                agg_dict[column] = pandas_methods
        
        # Apply aggregation
        if agg_dict:
            try:
                # Handle special case for mode
                mode_columns = []
                for col, methods in agg_dict.items():
                    if "mode" in methods:
                        mode_columns.append(col)
                        # Remove mode from methods to avoid errors
                        agg_dict[col] = [m for m in methods if m != "mode"]
                        # If no methods left, remove the column from agg_dict
                        if not agg_dict[col]:
                            del agg_dict[col]
                
                # Apply standard aggregations
                if agg_dict:
                    result = df_copy.groupby(groupby_cols).agg(agg_dict)
                else:
                    result = df_copy.groupby(groupby_cols).first()
                
                # Handle mode separately
                for col in mode_columns:
                    # Calculate mode for each group
                    mode_series = df_copy.groupby(groupby_cols)[col].apply(
                        lambda x: x.mode().iloc[0] if not x.mode().empty else None
                    )
                    # Add to result
                    result[f"{col}_mode"] = mode_series
                
                # Reset index to convert groupby columns back to regular columns
                result = result.reset_index()
                
                return result
            except Exception as e:
                logger.error(f"Error applying aggregation: {str(e)}")
                return df_copy
        
        return df_copy
    
    def _determine_groupby_columns(self, df: pd.DataFrame) -> list:
        """
        Determine appropriate groupby columns for aggregation
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of column names to use for groupby
        """
        # Try to get mappings from context
        mappings = self.context_manager.get_agent_output("field_mapper")
        
        groupby_cols = []
        
        # If mappings exist, use them to find ID columns
        if mappings:
            if isinstance(mappings, dict):
                # Look for common ID field names in mappings
                id_field_names = ['id', 'customer_id', 'user_id', 'client_id']
                for std_name, col_name in mappings.items():
                    if std_name.lower() in id_field_names:
                        groupby_cols.append(col_name)
        
        # If no groupby columns found from mappings, try to infer from column names
        if not groupby_cols:
            # Look for columns that might be IDs
            for col in df.columns:
                if 'id' in col.lower() and df[col].nunique() > df.shape[0] * 0.5:
                    groupby_cols.append(col)
                    break
        
        # If still no groupby columns, use the first column as a last resort
        if not groupby_cols and len(df.columns) > 0:
            groupby_cols = [df.columns[0]]
        
        return groupby_cols 