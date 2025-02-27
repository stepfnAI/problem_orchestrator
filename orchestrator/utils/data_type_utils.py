from typing import Dict, List, Any
import pandas as pd
import numpy as np

class DataTypeUtils:
    """Utility class for data type operations"""
    
    @staticmethod
    def classify_dtype(dtype) -> str:
        """
        Classify a pandas dtype into a category
        
        Args:
            dtype: The pandas dtype to classify
            
        Returns:
            String category: 'NUMERIC', 'TEXT', 'DATETIME', or 'OTHER'
        """
        dtype_str = str(dtype)
        
        if 'int' in dtype_str or 'float' in dtype_str:
            return 'NUMERIC'
        elif 'object' in dtype_str or 'string' in dtype_str:
            return 'TEXT'
        elif 'datetime' in dtype_str or 'date' in dtype_str:
            return 'DATETIME'
        else:
            return 'OTHER'
    
    @staticmethod
    def get_column_info(df: pd.DataFrame, exclude_columns: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get information about columns in a dataframe
        
        Args:
            df: The dataframe to analyze
            exclude_columns: List of column names to exclude
            
        Returns:
            Dict mapping column names to information about the column
        """
        if exclude_columns is None:
            exclude_columns = []
        
        result = {}
        
        for col in df.columns:
            if col in exclude_columns:
                continue
            
            dtype = df[col].dtype
            dtype_category = DataTypeUtils.classify_dtype(dtype)
            
            # Determine allowed aggregation methods based on data type
            allowed_methods = []
            
            if dtype_category == 'NUMERIC':
                allowed_methods = ['min', 'max', 'sum', 'mean', 'median', 'mode', 'last value']
            elif dtype_category == 'TEXT':
                allowed_methods = ['mode', 'last value', 'unique count']
            elif dtype_category == 'DATETIME':
                allowed_methods = ['min', 'max', 'last value']
            else:
                allowed_methods = ['last value']
            
            result[col] = {
                'dtype': str(dtype),
                'dtype_category': dtype_category,
                'allowed_methods': allowed_methods
            }
        
        return result 