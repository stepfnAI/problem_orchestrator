from typing import Dict, Any, List, Optional
import pandas as pd
from sfn_blueprint import SFNAgent, Task, SFNAIHandler
import os
import json

class DataLoadingAgent(SFNAgent):
    """Agent responsible for loading data from various sources"""
    
    def __init__(self):
        super().__init__(name="Data Loader", role="Data Engineer")
        
    def execute_task(self, task: Task) -> Dict[str, Any]:
        """
        Load data from a source
        
        Args:
            task: Task object containing:
                - source_type: Type of source (csv, excel, database)
                - source_path: Path to the source file or connection string
                - options: Additional options for loading
                
        Returns:
            Dict containing:
                - df: Loaded DataFrame
                - metadata: Metadata about the loaded data
        """
        source_type = task.data.get('source_type', 'csv')
        source_path = task.data.get('source_path')
        options = task.data.get('options', {})
        
        if not source_path:
            raise ValueError("Source path must be provided")
        
        # Load data based on source type
        if source_type.lower() == 'csv':
            df = self._load_csv(source_path, options)
        elif source_type.lower() == 'excel':
            df = self._load_excel(source_path, options)
        elif source_type.lower() == 'database':
            df = self._load_database(source_path, options)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
        
        # Generate metadata
        metadata = self._generate_metadata(df)
        
        return {
            'df': df,
            'metadata': metadata
        }
    
    def _load_csv(self, path: str, options: Dict) -> pd.DataFrame:
        """Load data from CSV file"""
        return pd.read_csv(path, **options)
    
    def _load_excel(self, path: str, options: Dict) -> pd.DataFrame:
        """Load data from Excel file"""
        sheet_name = options.pop('sheet_name', 0)
        return pd.read_excel(path, sheet_name=sheet_name, **options)
    
    def _load_database(self, connection_string: str, options: Dict) -> pd.DataFrame:
        """Load data from database"""
        query = options.get('query')
        if not query:
            raise ValueError("Query must be provided for database source")
        
        import sqlalchemy
        engine = sqlalchemy.create_engine(connection_string)
        return pd.read_sql(query, engine)
    
    def _generate_metadata(self, df: pd.DataFrame) -> Dict:
        """Generate metadata about the loaded data"""
        return {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'datetime_columns': df.select_dtypes(include=['datetime']).columns.tolist()
        } 