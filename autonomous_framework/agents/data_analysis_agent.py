from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sfn_blueprint import SFNAgent, Task, SFNAIHandler, SFNPromptManager
import os
from orchestrator.config.model_config import MODEL_CONFIG, DEFAULT_LLM_PROVIDER
import json
from autonomous_framework.context_manager import ContextManager

class DataAnalysisAgent(SFNAgent):
    """Agent responsible for analyzing datasets to understand their structure and quality"""
    
    def __init__(self, llm_provider=DEFAULT_LLM_PROVIDER):
        super().__init__(name="Data Analyzer", role="Data Scientist")
        self.ai_handler = SFNAIHandler()
        self.llm_provider = llm_provider
        self.model_config = MODEL_CONFIG.get("data_analyzer", MODEL_CONFIG["data_mapper"])  # Fallback to data_mapper config
        self.context_manager = None
        
        # Initialize prompt manager with the correct path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels
        prompt_config_path = os.path.join(project_root, 'orchestrator', 'config', 'prompt_config.json')
        
        self.prompt_manager = SFNPromptManager(prompt_config_path)
        
    def set_context_manager(self, context_manager):
        """Set the context manager for this agent"""
        self.context_manager = context_manager
        
    def execute_task(self, task: Task) -> Dict[str, Any]:
        """
        Analyze a dataset to understand its structure and quality
        
        Args:
            task: Task object containing:
                - df: DataFrame to analyze or table name string
                - analysis_type: Type of analysis to perform (basic, detailed, relationships)
                - options: Additional options for analysis
                
        Returns:
            Dict containing analysis results
        """
        # Extract data from task
        task_data = task.data
        df_or_name = task_data.get('df')
        analysis_type = task_data.get('analysis_type', 'basic')
        options = task_data.get('options', {})
        
        # Handle case where df is a string (table name)
        if isinstance(df_or_name, str):
            # Get the DataFrame from the context manager
            if self.context_manager:
                df = self.context_manager.get_table(df_or_name)
            else:
                # Fallback to creating a new context manager
                context_manager = ContextManager()  # This will use the existing session
                df = context_manager.get_table(df_or_name)
            
            if df is None:
                raise ValueError(f"Table '{df_or_name}' not found in context")
        else:
            df = df_or_name
        
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Task data must contain a pandas DataFrame or a valid table name")
        
        # Perform analysis based on type
        if analysis_type == 'basic':
            analysis_results = self._perform_basic_analysis(df)
        elif analysis_type == 'detailed':
            analysis_results = self._perform_detailed_analysis(df)
        elif analysis_type == 'relationships':
            analysis_results = self._analyze_relationships(df)
        else:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")
        
        # Get AI insights if requested
        if options.get('get_ai_insights', False):
            ai_insights = self._get_ai_insights(df, analysis_results)
            analysis_results['ai_insights'] = ai_insights
            
        return analysis_results
    
    def _perform_basic_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform basic analysis of the dataset"""
        # Basic dataset information
        basic_info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / (1024 * 1024),  # in MB
        }
        
        # Missing values analysis
        missing_values = {
            'total_missing': df.isna().sum().sum(),
            'missing_by_column': df.isna().sum().to_dict(),
            'missing_percentage': (df.isna().sum() / len(df) * 100).to_dict()
        }
        
        # Duplicate rows analysis
        duplicates = {
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': df.duplicated().sum() / len(df) * 100
        }
        
        # Column types categorization
        column_types = {
            'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'datetime_columns': df.select_dtypes(include=['datetime']).columns.tolist(),
            'boolean_columns': df.select_dtypes(include=['bool']).columns.tolist()
        }
        
        return {
            'basic_info': basic_info,
            'missing_values': missing_values,
            'duplicates': duplicates,
            'column_types': column_types
        }
    
    def _perform_detailed_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform detailed analysis of the dataset"""
        # Start with basic analysis
        results = self._perform_basic_analysis(df)
        
        # Numeric columns statistics
        numeric_stats = {}
        for col in results['column_types']['numeric_columns']:
            numeric_stats[col] = {
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'skew': df[col].skew(),
                'kurtosis': df[col].kurtosis(),
                'zeros_count': (df[col] == 0).sum(),
                'zeros_percentage': (df[col] == 0).sum() / len(df) * 100,
                'negative_count': (df[col] < 0).sum(),
                'outliers_count': self._count_outliers(df[col])
            }
        
        # Categorical columns statistics
        categorical_stats = {}
        for col in results['column_types']['categorical_columns']:
            value_counts = df[col].value_counts()
            categorical_stats[col] = {
                'unique_values': df[col].nunique(),
                'most_common': value_counts.index[0] if not value_counts.empty else None,
                'most_common_count': value_counts.iloc[0] if not value_counts.empty else 0,
                'most_common_percentage': value_counts.iloc[0] / len(df) * 100 if not value_counts.empty else 0,
                'least_common': value_counts.index[-1] if not value_counts.empty and len(value_counts) > 1 else None,
                'least_common_count': value_counts.iloc[-1] if not value_counts.empty and len(value_counts) > 1 else 0,
                'entropy': self._calculate_entropy(df[col])
            }
        
        # Datetime columns statistics
        datetime_stats = {}
        for col in results['column_types']['datetime_columns']:
            datetime_stats[col] = {
                'min_date': df[col].min(),
                'max_date': df[col].max(),
                'range_days': (df[col].max() - df[col].min()).days if not pd.isna(df[col].max()) and not pd.isna(df[col].min()) else None,
                'most_common_year': df[col].dt.year.value_counts().index[0] if not df[col].dt.year.value_counts().empty else None,
                'most_common_month': df[col].dt.month.value_counts().index[0] if not df[col].dt.month.value_counts().empty else None,
                'most_common_day': df[col].dt.day.value_counts().index[0] if not df[col].dt.day.value_counts().empty else None
            }
        
        # Potential ID columns
        potential_id_columns = self._identify_potential_id_columns(df)
        
        # Add to results
        results['numeric_stats'] = numeric_stats
        results['categorical_stats'] = categorical_stats
        results['datetime_stats'] = datetime_stats
        results['potential_id_columns'] = potential_id_columns
        
        return results
    
    def _analyze_relationships(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze relationships between columns in the dataset"""
        # Start with basic analysis
        results = self._perform_basic_analysis(df)
        
        # Correlation analysis for numeric columns
        numeric_columns = results['column_types']['numeric_columns']
        if len(numeric_columns) > 1:
            correlation_matrix = df[numeric_columns].corr().to_dict()
            
            # Find highly correlated pairs
            high_correlation_pairs = []
            for i, col1 in enumerate(numeric_columns):
                for col2 in numeric_columns[i+1:]:
                    corr = correlation_matrix.get(col1, {}).get(col2, 0)
                    if abs(corr) > 0.7:  # Threshold for high correlation
                        high_correlation_pairs.append({
                            'column1': col1,
                            'column2': col2,
                            'correlation': corr
                        })
            
            results['correlation'] = {
                'matrix': correlation_matrix,
                'high_correlation_pairs': high_correlation_pairs
            }
        
        # Categorical relationship analysis (chi-square test)
        categorical_columns = results['column_types']['categorical_columns']
        if len(categorical_columns) > 1:
            categorical_relationships = []
            for i, col1 in enumerate(categorical_columns):
                for col2 in categorical_columns[i+1:]:
                    if df[col1].nunique() < 20 and df[col2].nunique() < 20:  # Limit to manageable categories
                        try:
                            from scipy.stats import chi2_contingency
                            contingency_table = pd.crosstab(df[col1], df[col2])
                            chi2, p_value, _, _ = chi2_contingency(contingency_table)
                            categorical_relationships.append({
                                'column1': col1,
                                'column2': col2,
                                'chi2': chi2,
                                'p_value': p_value,
                                'significant': p_value < 0.05
                            })
                        except Exception as e:
                            # Skip if chi-square test fails
                            pass
            
            results['categorical_relationships'] = categorical_relationships
        
        # Potential foreign key relationships
        potential_foreign_keys = self._identify_potential_foreign_keys(df)
        results['potential_foreign_keys'] = potential_foreign_keys
        
        return results
    
    def _count_outliers(self, series: pd.Series) -> int:
        """Count outliers using IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return ((series < lower_bound) | (series > upper_bound)).sum()
    
    def _calculate_entropy(self, series: pd.Series) -> float:
        """Calculate entropy of a categorical variable"""
        value_counts = series.value_counts(normalize=True)
        return -np.sum(value_counts * np.log2(value_counts))
    
    def _identify_potential_id_columns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify columns that might be IDs or keys"""
        potential_ids = []
        
        for col in df.columns:
            unique_count = df[col].nunique()
            unique_ratio = unique_count / len(df)
            
            # Check if column name contains typical ID keywords
            name_suggests_id = any(id_term in col.lower() for id_term in ['id', 'key', 'code', 'num', 'no'])
            
            # Check if values match typical ID patterns
            sample_values = df[col].dropna().sample(min(5, len(df))).astype(str).tolist()
            
            if (unique_ratio > 0.9 and unique_count > 10) or name_suggests_id:
                potential_ids.append({
                    'column': col,
                    'unique_count': unique_count,
                    'unique_ratio': unique_ratio,
                    'name_suggests_id': name_suggests_id,
                    'sample_values': sample_values,
                    'confidence': 'high' if unique_ratio > 0.9 and name_suggests_id else 'medium'
                })
        
        return potential_ids
    
    def _identify_potential_foreign_keys(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify columns that might be foreign keys"""
        potential_foreign_keys = []
        
        # Get potential ID columns
        potential_ids = self._identify_potential_id_columns(df)
        potential_id_columns = [item['column'] for item in potential_ids]
        
        for col in df.columns:
            # Skip columns already identified as potential primary keys
            if col in potential_id_columns:
                continue
                
            # Check if column name suggests a foreign key
            name_suggests_fk = any(fk_term in col.lower() for fk_term in ['_id', '_key', '_code', '_no', 'id_', 'key_'])
            
            # Check cardinality - foreign keys typically have fewer unique values than rows
            unique_count = df[col].nunique()
            unique_ratio = unique_count / len(df)
            
            if name_suggests_fk or (0.01 < unique_ratio < 0.9 and unique_count > 1):
                potential_foreign_keys.append({
                    'column': col,
                    'unique_count': unique_count,
                    'unique_ratio': unique_ratio,
                    'name_suggests_fk': name_suggests_fk,
                    'sample_values': df[col].dropna().sample(min(5, len(df))).astype(str).tolist(),
                    'confidence': 'high' if name_suggests_fk and 0.01 < unique_ratio < 0.9 else 'medium'
                })
        
        return potential_foreign_keys
    
    def _get_ai_insights(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get AI-generated insights about the dataset"""
        try:
            # Prepare context for the LLM
            context = {
                'dataset_info': {
                    'shape': analysis_results['basic_info']['shape'],
                    'columns': analysis_results['basic_info']['columns'],
                    'column_types': analysis_results['column_types'],
                    'missing_values': analysis_results['missing_values'],
                    'duplicates': analysis_results['duplicates']
                }
            }
            
            # Add more detailed information if available
            if 'numeric_stats' in analysis_results:
                context['numeric_stats'] = analysis_results['numeric_stats']
            
            if 'categorical_stats' in analysis_results:
                context['categorical_stats'] = analysis_results['categorical_stats']
            
            if 'potential_id_columns' in analysis_results:
                context['potential_id_columns'] = analysis_results['potential_id_columns']
            
            if 'potential_foreign_keys' in analysis_results:
                context['potential_foreign_keys'] = analysis_results['potential_foreign_keys']
            
            # Get prompts from prompt config
            prompts = self.prompt_manager.get_prompt(
                agent_type='data_analyzer',
                llm_provider=self.llm_provider,
                prompt_type='main',
                context=json.dumps(context)
            )
            
            # Call LLM to get insights
            configuration = self.model_config[self.llm_provider].copy()
            response = self.ai_handler.get_llm_response(
                system_prompt=prompts['system_prompt'],
                user_prompt=prompts['user_prompt_template'].format(context=json.dumps(context)),
                llm_provider=self.llm_provider,
                configuration=configuration
            )
            
            # Parse the response
            try:
                if isinstance(response, dict):
                    content = response['choices'][0]['message']['content']
                else:
                    content = response
                    
                # Clean the content string
                cleaned_str = content.strip()
                cleaned_str = cleaned_str.replace('```json', '').replace('```', '')
                
                # Extract JSON content
                start_idx = cleaned_str.find('{')
                end_idx = cleaned_str.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    cleaned_str = cleaned_str[start_idx:end_idx + 1]
                    
                # Parse JSON
                insights = json.loads(cleaned_str)
                
                return insights
            except:
                # If JSON parsing fails, return the raw response
                return {
                    'general_insights': [response],
                    'data_quality_issues': [],
                    'potential_relationships': [],
                    'recommendations': []
                }
            
        except Exception as e:
            print(f"Error getting AI insights: {str(e)}")
            return {
                'error': str(e),
                'general_insights': ["Failed to generate AI insights."],
                'data_quality_issues': [],
                'potential_relationships': [],
                'recommendations': []
            } 