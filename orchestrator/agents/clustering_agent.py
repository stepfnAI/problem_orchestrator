from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from sfn_blueprint import SFNAgent, Task, SFNAIHandler, SFNPromptManager
import os
from orchestrator.config.model_config import MODEL_CONFIG, DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER
import json

class SFNClusteringAgent(SFNAgent):
    """Agent responsible for generating and executing clustering code"""
    
    SUPPORTED_ALGORITHMS = {
        'kmeans': {
            'metrics': [
                'silhouette_score',
                'within_cluster_sum_squares',
                'cluster_sizes'
            ]
        },
        'dbscan': {
            'metrics': [
                'silhouette_score',
                'within_cluster_sum_squares',
                'cluster_sizes',
                'n_clusters'  # Number of clusters found (excluding noise points)
            ]
        }
    }
    
    def __init__(self, llm_provider=DEFAULT_LLM_PROVIDER):
        super().__init__(name="Clustering Agent", role="Data Scientist")
        self.ai_handler = SFNAIHandler()
        self.llm_provider = llm_provider
        self.model_config = MODEL_CONFIG["clustering_agent"]
        parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        prompt_config_path = os.path.join(parent_path, 'config', 'prompt_config.json')
        self.prompt_manager = SFNPromptManager(prompt_config_path)
        
    def execute_task(self, task: Task, error_context: str = None) -> Dict:
        """Generate clustering code with optional error context for retries"""
        if not isinstance(task.data, dict):
            raise ValueError("Task data must be a dictionary")
            
        df_info = task.data.get('df_info')
        algorithm = task.data.get('algorithm')
        custom_instructions = task.data.get('custom_instructions', '')
        
        if df_info is None or algorithm is None:
            raise ValueError("Missing required data or algorithm")
            
        # Add error context to prompt if provided
        if error_context:
            custom_instructions += f"\nPrevious attempt failed with error: {error_context}"
            
        # Get clustering code from LLM
        code_response = self._get_clustering_code(
            df_info=df_info,
            algorithm=algorithm,
            custom_instructions=custom_instructions
        )
        
        return code_response

    def _get_clustering_code(self, df_info: Dict, algorithm: str, custom_instructions: str) -> Dict:
        """Get clustering code from LLM"""
        # Get metrics descriptions for algorithm
        metrics_descriptions = "\n      ".join([
            f"- {metric}" for metric in self.SUPPORTED_ALGORITHMS[algorithm]['metrics']
        ])
        
        # Format shape and features
        shape = df_info['shape']
        n_rows = int(shape[0])
        n_cols = int(shape[1])
        
        # Get numeric features and ID field
        numeric_features = df_info.get('numeric_features', [])
        id_field = df_info.get('id_field')  # Get ID field from df_info
        
        if not numeric_features:
            # Fallback to filtering by dtype if numeric_features not provided
            numeric_features = [
                col for col, dtype in df_info['dtypes'].items()
                if 'float' in dtype.lower() or 'int' in dtype.lower()
            ]
        
        # Debug prints
        print("DEBUG: Using numeric features:", numeric_features)
        print("DEBUG: Using ID field:", id_field)
        
        # Add algorithm-specific constraints
        algorithm_constraints = self._get_algorithm_constraints(
            algorithm, 
            n_samples=df_info['shape'][0],
            n_features=len(df_info['numeric_features'])
        )
        
        # Prepare kwargs for prompt
        prompt_kwargs = {
            'algorithm': algorithm,
            'rows': n_rows,
            'cols': n_cols,
            'features': ", ".join(numeric_features),
            'id_field': id_field,  # Add ID field to prompt kwargs
            'metrics_descriptions': metrics_descriptions,
            'algorithm_constraints': algorithm_constraints,
            'custom_instructions': (
                "IMPORTANT: Use ONLY these numeric features for clustering: " + 
                ", ".join(numeric_features) + 
                "\nDO NOT use any other columns like cust_id, billing_date, or product_id."
            )
        }
        
        system_prompt, user_prompt = self.prompt_manager.get_prompt(
            agent_type='clustering_agent',
            llm_provider=self.llm_provider,
            prompt_type='main',
            **prompt_kwargs
        )
        
        # Debug print
        print("DEBUG: Generated prompt:", user_prompt)
        
        provider_config = self.model_config.get(self.llm_provider)
        configuration = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": provider_config["temperature"],
            "max_tokens": provider_config["max_tokens"]
        }
        
        response, _ = self.ai_handler.route_to(
            llm_provider=self.llm_provider,
            configuration=configuration,
            model=provider_config['model']
        )
        
        return self._parse_code_response(response)

    def _get_algorithm_constraints(self, algorithm: str, n_samples: int, n_features: int) -> str:
        """Get algorithm-specific constraints"""
        if algorithm == 'kmeans':
            max_k = min(15, int(np.sqrt(n_samples/2)))
            return (
                f"1. Number of clusters (k):\n"
                f"   - Minimum: 2 clusters\n"
                f"   - Maximum: {max_k} clusters (min(15, sqrt(n_samples/2)))\n"
                f"   - If optimal k falls outside range, use nearest boundary value\n\n"
                f"Required Warning Messages:\n"
                f"- If k=2 is used: Add warning 'Data suggests limited clustering patterns'\n"
                f"- If k={max_k} is used: Add warning 'Data suggests more granular patterns exist'\n"
                f"- If silhouette score < 0.2: Add warning 'Weak clustering structure detected'"
            )
        elif algorithm == 'dbscan':
            min_min_samples = max(2 * n_features, 5)
            return (
                f"1. DBSCAN Parameters:\n"
                f"   - Use sklearn.neighbors.NearestNeighbors to find optimal eps\n"
                f"   - min_samples: minimum {min_min_samples}\n\n"
                f"2. Parameter Selection:\n"
                f"   - Calculate distances using NearestNeighbors(n_neighbors=2)\n"
                f"   - Sort distances to find optimal eps value\n"
                f"   - Try multiple eps values if needed\n\n"
                f"Required Warning Messages:\n"
                f"- If >80% points are noise: Add warning 'Too many noise points, consider adjusting parameters'\n"
                f"- If only 1 cluster found: Add warning 'Single cluster detected, try different parameters'\n"
                f"- If silhouette score < 0: Add warning 'Poor cluster separation detected'"
            )
        return ""

    def _parse_code_response(self, response) -> Dict:
        """Parse LLM response into code and explanation"""
        try:
            if isinstance(response, dict):
                content = response['choices'][0]['message']['content']
            else:
                content = response
                
            # Clean the response
            cleaned_str = content.strip()
            cleaned_str = cleaned_str.replace('```json', '').replace('```', '')
            start_idx = cleaned_str.find('{')
            end_idx = cleaned_str.rfind('}')
            if start_idx != -1 and end_idx != -1:
                cleaned_str = cleaned_str[start_idx:end_idx + 1]
            
            parsed = json.loads(cleaned_str)
            return {
                'code': parsed.get('code', ''),
                'explanation': parsed.get('explanation', '')
            }
            
        except Exception as e:
            print(f"Error parsing LLM response: {str(e)}")
            return {
                'code': '',
                'explanation': f'Error parsing response: {str(e)}'
            }

    def get_validation_params(self, response, task):
        """Get parameters for validation"""
        prompts = self.prompt_manager.get_prompt(
            agent_type='clustering_agent',
            llm_provider=self.llm_provider,
            prompt_type='validation',
            actual_output=response
        )
        return prompts 