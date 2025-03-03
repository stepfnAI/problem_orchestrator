from typing import Dict, List
import pandas as pd
from sfn_blueprint import SFNAgent, Task, SFNAIHandler, SFNPromptManager
import os
from orchestrator.config.model_config import MODEL_CONFIG, DEFAULT_LLM_PROVIDER
import json

class SFNFeatureSuggestionAgent(SFNAgent):
    """Agent responsible for suggesting features for user similarity calculation"""
    
    def __init__(self, llm_provider=DEFAULT_LLM_PROVIDER):
        super().__init__(name="Feature Suggester", role="Recommendation Engineer")
        self.ai_handler = SFNAIHandler()
        self.llm_provider = llm_provider
        self.model_config = MODEL_CONFIG["feature_suggester"]
        
        parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        prompt_config_path = os.path.join(parent_path, 'config', 'prompt_config.json')
        self.prompt_manager = SFNPromptManager(prompt_config_path)

    def execute_task(self, task: Task) -> Dict:
        """Suggest features for similarity calculation"""
        if not isinstance(task.data, dict):
            raise ValueError("Task data must be a dictionary")

        df = task.data.get('df')
        approach = task.data.get('approach', 'user_based')
        entity = 'item' if approach == 'item_based' else 'user'
        
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Task data must contain a DataFrame")

        # Get feature metadata
        feature_metadata = task.data.get('feature_metadata', {})
        feature_stats = self._get_feature_statistics(df)
        
        # Format features with their types
        features_with_types = []
        for col in feature_metadata:
            if feature_metadata[col]['can_use']:
                feature_type = "numeric" if feature_metadata[col]['is_numeric'] else f"categorical ({feature_metadata[col]['unique_count']} values)"
                features_with_types.append(f"{col} ({feature_type})")

        suggestions = self._get_feature_suggestions(
            features=[col for col, meta in feature_metadata.items() if meta['can_use']],
            feature_stats=feature_stats,
            feature_metadata=feature_metadata,
            approach=approach,
            entity=entity,
            features_with_types=features_with_types
        )
        
        return suggestions

    def _get_feature_statistics(self, df: pd.DataFrame, features: List[str] = None) -> Dict:
        """Calculate statistics for features
        
        Args:
            df: DataFrame containing the features
            features: Optional list of specific features to analyze. If None, analyzes all columns.
        """
        stats = {}
        # If no specific features provided, use all columns
        features_to_analyze = features if features is not None else df.columns
        
        for feature in features_to_analyze:
            if feature in df.columns:
                feature_data = df[feature]
                try:
                    stats[feature] = {
                        "dtype": str(feature_data.dtype),
                        "null_percentage": feature_data.isnull().mean(),
                        "unique_ratio": feature_data.nunique() / len(df),
                        "sample_values": feature_data.value_counts().head(3).to_dict(),
                        "nunique": feature_data.nunique(),
                        "missing_pct": (feature_data.isna().sum() / len(feature_data)) * 100
                    }
                    
                    # Add correlation info for numeric features
                    if feature_data.dtype.kind in 'biufc':
                        stats[feature]['correlation'] = df.select_dtypes(
                            include=['number']
                        ).corr()[feature].to_dict()
                    else:
                        stats[feature]['correlation'] = {}
                        
                except Exception as e:
                    print(f"Error calculating stats for {feature}: {str(e)}")
                    stats[feature] = {
                        "error": str(e),
                        "dtype": str(feature_data.dtype)
                    }
        
        return stats

    def _get_feature_suggestions(self, features, feature_stats, feature_metadata, approach, entity, features_with_types):
        """Get feature suggestions from LLM"""
        system_prompt, user_prompt = self.prompt_manager.get_prompt(
            agent_type='feature_suggester',
            llm_provider=self.llm_provider,
            prompt_type='main',
            features=features,
            feature_stats=feature_stats,
            feature_metadata=feature_metadata,
            approach=approach,
            entity=entity,
            features_with_types=features_with_types
        )

        configuration = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.model_config[self.llm_provider]["temperature"],
            "max_tokens": self.model_config[self.llm_provider]["max_tokens"]
        }

        response, _ = self.ai_handler.route_to(
            llm_provider=self.llm_provider,
            configuration=configuration,
            model=self.model_config[self.llm_provider]["model"]
        )
        print(f">>>Response: {response}")
        print(f">>>Response type: {type(response)}")
        return self._parse_response(response)

    def _parse_response(self, response) -> Dict:
        """Parse LLM response"""
        try:
            # Extract content from response
            if isinstance(response, dict):
                content = response['choices'][0]['message']['content']
            elif hasattr(response, 'choices'):
                content = response.choices[0].message.content
            else:
                content = response

            # Clean the content string
            cleaned_str = content.strip()
            cleaned_str = cleaned_str.replace('```json', '').replace('```', '')
            start_idx = cleaned_str.find('{')
            end_idx = cleaned_str.rfind('}')
            if start_idx != -1 and end_idx != -1:
                cleaned_str = cleaned_str[start_idx:end_idx + 1]

            # Parse JSON
            try:
                parsed_content = json.loads(cleaned_str)
            except json.JSONDecodeError as je:
                print(f"JSON Decode Error: {str(je)}")
                print(f"Problematic content: {cleaned_str}")
                return {
                    "recommended_features": [],
                    "excluded_features": [],
                    "feature_weights": {}
                }

            return self._clean_feature_suggestions(parsed_content)

        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            print(f"Raw response: {response}")
            return {
                "recommended_features": [],
                "excluded_features": [],
                "feature_weights": {}
            }

    def _clean_feature_suggestions(self, suggestions: Dict) -> Dict:
        """Clean and validate feature suggestions from LLM"""
        try:
            # Validate recommended features
            cleaned_recommended = []
            for feature in suggestions.get('recommended_features', []):
                if all(k in feature for k in ['feature_name', 'importance', 'reasoning']):
                    # Ensure importance is one of expected values
                    importance = feature['importance'].lower()
                    if importance not in ['high', 'medium', 'low']:
                        importance = 'medium'
                    
                    cleaned_recommended.append({
                        'feature_name': str(feature['feature_name']).strip(),
                        'importance': importance,
                        'reasoning': str(feature['reasoning']).strip()
                    })

            # Validate excluded features
            cleaned_excluded = []
            for feature in suggestions.get('excluded_features', []):
                if all(k in feature for k in ['feature_name', 'reason']):
                    cleaned_excluded.append({
                        'feature_name': str(feature['feature_name']).strip(),
                        'reason': str(feature['reason']).strip()
                    })

            # Validate feature weights
            cleaned_weights = {}
            weights = suggestions.get('feature_weights', {})
            if weights:
                # Ensure weights are valid numbers and sum to 1
                total_weight = sum(float(w) for w in weights.values() if isinstance(w, (int, float)))
                if total_weight > 0:
                    for feature, weight in weights.items():
                        if isinstance(weight, (int, float)) and weight > 0:
                            cleaned_weights[str(feature).strip()] = float(weight) / total_weight

            return {
                'recommended_features': cleaned_recommended,
                'excluded_features': cleaned_excluded,
                'feature_weights': cleaned_weights
            }

        except Exception as e:
            print(f"Error cleaning feature suggestions: {str(e)}")
            return {
                'recommended_features': [],
                'excluded_features': [],
                'feature_weights': {}
            }

    def get_validation_params(self, response, task):
        """Get validation parameters"""
        return self.prompt_manager.get_prompt(
            agent_type='feature_suggester',
            llm_provider=self.llm_provider,
            prompt_type='validation',
            actual_output=response
        )

    def get_feature_weights(self, task: Task) -> Dict:
        """Get weight suggestions for all features including new ones"""
        df = task.data['df']
        existing_features = task.data['existing_features']
        new_features = task.data['new_features']
        existing_weights = task.data['existing_weights']
        approach = task.data.get('approach', 'user_based')
        entity = 'item' if approach == 'item_based' else 'user'
        
        # Combine all features for weight calculation
        all_features = existing_features + new_features
        
        # Get feature metadata and format features with types
        feature_metadata = task.data.get('feature_metadata', {})
        features_with_types = []
        for feature in all_features:
            if feature in feature_metadata:
                meta = feature_metadata[feature]
                feature_type = "numeric" if meta['is_numeric'] else f"categorical ({meta['unique_count']} values)"
                features_with_types.append(f"{feature} ({feature_type})")

        system_prompt, user_prompt = self.prompt_manager.get_prompt(
            agent_type='feature_suggester',
            llm_provider=self.llm_provider,
            prompt_type='weight_suggestion',
            existing_features=existing_features,
            existing_weights=existing_weights,
            new_features=new_features,
            all_features=all_features,
            feature_stats=self._get_feature_statistics(df, all_features),
            entity=entity,
            features_with_types=features_with_types
        )
        
        configuration = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.model_config[self.llm_provider]["temperature"],
            "max_tokens": self.model_config[self.llm_provider]["max_tokens"]
        }
        
        response, _ = self.ai_handler.route_to(
            llm_provider=self.llm_provider,
            configuration=configuration,
            model=self.model_config[self.llm_provider]["model"]
        )
        
        return self._parse_weight_response(response)
        
    def _parse_weight_response(self, response) -> Dict:
        """Parse LLM response specifically for weight suggestions"""
        try:
            # Extract content from response
            if isinstance(response, dict):
                content = response['choices'][0]['message']['content']
            elif hasattr(response, 'choices'):
                content = response.choices[0].message.content
            else:
                content = response

            # Clean the content string
            cleaned_str = content.strip()
            cleaned_str = cleaned_str.replace('```json', '').replace('```', '')
            start_idx = cleaned_str.find('{')
            end_idx = cleaned_str.rfind('}')
            if start_idx != -1 and end_idx != -1:
                cleaned_str = cleaned_str[start_idx:end_idx + 1]

            # Parse JSON
            try:
                parsed_content = json.loads(cleaned_str)
                print("Weight suggestion parsed content:", parsed_content)
                
                # Clean and validate the weights response
                cleaned_response = {
                    'weights': {},
                    'importance': {},
                    'reasoning': {}
                }
                
                if 'weights' in parsed_content:
                    cleaned_response['weights'] = parsed_content['weights']
                if 'importance' in parsed_content:
                    cleaned_response['importance'] = parsed_content['importance']
                if 'reasoning' in parsed_content:
                    cleaned_response['reasoning'] = parsed_content['reasoning']
                
                return cleaned_response

            except json.JSONDecodeError as je:
                print(f"JSON Decode Error in weight suggestion: {str(je)}")
                print(f"Problematic content: {cleaned_str}")
                return {
                    'weights': {},
                    'importance': {},
                    'reasoning': {}
                }

        except Exception as e:
            print(f"Error parsing weight suggestion response: {str(e)}")
            print(f"Raw response: {response}")
            return {
                'weights': {},
                'importance': {},
                'reasoning': {}
            } 