from sfn_blueprint import Task, SFNValidateAndRetryAgent

from orchestrator.agents.reco_feature_suggestion_agent import SFNFeatureSuggestionAgent
from orchestrator.config.model_config import DEFAULT_LLM_PROVIDER
import pandas as pd
from typing import Dict

class FeatureSuggestion:
    def __init__(self, session_manager, view):
        self.session = session_manager
        self.view = view
        self.feature_agent = SFNFeatureSuggestionAgent()
        
    def analyze_features(self, df: pd.DataFrame) -> Dict:
        """Analyze and filter features suitable for similarity calculation"""
        feature_metadata = {}
        
        for column in df.columns:
            is_numeric = pd.api.types.is_numeric_dtype(df[column])
            unique_count = df[column].nunique()
            
            feature_metadata[column] = {
                'is_numeric': is_numeric,
                'unique_count': unique_count,
                'can_use': is_numeric or (not is_numeric and unique_count <= 10)
            }
        
        return feature_metadata

    def _get_ai_suggestions(self):
        """Get AI suggestions for features"""
        try:
            df = self.session.get('df')
            mappings = self.session.get('field_mappings')
            feature_metadata = self.analyze_features(df)
            self.session.set('feature_metadata', feature_metadata)
            
            with self.view.display_spinner('🤖 AI is analyzing features...'):
                task_data = {
                    'df': df,
                    'mappings': mappings,
                    'feature_metadata': feature_metadata
                }
                
                task = Task("Suggest features", data=task_data)
                validation_task = Task("Validate suggestions", data=task_data)
                
                validate_and_retry_agent = SFNValidateAndRetryAgent(
                    llm_provider=DEFAULT_LLM_PROVIDER,
                    for_agent='feature_suggester'
                )
                
                suggestions, validation_message, is_valid = validate_and_retry_agent.complete(
                    agent_to_validate=self.feature_agent,
                    task=task,
                    validation_task=validation_task,
                    method_name='execute_task',
                    get_validation_params='get_validation_params',
                    max_retries=2,
                    retry_delay=3.0
                )
                
                if is_valid:
                    self.session.set('feature_suggestions', suggestions)
                    return True
                else:
                    self.view.show_message("❌ Error getting feature suggestions", "error")
                    return False
                    
        except Exception as e:
            self.view.show_message(f"❌ Error in AI suggestions: {str(e)}", "error")
            return False

    def execute(self):
        """Execute the feature suggestion step"""
        # Check prerequisites
        df = self.session.get('df')
        mappings = self.session.get('field_mappings')
        
        if df is None or mappings is None:
            self.view.show_message("❌ Please complete previous steps first.", "error")
            return False
        
        # Get feature suggestions if not already present
        if not self.session.get('feature_suggestions'):
            if not self._get_ai_suggestions():
                return False
        
        # Only move to next step if both features and weights are confirmed
        feature_complete = self.session.get('feature_complete', False)
        weights_confirmed = self.session.get('weights_confirmed', False)
        
        if feature_complete and weights_confirmed:
            return True
            
        return self._display_feature_interface()
        
    def _display_feature_interface(self):
        """Display interface for feature selection"""
        suggestions = self.session.get('feature_suggestions', {})
        current_state = self.session.get('feature_interface_state', 'feature_selection')
        
        self.view.display_subheader("Feature Selection")
        
        if current_state == 'feature_selection':
            self._handle_feature_selection(suggestions)
            # Always return False to prevent step completion
            return False
        elif current_state == 'weight_assignment':
            weights_result = self._handle_weight_assignment(suggestions)
            # Only return True if weights are confirmed
            return weights_result
            
        return False
        
    def _handle_feature_selection(self, suggestions):
        """Handle interactive feature selection"""
        # Get all available features
        df = self.session.get('df')
        feature_metadata = self.analyze_features(df)
        available_features = [
            col for col, meta in feature_metadata.items() 
            if meta['can_use']
        ]
        excluded_features = [
            col for col, meta in feature_metadata.items() 
            if not meta['can_use']
        ]
        recommended_features = [f['feature_name'] for f in suggestions.get('recommended_features', [])]
        
        # Display AI recommendations
        self.view.display_markdown("### Step 1: Select Important Features")
        
        # Display feature info
        numerical_features = sum(1 for _, meta in feature_metadata.items() if meta['is_numeric'])
        categorical_features = sum(1 for _, meta in feature_metadata.items() 
                                 if not meta['is_numeric'] and meta['unique_count'] <= 10)
        
        self.view.display_markdown("**ℹ️ Available Features:**")
        self.view.display_markdown(
            f"* 🔢 {numerical_features} numerical features\n"
            f"* 📊 {categorical_features} categorical features (≤10 unique values)"
        )
        
        if excluded_features:
            self.view.display_markdown(
                f"ℹ️ {len(excluded_features)} features excluded (too many unique values)"
                )

        self.view.display_markdown("\n### AI Recommendations:")
        for feature in suggestions.get('recommended_features', []):
            self.view.display_markdown(
                f"- **{feature['feature_name']}** (Importance: {feature['importance']})\n  "
                f"_{feature['reasoning']}_"
            )
        
        # Get stored selected features or use recommended ones
        temp_selected_features = self.session.get('temp_selected_features', recommended_features)
        
        # Let user modify feature selection
        selected_features = self.view.multiselect(
            "Modify Feature Selection:",
            options=available_features,
            default=temp_selected_features
        )
        
        # Update selected features immediately
        self.session.set('temp_selected_features', selected_features)
        
        # Handle feature confirmation
        if self.view.display_button("✅ Confirm Features"):
            with self.view.display_spinner('🔄 Preparing feature weights...'):
                # Get weights for all selected features using LLM
                task_data = {
                    'df': df,
                    'existing_features': [],  # Empty since we want weights for all features
                    'new_features': selected_features,  # All selected features
                    'existing_weights': {}  # Empty since we want fresh weights
                }
                task = Task("Get weights for selected features", data=task_data)
                weight_suggestions = self.feature_agent.get_feature_weights(task)
                
                # Update suggestions with new features and weights
                updated_suggestions = {
                    'recommended_features': [],
                    'feature_weights': weight_suggestions.get('weights', {}),
                    'excluded_features': suggestions.get('excluded_features', [])
                }
                
                # Update recommended features list with new information
                for feature in selected_features:
                    updated_suggestions['recommended_features'].append({
                        'feature_name': feature,
                        'importance': weight_suggestions.get('importance', {}).get(feature, 'medium'),
                        'reasoning': weight_suggestions.get('reasoning', {}).get(feature, 'Selected by user')
                    })
                
                # Update session state
                self.session.set('feature_suggestions', updated_suggestions)
                self.session.set('features_confirmed', True)
                self.session.set('feature_interface_state', 'weight_assignment')
                
                # Force a rerun using the view's rerun method
                self.view.rerun_script()
        
        return False
    
    def _handle_weight_assignment(self, suggestions):
        """Handle interactive weight assignment"""
        self.view.display_markdown("### Step 2: Assign Feature Weights")
        self.view.display_markdown("Adjust the importance weights for each selected feature:")
        
        updated_weights = {}
        selected_features = [f['feature_name'] for f in suggestions.get('recommended_features', [])]
        
        for feature in selected_features:
            default_weight = suggestions.get('feature_weights', {}).get(feature, 1.0)
            # Round the default weight to match slider step of 0.05
            default_weight = round(float(default_weight), 2)
            weight = self.view.slider(
                f"Weight for {feature}",
                min_value=0.0,
                max_value=1.0,
                value=default_weight,
                step=0.05,  # Changed to 0.05 to allow finer control
                key=f"weight_{feature}"
            )
            updated_weights[feature] = weight
            
        if self.view.display_button("✅ Confirm Weights"):
            suggestions['feature_weights'] = updated_weights
            self.session.set('feature_suggestions', suggestions)
            
            # Display summary
            self.view.display_subheader("✅ Feature Selection Summary")
            summary_msg = "Selected Features and Weights:\n"
            for feature in selected_features:
                weight = updated_weights[feature]
                importance = next(
                    (f['importance'] for f in suggestions['recommended_features'] 
                     if f['feature_name'] == feature), 
                    'medium'
                )
                summary_msg += f"- **{feature}**\n"
                summary_msg += f"  - Weight: {weight:.2f}\n"  # Changed to 2 decimal places
                summary_msg += f"  - Importance: {importance}\n"
            
            self.view.show_message(summary_msg, "success")
            
            # Set completion states but don't advance step yet
            self.session.set('weights_confirmed', True)
            self.session.set('feature_complete', True)
            self.session.set('step_3_complete', True)
            self._save_step_summary()
            
            # Add a proceed button to explicitly move to next step
            if self.view.display_button("➡️ Proceed to Similarity Calculation"):
                return True
            
        return False
        
    def _update_feature_suggestions(self, suggestions, selected_features):
        """Update feature suggestions based on user selection"""
        # Keep existing recommended features that are still selected
        suggestions['recommended_features'] = [
            f for f in suggestions.get('recommended_features', [])
            if f['feature_name'] in selected_features
        ]
        
        # Handle new user-selected features
        existing_features = [f['feature_name'] for f in suggestions['recommended_features']]
        new_features = [f for f in selected_features if f not in existing_features]
        
        if new_features:
            # Get AI suggestions for new features
            task_data = {
                'df': self.session.get('df'),
                'existing_features': existing_features,
                'new_features': new_features,
                'existing_weights': suggestions.get('feature_weights', {})
            }
            
            task = Task("Suggest weights for new features", data=task_data)
            llm_suggestions = self.feature_agent.get_feature_weights(task)
            print("\nRaw LLM Response:", llm_suggestions)
            
            # Add new features with AI-suggested weights and reasoning
            for feature in new_features:
                print(f"\nProcessing feature: {feature}")
                try:
                    # Get weight from LLM response
                    weight = llm_suggestions.get('weights', {}).get(feature)
                    print(f"Found weight in LLM response: {weight}")
                    
                    if weight is None:
                        print("No weight found in LLM response, using fallback")
                        weight = 0.5
                    
                    importance = llm_suggestions.get('importance', {}).get(feature, 'medium')
                    reasoning = llm_suggestions.get('reasoning', {}).get(feature, 'Added by user')
                    
                    print(f"Final values for {feature}:")
                    print(f"- Weight: {weight}")
                    print(f"- Importance: {importance}")
                    print(f"- Reasoning: {reasoning}")
                    
                    # Add to recommendations
                    suggestions['recommended_features'].append({
                        'feature_name': feature,
                        'importance': importance,
                        'reasoning': reasoning
                    })
                    
                    # Add weight to feature_weights
                    suggestions['feature_weights'][feature] = weight
                    
                except Exception as e:
                    print(f"Error processing feature {feature}: {str(e)}")
                    # Fallback values
                    suggestions['recommended_features'].append({
                        'feature_name': feature,
                        'importance': 'medium',
                        'reasoning': 'Added by user (fallback)'
                    })
                    suggestions['feature_weights'][feature] = 0.5
        
        return suggestions
        
    def _save_step_summary(self):
        """Save step summary"""
        suggestions = self.session.get('feature_suggestions', {})
        
        summary = "✅ Step 3 Complete\n\n"
        summary += "Selected Features and Weights:\n"
        
        for feature in suggestions.get('recommended_features', []):
            feature_name = feature['feature_name']
            weight = suggestions.get('feature_weights', {}).get(feature_name, 0.0)
            summary += (
                f"- **{feature_name}**\n"
                f"  - Weight: {weight:.1f}\n"
                f"  - Importance: {feature['importance']}\n"
            )
            
        self.session.set('step_3_summary', summary)

    def _get_available_features(self, df):
        """Get features available for similarity calculation"""
        feature_metadata = self.analyze_features(df)
        
        # Filter features that can be used
        available_features = [
            col for col, meta in feature_metadata.items() 
            if meta['can_use']
        ]
        
        # Store metadata for later use
        self.session.set('feature_metadata', feature_metadata)
        return available_features 