from sfn_blueprint import Task
import pandas as pd
import numpy as np
from orchestrator.utils.similarity_utils import calculate_cosine_similarity, get_similar_users, calculate_item_similarity
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class SimilarityCalculation:
    def __init__(self, session_manager, view):
        self.session = session_manager
        self.view = view
        
    def execute(self):
        """Execute similarity calculation step"""
        # Get required data from session
        df = self.session.get('df')
        mappings = self.session.get('field_mappings')
        feature_suggestions = self.session.get('feature_suggestions')
        feature_metadata = self.session.get('feature_metadata')
        approach = self.session.get('recommendation_approach', 'user_based')
        
        # Validate prerequisites
        if not all([df is not None, mappings, feature_suggestions, feature_metadata]):
            self.view.show_message("❌ Please complete previous steps first.", "error")
            return False
        
        # Check if already complete
        if self.session.get('similarity_complete', False):
            return True
        
        # Calculate similarity matrix
        with self.view.display_spinner('🔄 Calculating similarities...'):
            similarity_data = self._calculate_similarities(df, mappings, feature_suggestions, feature_metadata, approach)
            
        # Save results and display stats
        self.session.set('similarity_matrix', similarity_data)
        return self._display_similarity_stats(similarity_data, approach)
        
    def _calculate_similarities(self, df, mappings, feature_suggestions, feature_metadata, approach):
        """Calculate similarity matrix based on selected approach"""
        feature_weights = feature_suggestions.get('feature_weights', {})
        selected_features = list(feature_weights.keys())
        
        if approach == 'item_based':
            return self._calculate_item_similarities(df, selected_features, feature_weights, feature_metadata)
        else:
            return self._calculate_user_similarities(df, mappings, selected_features, feature_weights, feature_metadata)
        
    def _calculate_item_similarities(self, df, selected_features, feature_weights, feature_metadata):
        """Calculate item-item similarity matrix"""
        product_id_field = self.session.get('field_mappings')['product_id']
        
        unique_items = df[product_id_field].unique()
        item_indices = {item: idx for idx, item in enumerate(unique_items)}
        
        # Prepare feature matrix for each unique item
        item_feature_matrix = []
        for item in unique_items:
            item_data = df[df[product_id_field] == item]
            item_features = self._prepare_feature_matrix(item_data, selected_features, feature_weights, feature_metadata)
            # If we have multiple rows for an item, take the first one
            # This preserves the actual feature values instead of averaging
            item_feature_matrix.append(item_features[0])
        
        item_feature_matrix = np.array(item_feature_matrix)
        
        # Debug print
        print(f"Feature matrix shape: {item_feature_matrix.shape}")
        print(f"Sample of feature matrix:\n{item_feature_matrix[:5]}")
        
        # Calculate similarity matrix
        similarity_matrix = calculate_item_similarity(item_feature_matrix)
        
        # Add some checks
        if np.allclose(similarity_matrix, 1.0):
            print("Warning: All similarities are 1.0!")
            print("Unique feature values:", np.unique(item_feature_matrix))
        
        return {
            'matrix': similarity_matrix,
            'item_indices': item_indices,
            'items': unique_items,
            'approach': 'item_based'
        }
        
    def _calculate_user_similarities(self, df, mappings, selected_features, feature_weights, feature_metadata):
        """Calculate user-user similarity matrix"""
        # First prepare the complete feature matrix
        all_features = self._prepare_feature_matrix(df, selected_features, feature_weights, feature_metadata)
        
        # Then aggregate by user
        user_feature_matrix = []
        print(">>><<< mappings", mappings)
        unique_users = df[mappings['id']].unique()
        user_indices = {user: idx for idx, user in enumerate(unique_users)}
        
        for user in unique_users:
            user_mask = df[mappings['id']] == user
            if user_mask.any():
                # Take mean of all feature values for this user
                user_features = all_features[user_mask].mean(axis=0)
                user_feature_matrix.append(user_features)
            else:
                # If no data for user, use zeros
                user_feature_matrix.append(np.zeros(all_features.shape[1]))
        
        user_feature_matrix = np.array(user_feature_matrix)
        user_feature_matrix = np.nan_to_num(user_feature_matrix, nan=0.0)
        
        similarity_matrix = calculate_cosine_similarity(user_feature_matrix)
        
        return {
            'matrix': similarity_matrix,
            'user_indices': user_indices,
            'users': unique_users,
            'approach': 'user_based'
        }
        
    def _display_similarity_stats(self, similarity_data, approach):
        """Display similarity matrix statistics"""
        self.view.display_subheader("✅ Similarity Calculation Complete")
        
        matrix = similarity_data['matrix']
        entity_type = "Items" if approach == 'item_based' else "Users"
        
        # Calculate statistics
        total_entities = len(matrix)
        avg_similarity = matrix.mean()
        max_similarity = matrix.max()
        min_similarity = matrix[matrix > 0].min() if matrix.any() else 0
        
        # Create summary message that will be used both for display and storage
        summary_msg = f"**Similarity Calculation Summary:**\n\n"
        summary_msg += f"- Total {entity_type} Analyzed: **{total_entities}**\n"
        summary_msg += f"- Average Similarity: **{avg_similarity:.3f}**\n"
        summary_msg += f"- Maximum Similarity: **{max_similarity:.3f}**\n"
        summary_msg += f"- Minimum Non-zero Similarity: **{min_similarity:.3f}**\n"
        
        # Add distribution info to summary
        percentiles = np.percentile(matrix[matrix > 0], [25, 50, 75])
        summary_msg += f"\nSimilarity Distribution:\n"
        summary_msg += f"- 25th Percentile: **{percentiles[0]:.3f}**\n"
        summary_msg += f"- Median: **{percentiles[1]:.3f}**\n"
        summary_msg += f"- 75th Percentile: **{percentiles[2]:.3f}**\n"
        
        # Add insights to summary
        strong_similarities = (matrix > 0.5).sum() / 2
        total_possible = (total_entities * (total_entities - 1)) / 2
        
        summary_msg += f"\nQuick Insights:\n"
        summary_msg += (
            f"- Each {entity_type.lower()[:-1]} has "
            f"**{(matrix > 0.3).sum() / total_entities:.1f}** similar {entity_type.lower()} "
            f"(similarity > 0.3)\n"
        )
        summary_msg += (
            f"- **{(strong_similarities / total_possible * 100):.1f}%** of all possible "
            f"{entity_type.lower()} pairs have strong similarity (> 0.5)"
        )
        
        # Display current stats
        self.view.display_markdown(summary_msg)
        
        # Save summary for later display
        self.session.set('step_4_summary', summary_msg)
        
        # Show proceed button and wait for user confirmation
        if self.view.display_button("➡️ Proceed to Recommendations"):
            self.session.set('similarity_complete', True)
            self.session.set('step_4_complete', True)
            return True
        
        return False

    def _prepare_feature_matrix(self, df, selected_features, feature_weights, feature_metadata):
        """Prepare feature matrix with scaling and encoding"""
        processed_features = []
        feature_names = []
        
        for feature in selected_features:
            # Get feature values and handle NaN before processing
            feature_values = df[feature].fillna(df[feature].mean() if pd.api.types.is_numeric_dtype(df[feature]) 
                                              else df[feature].mode()[0])
            
            if feature_metadata[feature]['is_numeric']:
                # Scale numeric features
                scaler = StandardScaler()
                # Only scale if we have variation in the values
                if feature_values.nunique() > 1:
                    scaled_values = scaler.fit_transform(feature_values.values.reshape(-1, 1))
                else:
                    scaled_values = feature_values.values.reshape(-1, 1)
                processed_features.append(scaled_values * feature_weights[feature])
                feature_names.append(feature)
            else:
                # One-hot encode categorical features
                encoder = OneHotEncoder(sparse_output=False)
                encoded_values = encoder.fit_transform(feature_values.values.reshape(-1, 1))
                # Apply weights to encoded features
                weighted_encoded = encoded_values * feature_weights[feature]
                processed_features.append(weighted_encoded)
                feature_names.extend([f"{feature}_{val}" for val in encoder.categories_[0]])
        
        # Combine all features
        feature_matrix = np.hstack(processed_features)
        
        # Final NaN check and handling
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
        
        # Add tiny random noise to help differentiate identical items
        feature_matrix += np.random.uniform(low=0.0001, high=0.001, 
                                          size=feature_matrix.shape)
        
        # Debug print
        print(f"Processed features shape: {feature_matrix.shape}")
        print(f"Sample processed features:\n{feature_matrix[:5]}")
        
        return feature_matrix 