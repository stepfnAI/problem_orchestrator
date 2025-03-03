from sfn_blueprint import Task, SFNValidateAndRetryAgent
from orchestrator.agents.reco_explanation_agent import SFNRecommendationExplanationAgent
from orchestrator.config.model_config import DEFAULT_LLM_PROVIDER
from orchestrator.utils.similarity_utils import get_similar_users, filter_recommendations, get_similar_items
import pandas as pd
import numpy as np

class RecommendationGeneration:
    def __init__(self, session_manager, view):
        self.session = session_manager
        self.view = view
        # Removing explanation agent initialization
        # self.explanation_agent = SFNRecommendationExplanationAgent()
        
    def execute(self):
        """Execute the recommendation generation step"""
        # Check prerequisites
        similarity_data = self.session.get('similarity_matrix')
        df = self.session.get('df')
        mappings = self.session.get('field_mappings')
        
        if not all([similarity_data, df is not None, mappings]):
            self.view.show_message("❌ Please complete previous steps first.", "error")
            return False
            
        recommendations_complete = self.session.get('recommendations_complete', False)
        if recommendations_complete:
            return True
            
        return self._display_recommendation_interface()
        
    def _display_recommendation_interface(self):
        """Display staged recommendation interface"""
        # Initialize stage in session if not present
        if 'recommendation_stage' not in self.session.data:
            self.session.set('recommendation_stage', 'parameter_tuning')
            self.view.rerun_script()
            return False
        
        stage = self.session.get('recommendation_stage')
        
        # Always show parameters in tuning stage
        if stage == 'parameter_tuning':
            self.view.display_subheader("🎛️ Recommendation Parameters")
            params = self._display_parameter_controls()
            preview_stats = self._calculate_preview_stats(params)
            self._display_preview_stats(preview_stats)
            
            # Move to preview stage
            if self.view.display_button("🔍 Preview Recommendations"):
                preview_results = self._generate_preview_recommendations(params)
                # Set all states at once before rerun
                self.session.set('preview_results', preview_results)
                self.session.set('current_params', params)
                self.session.set('recommendation_stage', 'preview')
                self.view.rerun_script()
                return False
        
        # Show preview results and final generation button
        elif stage == 'preview':
            # Show saved preview results
            preview_results = self.session.get('preview_results')
            params = self.session.get('current_params')
            self._display_preview_results(preview_results)
            
            # Move to final stage
            if self.view.display_button("📊 Generate Final Recommendations"):
                with self.view.display_spinner('🎯 Generating recommendations...'):
                    recommendations = self._generate_recommendations(
                        df=self.session.get('df'),
                        mappings=self.session.get('field_mappings'),
                        similarity_data=self.session.get('similarity_matrix'),
                        params=params
                    )
                    # Set all states at once before rerun
                    self.session.set('recommendations', recommendations)
                    self.session.set('recommendation_stage', 'final')
                    self.view.rerun_script()
                    return False
                
            # Add button to go back to parameter tuning
            if self.view.display_button("⬅️ Back to Parameters"):
                self.session.set('recommendation_stage', 'parameter_tuning')
                self.view.rerun_script()
                return False
        
        # Show final results and completion button
        elif stage == 'final':
            recommendations = self.session.get('recommendations')
            self._display_final_results(recommendations)
            # The _display_final_results method now handles the finish button
            # which will reset all states and return to step 0
        
        return False
        
    def _display_parameter_controls(self):
        """Display parameter sliders and return current parameters"""
        approach = self.session.get('recommendation_approach', 'user_based')
        
        if approach == 'item_based':
            return self._display_item_based_controls()
        else:
            return self._display_user_based_controls()

    def _display_item_based_controls(self):
        """Display controls specific to item-based recommendations"""
        self.view.display_markdown("**Similar Items Settings**")
        n_similar_items = self.view.slider(
            "Number of similar items to consider",
            min_value=3,
            max_value=20,
            value=5,
            step=1,
            help="Higher values consider more items but may reduce recommendation precision"
        )
        
        min_similarity = self.view.slider(
            "Minimum similarity threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.3,
            step=0.1,
            help="Higher values ensure more similar items but may reduce coverage"
        )

        # Parameter presets for item-based
        preset_options = {
            "Balanced": {
                'n_similar_items': 5,
                'min_similarity': 0.3
            },
            "High Precision": {
                'n_similar_items': 3,
                'min_similarity': 0.6
            },
            "High Coverage": {
                'n_similar_items': 10,
                'min_similarity': 0.2
            }
        }
        
        selected_preset = self.view.select_box(
            "📋 Load Parameter Preset",
            ["Custom"] + list(preset_options.keys())
        )
        
        if selected_preset != "Custom":
            return preset_options[selected_preset]
            
        return {
            'n_similar_items': n_similar_items,
            'min_similarity': min_similarity,
            'n_similar_users': 5  # Add this for compatibility with preview stats
        }

    def _display_user_based_controls(self):
        """Display controls specific to user-based recommendations"""
        self.view.display_markdown("**Similar Users Settings**")
        n_similar_users = self.view.slider(
            "Number of similar users to consider",
            min_value=3,
            max_value=20,
            value=5,
            step=1,
            help="Higher values consider more users but may reduce recommendation precision"
        )
        
        min_similarity = self.view.slider(
            "Minimum similarity threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.3,
            step=0.1,
            help="Higher values ensure more similar users but may reduce coverage"
        )

        return {
            'n_similar_users': n_similar_users,
            'min_similarity': min_similarity
        }
    
    def _calculate_preview_stats(self, params):
        """Calculate quick preview statistics"""
        similarity_data = self.session.get('similarity_matrix')
        matrix = similarity_data['matrix']
        approach = similarity_data.get('approach', 'user_based')
        
        if approach == 'item_based':
            # Get items meeting minimum similarity threshold
            items_above_threshold = (matrix >= params['min_similarity'])
            
            # For each item, count similar items but cap at n_similar_items
            items_meeting_threshold = np.minimum(
                np.sum(items_above_threshold, axis=1),
                params['n_similar_items']
            )
            
            # Calculate statistics
            avg_similar_items = np.mean(items_meeting_threshold)
            items_with_enough = np.sum(items_meeting_threshold >= params['n_similar_items'])
            total_items = matrix.shape[0]
            
            return {
                'avg_similar_items': avg_similar_items,
                'items_with_enough': items_with_enough,
                'total_items': total_items,
                'coverage_percent': (items_with_enough / total_items) * 100
            }
        else:
            # Original user-based calculations
            users_above_threshold = (matrix >= params['min_similarity'])
            
            users_meeting_threshold = np.minimum(
                np.sum(users_above_threshold, axis=1),
                params['n_similar_users']
            )
            
            avg_similar_users = np.mean(users_meeting_threshold)
            users_with_enough = np.sum(users_meeting_threshold >= params['n_similar_users'])
            total_users = matrix.shape[0]
            
            return {
                'avg_similar_users': avg_similar_users,
                'users_with_enough': users_with_enough,
                'total_users': total_users,
                'coverage_percent': (users_with_enough / total_users) * 100
            }

    def _display_preview_stats(self, stats):
        """Display preview statistics"""
        self.view.display_subheader("📊 Parameter Impact Preview")
        
        # Create metrics display
        metric_cols = self.view.create_columns([1, 1])
        approach = self.session.get('similarity_matrix', {}).get('approach', 'user_based')
        
        with metric_cols[0]:
            if approach == 'item_based':
                self.view.metric(
                    "Average Similar Items per Item",
                    f"{stats['avg_similar_items']:.1f}"
                )
            else:
                self.view.metric(
                    "Average Similar Users per User",
                    f"{stats['avg_similar_users']:.1f}"
                )
        
        with metric_cols[1]:
            if approach == 'item_based':
                self.view.metric(
                    "Items with Sufficient Similar Items",
                    f"{stats['coverage_percent']:.1f}%"
                )
            else:
                self.view.metric(
                    "Users with Sufficient Similar Users",
                    f"{stats['coverage_percent']:.1f}%"
                )
        
        # Show warnings if needed
        if stats['coverage_percent'] < 50:
            entity_type = "items" if approach == 'item_based' else "users"
            self.view.show_message(
                f"⚠️ Current parameters might result in limited recommendations. "
                f"Consider adjusting the similarity threshold or number of similar {entity_type}.",
                "warning"
            )
    
    def _generate_preview_recommendations(self, params):
        """Generate preview of recommendations"""
        df = self.session.get('df')
        mappings = self.session.get('field_mappings')
        similarity_data = self.session.get('similarity_matrix')
        approach = similarity_data.get('approach', 'user_based')
        
        preview_results = []
        
        if approach == 'item_based':
            # Take a sample of items for preview
            sample_size = min(10, len(similarity_data['items']))
            sample_items = similarity_data['items'][:sample_size]
            
            for item in sample_items:
                item_idx = similarity_data['item_indices'][item]
                
                # Get all similar items first
                similar_items = get_similar_items(
                    similarity_data['matrix'],
                    item_idx
                )
                
                # Apply similarity threshold
                similar_items = [
                    s for s in similar_items 
                    if s['similarity'] >= params['min_similarity']
                ]
                
                # Filter by minimum interactions if specified
                if params.get('min_interactions', 0) > 0 and 'customer_id' in mappings and mappings['customer_id']:
                    similar_items = [
                        s for s in similar_items
                        if df[df[mappings['product_id']] == similarity_data['items'][s['index']]][mappings['customer_id']].nunique() >= params['min_interactions']
                    ]
                
                # Sort by similarity and take top N
                similar_items.sort(key=lambda x: x['similarity'], reverse=True)
                similar_items = similar_items[:params['n_similar_items']]
                
                preview_results.append({
                    'item': item,
                    'similar_count': len(similar_items),
                    'notes': "✅ Sufficient similar items found" if similar_items else "❌ No similar items found"
                })
        else:
            # Handle user-based preview
            # Sample users for preview
            sample_size = min(10, len(similarity_data['users']))
            sample_users = similarity_data['users'][:sample_size]
            
            # Generate preview for users
            for user in sample_users:
                user_idx = similarity_data['user_indices'][user]
                similar_users = self._get_filtered_similar_users(similarity_data, user_idx, params)
                
                preview_results.append({
                    'item': user,  # Using 'item' as the column name for consistency
                    'similar_count': len(similar_users),
                    'notes': ("✅ Sufficient similar users found" 
                            if len(similar_users) >= params['n_similar_users']
                            else f"⚠️ Only found {len(similar_users)} similar users")
                })
        
        return preview_results
    
    def _display_preview_results(self, preview_results):
        """Display preview recommendation results"""
        self.view.display_subheader("🔍 Preview Results")
        approach = self.session.get('recommendation_approach', 'user_based')
        entity_type = "Item" if approach == 'item_based' else "User"
        
        # Convert list of results to DataFrame directly
        preview_df = pd.DataFrame(preview_results)
        
        # Rename columns for display
        preview_df.columns = [
            entity_type,
            f"Similar {entity_type}s Found",
            "Notes"
        ]
        
        # Add index column
        preview_df.index = range(len(preview_df))
        
        self.view.display_dataframe(preview_df)
        
        # Show summary statistics
        total_with_sufficient = sum(1 for r in preview_results if r['similar_count'] >= 4)
        self.view.display_markdown(f"\n**Summary:**")
        self.view.display_markdown(f"- {total_with_sufficient} out of {len(preview_results)} "
                                 f"{entity_type.lower()}s have sufficient similar items")
    
    def _display_final_results(self, recommendations):
        """Display final recommendation results"""
        self.view.display_subheader("📊 Final Recommendations")
        
        # Display finalized parameters
        approach = self.session.get('similarity_matrix', {}).get('approach', 'user_based')
        params = self.session.get('current_params', {})
        
        self.view.display_markdown("**🎛️ Finalized Parameters**")
        param_cols = self.view.create_columns([1, 1])
        with param_cols[0]:
            if approach == 'item_based':
                self.view.metric(
                    "Number of Similar Items",
                    str(params.get('n_similar_items', 5))
                )
            else:
                self.view.metric(
                    "Number of Similar Users",
                    str(params.get('n_similar_users', 5))
                )
        with param_cols[1]:
            self.view.metric(
                "Similarity Threshold",
                f"{params.get('min_similarity', 0.3):.2f}"
            )
        
        self.view.display_markdown("---")  # Add a separator
        
        # Calculate overall metrics
        metrics = self._calculate_recommendation_metrics(recommendations)
        
        # Display metrics
        metric_cols = self.view.create_columns([1, 1, 1])
        with metric_cols[0]:
            self.view.metric(
                "Coverage",
                f"{metrics['coverage']:.1f}%"
            )
        with metric_cols[1]:
            self.view.metric(
                "Average Confidence",
                f"{metrics['avg_confidence']:.2f}"
            )
        with metric_cols[2]:
            self.view.metric(
                "Unique Products",
                str(metrics['unique_products'])
            )
        
        # Create results table
        table_data = []
        for user, user_data in recommendations.items():
            recommended_products = user_data.get('recommended_products', [])
            if not recommended_products:  # Skip users with no recommendations
                continue
            
            recommendations_text = ', '.join([
                f"{r['product_id']} ({r['confidence_score']:.2f})"
                for r in recommended_products
            ])
            
            table_data.append({
                'ID': user,
                'Number of Recommendations': len(recommended_products),
                'Recommended Products (Confidence)': recommendations_text,
                'Average Confidence': f"{sum(r['confidence_score'] for r in recommended_products) / len(recommended_products):.2f}"
            })
        
        if table_data:  # Only show table if there are results
            results_df = pd.DataFrame(table_data)
            self.view.display_dataframe(results_df)
            
            # Save results and create download button
            self.session.set('recommendation_results', results_df)
            self.view.create_download_button(
                label="📥 Download Recommendations as CSV",
                data=results_df.to_csv(index=False),
                file_name="recommendation_results.csv",
                mime_type='text/csv'
            )
            
            # Add finish button
            if self.view.display_button("✅ Finish"):
                # Reset all states
                self.session.clear()
                self.view.rerun_script()
                return True
        else:
            self.view.show_message("⚠️ No recommendations generated. Try adjusting the parameters.", "warning")

    def _generate_recommendations(self, df, mappings, similarity_data, params):
        """Generate recommendations based on selected approach"""
        approach = similarity_data.get('approach', 'user_based')
        
        if approach == 'item_based':
            return self._generate_item_based_recommendations(df, mappings, similarity_data, params)
        else:
            return self._generate_user_based_recommendations(df, mappings, similarity_data, params)

    def _generate_item_based_recommendations(self, df, mappings, similarity_data, params):
        """Generate item-based recommendations"""
        recommendations = {}
        product_id_field = mappings['product_id']
        
        for item in similarity_data['items']:
            item_idx = similarity_data['item_indices'][item]
            
            # Get all similar items first
            similar_items = get_similar_items(
                similarity_data['matrix'],
                item_idx
            )
            
            # Apply similarity threshold
            similar_items = [
                s for s in similar_items 
                if s['similarity'] >= params['min_similarity']
            ]
            
            # Sort by similarity score
            similar_items.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Only add recommendations if we found similar items
            if similar_items:
                recommendations[item] = {
                    'similar_items': similar_items[:params['n_similar_items']],
                    'recommended_products': [
                        {
                            'product_id': similarity_data['items'][s['index']],
                            'confidence_score': min(s['similarity'], 1.0)
                        }
                        for s in similar_items[:params['n_similar_items']]
                    ]
                }
        
        return recommendations

    def _generate_user_based_recommendations(self, df, mappings, similarity_data, params):
        """Generate user-based recommendations"""
        recommendations = {}
        
        for user in similarity_data['users']:
            user_idx = similarity_data['user_indices'][user]
            
            # Use tuned parameters for similar users
            similar_users = self._get_filtered_similar_users(similarity_data, user_idx, params)
            
            # Get products from similar users with confidence scores
            product_confidence = {}
            for similar in similar_users:
                similar_user = similarity_data['users'][similar['index']]
                similar_user_df = df[df[mappings['id']] == similar_user]
                print(">>><<< mappings", mappings)
                products = similar_user_df[mappings['product_id']].unique()
                
                for product in products:
                    if product not in product_confidence:
                        product_confidence[product] = {
                            'total_similarity': 0,
                            'user_count': 0
                        }
                    
                    product_confidence[product]['total_similarity'] += similar['similarity']
                    product_confidence[product]['user_count'] += 1
            
            # Calculate final confidence scores and prepare recommendations
            user_products = set(df[df[mappings['id']] == user][mappings['product_id']].unique())
            recommended_products = []
            
            for product, conf_data in product_confidence.items():
                if product not in user_products:
                    confidence = conf_data['total_similarity'] / conf_data['user_count']
                    recommended_products.append({
                        'product_id': product,
                        'confidence_score': min(confidence, 1.0)
                    })
            
            # Sort by confidence and take top 5
            recommended_products.sort(key=lambda x: x['confidence_score'], reverse=True)
            recommendations[user] = {
                'similar_users': similar_users,
                'recommended_products': recommended_products[:5]
            }
        
        return recommendations
        
    def _calculate_recommendation_metrics(self, recommendations):
        """Calculate metrics for the generated recommendations"""
        approach = self.session.get('recommendation_approach', 'user_based')
        entity_type = "items" if approach == 'item_based' else "users"
        
        total_entities = len(recommendations)
        entities_with_recs = sum(1 for data in recommendations.values() 
                               if data['recommended_products'])
        
        all_products = set()
        total_confidence = 0
        total_recs = 0
        
        for data in recommendations.values():
            for rec in data['recommended_products']:
                all_products.add(rec['product_id'])
                total_confidence += rec['confidence_score']
                total_recs += 1
        
        return {
            'coverage': (entities_with_recs / total_entities * 100) if total_entities > 0 else 0,
            'avg_confidence': total_confidence / total_recs if total_recs > 0 else 0,
            'unique_products': len(all_products)
        }

    def _get_filtered_similar_users(self, similarity_data, user_idx, params):
        """Get filtered similar users based on tuning parameters"""
        similar_users = get_similar_users(
            similarity_data['matrix'],
            user_idx
        )
        
        # Filter by minimum similarity
        similar_users = [
            user for user in similar_users 
            if user['similarity'] >= params['min_similarity']
        ]
        
        # Sort by similarity and get top N
        similar_users.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_users[:params['n_similar_users']]