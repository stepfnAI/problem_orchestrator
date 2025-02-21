from sfn_blueprint import Task, SFNValidateAndRetryAgent
from recommendation_agent.agents import SFNDataMappingAgent, SFNApproachSelectionAgent
from recommendation_agent.config.model_config import DEFAULT_LLM_PROVIDER

class DataMapping:
    # Add field display names as a class constant
    FIELD_DISPLAY_NAMES = {
        "customer_id": "ID",
        "product_id": "Product ID",
        "timestamp": "Timestamp",
        "interaction_value": "Interaction Value"
    }

    # Add approach display names
    APPROACH_DISPLAY_NAMES = {
        "user_based": "User Collaborative Filtering",
        "item_based": "Item-Based Similarity"
    }

    def __init__(self, session_manager, view):
        self.session = session_manager
        self.view = view
        self.mapping_agent = SFNDataMappingAgent()
        print("Initializing approach_agent")
        self.approach_agent = SFNApproachSelectionAgent(llm_provider=DEFAULT_LLM_PROVIDER)
        print("Initialized approach_agent")
        
    def execute(self):
        """Main execution flow for data mapping and approach selection"""
        if not self._validate_data():
            return False
            
        # Get current state
        is_complete = self.session.get('step_2_complete', False)
        if is_complete:
            return True
            
        # Handle the mapping and approach selection flow
        mapping_confirmed = self.session.get('mapping_confirmed', False)
        approach_confirmed = self.session.get('approach_confirmed', False)
        
        if not mapping_confirmed:
            return self._handle_mapping_flow()
        elif not approach_confirmed:
            return self._handle_approach_selection()
            
        return True
        
    def _validate_data(self):
        """Check if data is available for mapping"""
        df = self.session.get('df')
        if df is None:
            self.view.show_message("❌ No data found. Please upload data first.", "error")
            return False
        return True
        
    def _handle_mapping_flow(self):
        """Main mapping flow controller"""
        if not self.session.get('suggested_mappings'):
            if not self._get_ai_suggestions():
                return False
        
        return self._display_mapping_interface()
        
    def _get_ai_suggestions(self):
        """Get AI suggestions for column mapping"""
        try:
            df = self.session.get('df')
            with self.view.display_spinner('🤖 AI is mapping critical fields...'):
                mapping_task = Task("Map columns", data=df)
                validation_task = Task("Validate field mapping", data=df)
                
                validate_and_retry_agent = SFNValidateAndRetryAgent(
                    llm_provider=DEFAULT_LLM_PROVIDER,
                    for_agent='data_mapper'
                )
                
                mappings, validation_message, is_valid = validate_and_retry_agent.complete(
                    agent_to_validate=self.mapping_agent,
                    task=mapping_task,
                    validation_task=validation_task,
                    method_name='execute_task',
                    get_validation_params='get_validation_params',
                    max_retries=2,
                    retry_delay=3.0
                )
                if is_valid:
                    self.session.set('suggested_mappings', mappings)
                    return True
                else:
                    self.view.show_message("❌ AI couldn't generate valid field mappings.", "error")
                    self.view.show_message(validation_message, "warning")
                    return False
                    
        except Exception as e:
            self.view.show_message(f"Error in AI mapping: {str(e)}", "error")
            return False
            
    def _display_mapping_interface(self):
        """Display and handle mapping interface"""
        df = self.session.get('df')
        suggested_mappings = self.session.get('suggested_mappings', {})
        
        # Show AI suggestions if available
        if suggested_mappings:
            self._display_ai_suggestions(suggested_mappings)
        
        # Get user's mapping choice
        action = self.view.radio_select(
            "How would you like to proceed?",
            options=["Use AI Recommended Mappings", "Select Columns Manually"],
            key="mapping_choice"
        )
        
        # Handle mapping based on user choice
        if action == "Use AI Recommended Mappings":
            return self._handle_ai_mapping(suggested_mappings)
        else:
            return self._handle_manual_mapping(df, suggested_mappings)
            
    def _display_ai_suggestions(self, suggested_mappings):
        """Display AI suggested mappings"""
        self.view.display_subheader("AI Suggested Field Mappings")
        
        message = "🎯 AI Suggested Mappings:\n"
        for field, mapped_col in suggested_mappings.items():
            display_name = self.FIELD_DISPLAY_NAMES.get(field, field)
            message += f"- {display_name}:  **{mapped_col or 'Not Found'}**\n"
        
        self.view.show_message(message, "info")
        self.view.display_markdown("---")
        
    def _handle_ai_mapping(self, suggested_mappings):
        """Handle AI mapping confirmation"""
        self.session.set('field_mappings', suggested_mappings)
        if self.view.display_button("✅ Confirm AI Mappings"):
            self.session.set('mapping_confirmed', True)
            # Don't set step_2_complete here, move to approach selection
            self.view.rerun_script()
            return False
        return False
        
    def _handle_manual_mapping(self, df, suggested_mappings):
        """Handle manual mapping selection"""
        mappings = self._get_manual_mappings(df, suggested_mappings)
        
        if not self._validate_manual_mappings(mappings):
            return False
            
        if self.view.display_button("✅ Confirm Manual Mappings"):
            self.session.set('field_mappings', mappings)
            self.session.set('mapping_confirmed', True)
            # Don't set step_2_complete here, move to approach selection
            self.view.rerun_script()
            return False
        return False
        
    def _get_manual_mappings(self, df, suggested_mappings):
        """Get manual mapping selections from user"""
        mappings = {}
        
        self.view.display_subheader("Required Fields")
        mappings['product_id'] = self.view.select_box(
            f"{self.FIELD_DISPLAY_NAMES['product_id']} Column",
            options=[""] + df.columns.tolist(),
            default=suggested_mappings.get('product_id', ""),
            key="product_id_select"
        )
        
        self.view.display_subheader("Optional Fields")
        for field in ["customer_id", "timestamp", "interaction_value"]:
            value = self.view.select_box(
                f"{self.FIELD_DISPLAY_NAMES[field]} Column (optional)",
                options=[""] + df.columns.tolist(),
                default=suggested_mappings.get(field, ""),
                key=f"{field}_select"
            )
            if value:
                mappings[field] = value
                
        return mappings
        
    def _validate_manual_mappings(self, mappings):
        """Validate manual mapping selections"""
        if not mappings.get('product_id'):
            self.view.show_message("❌ Product ID is a required field", "error")
            return False
        return True
        
    def _handle_approach_selection(self):
        """Handle the approach selection flow"""
        mappings = self.session.get('field_mappings', {})
        print(f">>><<<Mappings: {mappings}")
        
        # Check for all possible empty/null values
        customer_id = mappings.get('customer_id')
        has_user_data = (
            customer_id is not None and 
            customer_id != '' and 
            customer_id != 'null' and 
            customer_id != 'NULL' and 
            customer_id != 'nan' and 
            customer_id != 'NaN'
        )

        # Display current mapping summary
        self._display_mapping_summary(mappings)

        if not has_user_data:
            return self._handle_item_based_only_approach()

        # First get AI suggestion if not already present
        if not self.session.get('suggested_approach'):
            with self.view.display_spinner('🤖 AI is analyzing your dataset...'):
                if not self._get_ai_approach_suggestion():
                    return False

        suggested_approach = self.session.get('suggested_approach')

        # Show choice interface
        self.view.display_markdown("### 🔄 Choose How to Proceed")
        action = self.view.radio_select(
            "How would you like to proceed with recommendation approach?",
            options=["Use AI Recommended Approach", "Select Approach Manually"],
            key="approach_choice"
        )

        if action == "Use AI Recommended Approach":
            return self._handle_ai_approach()
        else:
            return self._handle_manual_approach(has_user_data)

    def _handle_item_based_only_approach(self):
        """Handle approach selection when only item-based similarity is possible"""
        self.view.show_message(
            "ℹ️ Since no customer data is mapped, we'll use Item-Based Similarity as the recommendation approach.", 
            "info"
        )
        
        if self.view.display_button("⏩ Proceed to Feature Selection"):
            self.session.set('recommendation_approach', 'item_based')
            self.session.set('approach_confirmed', True)
            self.session.set('step_2_complete', True)
            self._save_step_summary()
            return True
        return False

    def _get_ai_approach_suggestion(self):
        """Get AI suggestion for recommendation approach"""
        print("Starting approach suggestion process")
        df = self.session.get('df')
        print(f"Retrieved DataFrame from session: {df is not None}")
        
        approach_task = Task("Select approach", data={'df': df})
        validation_task = Task("Validate approach", data={'df': df})
        
        validate_and_retry_agent = SFNValidateAndRetryAgent(
            llm_provider=DEFAULT_LLM_PROVIDER,
            for_agent='approach_selector'
        )
        print("Created validate_and_retry_agent")
        
        try:
            suggestion, validation_message, is_valid = validate_and_retry_agent.complete(
                agent_to_validate=self.approach_agent,
                task=approach_task,
                validation_task=validation_task,
                method_name='execute_task',
                get_validation_params='get_validation_params',
                max_retries=2,
                retry_delay=3.0
            )
            print("Completed validate_and_retry_agent.complete()")
            
            if is_valid:
                # Store the suggestion in session
                self.session.set('suggested_approach', suggestion)
                # Display the suggestion
                self._display_ai_approach_suggestion(suggestion)
                return True
            else:
                self.view.show_message("❌ AI couldn't determine the best approach.", "error")
                self.view.show_message(validation_message, "warning")
                return False
                
        except Exception as e:
            print(f"Error in validate_and_retry_agent.complete(): {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            self.view.show_message("❌ Error getting approach suggestion", "error")
            return False

    def _display_ai_approach_suggestion(self, suggestion):
        """Display AI suggested approach"""
        approach = suggestion.get('suggested_approach')
        explanation = suggestion.get('explanation')
        confidence = suggestion.get('confidence', 0)
        
        self.view.display_subheader("🤖 AI Suggested Approach")
        self.view.display_markdown(
            f"**Recommended Approach**: {self.APPROACH_DISPLAY_NAMES.get(approach)}\n\n"
            f"**Confidence**: {confidence:.0%}\n\n"
            f"**Reasoning**:\n{explanation}"
        )
        self.view.display_markdown("---")

    def _handle_ai_approach(self):
        """Handle AI approach confirmation"""
        suggested_approach = self.session.get('suggested_approach')
        if self.view.display_button("✅ Confirm AI Suggested Approach"):
            self.session.set('recommendation_approach', suggested_approach.get('suggested_approach'))
            self.session.set('approach_confirmed', True)
            # Now we can mark step 2 as complete
            self.session.set('step_2_complete', True)
            self._save_step_summary()
            return True
        return False

    def _handle_manual_approach(self, has_user_data):
        """Handle manual approach selection"""
        self.view.display_subheader("Select Recommendation Approach")
        
        options = ["Item-Based Similarity"]
        if has_user_data:
            options.append("User Collaborative Filtering")
            
        selected_approach = self.view.radio_select(
            "Choose an approach that best suits your needs:",
            options=options
        )
        
        approach_mapping = {
            "Item-Based Similarity": "item_based",
            "User Collaborative Filtering": "user_based"
        }
        
        if self.view.display_button("✅ Confirm Selected Approach"):
            self.session.set('recommendation_approach', approach_mapping[selected_approach])
            self.session.set('approach_confirmed', True)
            # Now we can mark step 2 as complete
            self.session.set('step_2_complete', True)
            self._save_step_summary()
            return True
        return False

    def _display_mapping_summary(self, mappings):
        """Display current mapping summary"""
        self.view.display_subheader("Current Field Mappings")
        
        message = "📍 Mapped Fields:\n"
        for field, col in mappings.items():
            if col:  # Only show mapped fields
                display_name = self.FIELD_DISPLAY_NAMES.get(field, field)
                message += f"- {display_name}: **{col}**\n"
        
        self.view.show_message(message, "info")
        self.view.display_markdown("---")

    def _save_step_summary(self):
        """Save step summary including both mapping and approach"""
        mappings = self.session.get('field_mappings')
        approach = self.session.get('recommendation_approach')
        
        summary = "✅ Step 2 Complete\n\n"
        summary += "Field Mappings:\n"
        for field, col in mappings.items():
            if col:  # Only show mapped fields
                display_name = self.FIELD_DISPLAY_NAMES.get(field, field)
                summary += f"- {display_name}: **{col}**\n"
        
        summary += f"\nSelected Approach: **{self.APPROACH_DISPLAY_NAMES.get(approach)}**"
        
        self.session.set('step_2_summary', summary) 