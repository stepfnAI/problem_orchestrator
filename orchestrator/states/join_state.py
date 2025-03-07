from orchestrator.states.base_state import BaseState
from orchestrator.storage.db_connector import DatabaseConnector
from orchestrator.agents.join_suggestion_agent import SFNJoinSuggestionAgent
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import json
import sqlite3
from sfn_blueprint import Task
import logging
import uuid

logger = logging.getLogger(__name__)

class JoinState(BaseState):
    """State handling the joining of tables based on AI suggestions and user input"""
    
    def __init__(self, session_manager, view):
        self.session = session_manager
        self.view = view
        self.db = DatabaseConnector()
        self.join_agent = SFNJoinSuggestionAgent()
        
    def execute(self) -> bool:
        """Execute the join state logic"""
        # Check if state is already complete
        if self.session.get('join_complete'):
            self._show_state_summary()  # Show summary if already complete
            return True
        
        # Step 1: Fetch aggregation summary if not done
        if not self.session.get('aggregation_summary'):
            if not self._fetch_aggregation_summary():
                return False
                
        # Step 2: Fetch mapping summary if not done
        if not self.session.get('mapping_summary'):
            if not self._fetch_mapping_summary():
                return False
                
        # Check if we only have one table - special case handling
        tables_data = self.session.get('tables_data', {})
        if len(tables_data) == 1:
            return self._handle_single_table_case()
            
        # Step 3: Initialize join process if not done
        if not self.session.get('join_initialized'):
            if not self._initialize_join_process():
                return False
                
        # Step 4: Process joins until all tables are joined
        available_tables = self.session.get('available_tables', [])
        
        if len(available_tables) >= 2:
            # Still have tables to join
            return self._process_join()
        else:
            # All tables joined, show final summary
            return self._show_final_summary()
            
    def _fetch_aggregation_summary(self) -> bool:
        """Fetch aggregation summary from database"""
        try:
            session_id = self.session.get('session_id')
            print(f">>> Fetching aggregation summary for session_id: {session_id}")
            summary = self.db.fetch_state_summary('aggregation', session_id)
            
            if not summary:
                print(">>> ❌ Aggregation summary not found")
                self.view.show_message("❌ Aggregation summary not found", "error")
                return False
                
            print(f">>> Aggregation summary found: {summary}")
            # Store the summary in the session
            self.session.set('aggregation_summary', summary)
            
            # Fetch aggregated tables data
            print(f">>> Fetching aggregated tables data for session_id: {session_id}")
            tables_data = self._fetch_aggregated_tables(session_id)
            if not tables_data:
                print(">>> ❌ Aggregated tables data not found")
                self.view.show_message("❌ Aggregated tables data not found", "error")
                return False
                
            print(f">>> Found {len(tables_data)} aggregated tables")
            self.session.set('tables_data', tables_data)
            return True
            
        except Exception as e:
            print(f">>> Error fetching aggregation summary: {str(e)}")
            self.view.show_message(f"Error fetching aggregation summary: {str(e)}", "error")
            return False
            
    def _fetch_aggregated_tables(self, session_id: str) -> Dict[str, pd.DataFrame]:
        """Fetch all aggregated tables data"""
        print(f">>> Inside _fetch_aggregated_tables for session_id: {session_id}")
        
        try:
            # Connect to database
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                
                # First, get the aggregation summary to find the actual table names
                print(">>> Fetching aggregation summary to get actual table names")
                cursor.execute(
                    "SELECT table_name, aggregated_table_name FROM aggregation_summary WHERE session_id = ?",
                    (session_id,)
                )
                table_mappings = cursor.fetchall()
                
                print(f">>> Found table mappings: {table_mappings}")
                if not table_mappings:
                    print(">>> No table mappings found in aggregation_summary")
                    
                    # Try to get all tables from aggregation_summary
                    cursor.execute(
                        "SELECT * FROM aggregation_summary WHERE session_id = ?",
                        (session_id,)
                    )
                    all_summary = cursor.fetchall()
                    print(f">>> All aggregation summary data: {all_summary}")
                    
                    # If we have summary data but no mappings, try to extract from the first row
                    if all_summary:
                        # Get column names
                        column_names = [description[0] for description in cursor.description]
                        print(f">>> Column names in aggregation_summary: {column_names}")
                        
                        # Try to find the right columns
                        table_mappings = []
                        for row in all_summary:
                            row_dict = {column_names[i]: row[i] for i in range(len(column_names))}
                            
                            original_table = row_dict.get('table_name')
                            aggregated_table = row_dict.get('aggregated_table_name')
                            
                            if original_table and aggregated_table:
                                table_mappings.append((original_table, aggregated_table))
                                print(f">>> Extracted mapping: {original_table} -> {aggregated_table}")
                    
                    if not table_mappings:
                        return {}
                    
                # Fetch data for each aggregated table
                tables_data = {}
                for original_name, aggregated_name in table_mappings:
                    print(f">>> Fetching data for aggregated table: {aggregated_name} (original: {original_name})")
                    
                    # Check if the table exists
                    cursor.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                        (aggregated_name,)
                    )
                    if not cursor.fetchone():
                        print(f">>> Table {aggregated_name} does not exist in database")
                        
                        # If aggregated table doesn't exist, try to use the original table
                        print(f">>> Falling back to original table: {original_name}")
                        try:
                            # Try to get the original table data
                            cursor.execute(
                                "SELECT data FROM uploaded_tables WHERE session_id = ? AND table_name = ?",
                                (session_id, original_name)
                            )
                            result = cursor.fetchone()
                            
                            if result and result[0]:
                                print(f">>> Data found for original table: {original_name}")
                                # Deserialize the data
                                df = pd.read_json(result[0])
                                tables_data[original_name] = df
                                print(f">>> Original table {original_name} shape: {df.shape}")
                            else:
                                print(f">>> No data found for original table: {original_name}")
                        except Exception as e:
                            print(f">>> Error fetching original table {original_name}: {str(e)}")
                        
                        continue
                    
                    # Fetch the data from the actual table
                    try:
                        # Read the entire table
                        df = pd.read_sql_query(f"SELECT * FROM {aggregated_name}", conn)
                        if not df.empty:
                            print(f">>> Data found for table: {aggregated_name}, shape: {df.shape}")
                            # Store with original table name for consistency
                            tables_data[original_name] = df
                        else:
                            print(f">>> No data found in table: {aggregated_name}")
                    except Exception as e:
                        print(f">>> Error reading table {aggregated_name}: {str(e)}")
                
                print(f">>> Returning {len(tables_data)} tables")
                return tables_data
                
        except Exception as e:
            print(f">>> Error fetching aggregated tables: {str(e)}")
            logger.error(f"Error fetching aggregated tables: {str(e)}")
            return {}
            
    def _fetch_mapping_summary(self) -> bool:
        """Fetch mapping summary from database"""
        try:
            session_id = self.session.get('session_id')
            print(f">>> Fetching mapping summary for session_id: {session_id}")
            summary = self.db.fetch_state_summary('mapping', session_id)
            
            if not summary:
                print(">>> ❌ Mapping summary not found")
                self.view.show_message("❌ Mapping summary not found", "error")
                return False
                
            print(f">>> Mapping summary found: {summary}")
            # Store the summary in the session
            self.session.set('mapping_summary', summary)
            
            # Also fetch detailed mappings
            print(">>> Fetching detailed table mappings")
            mappings = self._fetch_table_mappings(session_id)
            if mappings:
                print(f">>> Found mappings for {len(mappings)} tables")
                self.session.set('table_mappings', mappings)
            else:
                print(">>> No table mappings found")
                
            return True
            
        except Exception as e:
            print(f">>> Error fetching mapping summary: {str(e)}")
            self.view.show_message(f"Error fetching mapping summary: {str(e)}", "error")
            return False
            
    def _fetch_table_mappings(self, session_id: str) -> Dict[str, Dict[str, str]]:
        """Fetch detailed mappings for all tables"""
        try:
            print(f">>> Inside _fetch_table_mappings for session_id: {session_id}")
            # Connect to database
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                
                # Get mappings
                print(">>> Executing SQL to get table mappings")
                cursor.execute(
                    "SELECT table_name, mappings FROM table_mappings WHERE session_id = ?",
                    (session_id,)
                )
                results = cursor.fetchall()
                
                print(f">>> Found {len(results)} mapping results")
                if not results:
                    return {}
                    
                # Parse mappings
                mappings = {}
                for table_name, mapping_json in results:
                    if mapping_json:
                        print(f">>> Parsing mapping for table: {table_name}")
                        mappings[table_name] = json.loads(mapping_json)
                
                print(f">>> Returning mappings for {len(mappings)} tables")
                return mappings
                
        except Exception as e:
            print(f">>> Error fetching table mappings: {str(e)}")
            logger.error(f"Error fetching table mappings: {str(e)}")
            return {}
            
    def _initialize_join_process(self) -> bool:
        """Initialize the join process"""
        try:
            # Get tables data
            tables_data = self.session.get('tables_data', {})
            print(f">>> Initializing join process with {len(tables_data)} tables")
            
            if not tables_data:
                print(">>> ❌ No tables data found")
                self.view.show_message("❌ No tables data found", "error")
                return False
                
            # Initialize available tables
            available_tables = list(tables_data.keys())
            print(f">>> Available tables: {available_tables}")
            self.session.set('available_tables', available_tables)
            
            # Initialize joined tables (empty at start)
            self.session.set('joined_tables', {})
            
            # Initialize join history
            self.session.set('join_history', [])
            
            # Mark as initialized
            self.session.set('join_initialized', True)
            print(">>> Join process initialized successfully")
            
            return True
            
        except Exception as e:
            print(f">>> Error initializing join process: {str(e)}")
            self.view.show_message(f"Error initializing join process: {str(e)}", "error")
            return False
            
    def _process_join(self) -> bool:
        """Process the next join"""
        try:
            # Get available tables
            available_tables = self.session.get('available_tables', [])
            tables_data = self.session.get('tables_data', {})
            joined_tables = self.session.get('joined_tables', {})
            join_history = self.session.get('join_history', [])
            
            print(f">>> Processing join with {len(available_tables)} available tables")
            print(f">>> Available tables: {available_tables}")
            
            # Combine original and joined tables data
            all_tables_data = {**tables_data, **joined_tables}
            
            # Check if we're in the input collection phase
            if not self.session.get('join_input_complete') and not self.session.get('current_join_suggestion'):
                # Get user input for important table and custom instructions
                important_table = None
                custom_instructions = None
                
                # Display header for user input
                self.view.display_subheader("Join Configuration")
                
                # Only ask for important table if we haven't already
                if not self.session.get('important_table') and len(available_tables) > 1:
                    # Ask user to select their most important table
                    self.view.display_markdown(
                        "Select your most important table that should be prioritized in joins (optional):"
                    )
                    
                    important_table = self.view.select_box(
                        "Most important table:",
                        options=["None"] + available_tables,
                        default="None",
                        key="important_table_selection"
                    )
                    
                    if important_table != "None":
                        self.session.set('important_table', important_table)
                        # Also track the original important table name for lineage tracking
                        self.session.set('original_important_table', important_table)
                        self.view.display_markdown(f"✅ Selected **{important_table}** as the most important table")
                    else:
                        important_table = None
                else:
                    # Get previously selected important table if any
                    important_table = self.session.get('important_table')
                
                # Get custom join instructions
                self.view.display_markdown(
                    "Provide any specific instructions for how tables should be joined (optional):"
                )
                
                custom_instructions = self.view.text_area(
                    "Custom join instructions:",
                    value=self.session.get('custom_join_instructions', ''),
                    key="custom_join_instructions"
                )
                
                if custom_instructions:
                    self.session.set('custom_join_instructions', custom_instructions)
                
                # Add a button to proceed to AI suggestion
                if self.view.display_button("Proceed to AI Join Suggestion", key="proceed_to_ai_suggestion"):
                    self.session.set('join_input_complete', True)
                    self.view.rerun_script()
                
                return False
            
            # If input is complete or we already have a suggestion, proceed with AI suggestion
            if not self.session.get('current_join_suggestion'):
                # Get important table and custom instructions from session
                important_table = self.session.get('important_table')
                original_important_table = self.session.get('original_important_table')
                custom_instructions = self.session.get('custom_join_instructions')
                
                # Track the important table through joins
                if important_table and important_table not in available_tables:
                    # The important table might have been joined already
                    # Look through join history to find where it went
                    for join_record in join_history:
                        if join_record.get('left_table') == important_table or join_record.get('right_table') == important_table:
                            # Found where our important table went - it's now part of this result table
                            important_table = join_record.get('result_table')
                            self.session.set('important_table', important_table)
                            print(f">>> Updated important table to: {important_table}")
                            break
                
                # Prepare other_info for the join agent
                other_info = ""
                if important_table:
                    other_info += f"Important table that should be prioritized in joins: {important_table}\n\n"
                    
                    # Add lineage information if this is a joined table
                    if important_table != original_important_table:
                        other_info += f"Note: The important table '{important_table}' contains the original important table '{original_important_table}' from a previous join.\n\n"
                
                if custom_instructions:
                    other_info += f"Custom join instructions: {custom_instructions}\n\n"
                
                # Add table mappings information
                table_mappings = self.session.get('table_mappings', {})
                other_info += f"Table mappings: {json.dumps(table_mappings)}"
                
                # Add join history information to help the LLM understand table lineage
                if join_history:
                    other_info += f"\n\nJoin History: {json.dumps(join_history)}\n"
                
                # Get AI suggestion for the next join
                print(">>> Getting AI join suggestion")
                with self.view.display_spinner('🤖 AI is analyzing tables for join suggestions...'):
                    join_suggestion = self._get_join_suggestion(available_tables, all_tables_data, other_info)
                    
                if not join_suggestion:
                    print(">>> ❌ Failed to get join suggestion")
                    self.view.show_message("❌ Failed to get join suggestion", "error")
                    return False
                
                print(f">>> AI suggested join: {join_suggestion}")
                self.session.set('current_join_suggestion', join_suggestion)
            
            # Display join interface
            return self._display_join_interface()
            
        except Exception as e:
            self.view.show_message(f"Error processing join: {str(e)}", "error")
            return False
            
    def _get_join_suggestion(self, available_tables: List[str], tables_data: Dict[str, pd.DataFrame], other_info: str) -> Dict[str, Any]:
        """Get AI suggestion for the next join"""
        try:
            # Prepare tables metadata
            tables_metadata = []
            for table_name in available_tables:
                if table_name in tables_data:
                    df = tables_data[table_name]
                    
                    # Create metadata for this table
                    metadata = {
                        "table_name": table_name,
                        "columns": df.columns.tolist(),
                        "sample_data": df.head(5).to_dict(orient="records"),
                        "statistics": {
                            col: {
                                "dtype": str(df[col].dtype),
                                "nunique": int(df[col].nunique()),
                                "has_nulls": bool(df[col].isnull().any())
                            } for col in df.columns
                        }
                    }
                    
                    # Add numeric column statistics if applicable
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        for col in numeric_cols:
                            metadata["statistics"][col].update({
                                "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                                "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                                "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None
                            })
                    
                    tables_metadata.append(metadata)
            
            # Get onboarding and mapping information to provide smart join suggestions
            session_id = self.session.get('session_id')
            smart_join_guidance = self._get_smart_join_guidance(session_id)
            
            if smart_join_guidance:
                # Add the smart join guidance to other_info
                other_info += f"\n\n{smart_join_guidance}"
            
            # Create task data
            task_data = {
                'available_tables': available_tables,
                'tables_metadata': tables_metadata,
                'other_info': other_info
            }
            
            # Create task
            join_task = Task("Suggest join", data=task_data)
            
            # Execute the task
            join_suggestion = self.join_agent.execute_task(join_task)
            
            return join_suggestion
            
        except Exception as e:
            logger.error(f"Error getting join suggestion: {str(e)}")
            return {}
        
    def _get_smart_join_guidance(self, session_id: str) -> str:
        """
        Generate smart join guidance based on onboarding and mapping data
        """
        try:
            # Connect to database
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                
                # Get onboarding summary - specifically query the is_time_series column
                cursor.execute(
                    "SELECT is_time_series FROM onboarding_summary WHERE session_id = ?",
                    (session_id,)
                )
                time_series_row = cursor.fetchone()
                
                # Get mapping summary
                cursor.execute(
                    "SELECT mappings FROM mappings_summary WHERE session_id = ? AND table_name = '_state_summary'",
                    (session_id,)
                )
                mapping_row = cursor.fetchone()
                
                # Initialize guidance
                guidance = "Smart Join Guidance: "
                
                # Base guidance - always suggest joining on ID
                guidance += "Suggest join on ID columns. "
                
                # Check if this is a time series problem
                is_time_series = False
                if time_series_row:
                    # Print the raw time_series value from the database
                    print(f">>>>> Raw is_time_series value: {time_series_row[0]}")
                    
                    # Check various possible formats of True
                    time_series_value = time_series_row[0]
                    if time_series_value is True or time_series_value == 'True' or time_series_value == 'true' or time_series_value == 1:
                        is_time_series = True
                        print(">>>>> Time series data detected!")
                        guidance += "Also suggest join on date/timestamp columns. "
                
                # Check if this is ID+Product Level prediction
                is_product_level = False
                if mapping_row and mapping_row[0]:
                    try:
                        # Parse the JSON from the mappings column
                        mapping_data = json.loads(mapping_row[0])
                        
                        # Print debug info about mapping data
                        print(f">>>>> Mapping data: {mapping_data}")
                        print(f">>>>> Prediction level: {mapping_data.get('prediction_level')}")
                        
                        # Extract prediction_level from the parsed JSON
                        if mapping_data.get('prediction_level') == 'ID+Product Level':
                            is_product_level = True
                            print(">>>>> ID+Product Level detected!")
                            guidance += "Also suggest join on product ID columns. "
                            
                    except json.JSONDecodeError as e:
                        print(f">>>>> Error parsing mapping JSON: {str(e)}")
                        print(f">>>>> Raw mapping data: {mapping_row[0]}")
                
                # Print debug info about detected features
                print(f">>>>> is_time_series: {is_time_series}, is_product_level: {is_product_level}")
                
                # Combine all guidance
                if is_time_series and is_product_level:
                    guidance = "Smart Join Guidance: Suggest join on ID, product ID, and date/timestamp columns. "
                elif is_time_series:
                    guidance = "Smart Join Guidance: Suggest join on ID and date/timestamp columns. "
                elif is_product_level:
                    guidance = "Smart Join Guidance: Suggest join on ID and product ID columns. "
                
                # Log the final guidance
                print(f">>>>> Generated join guidance: {guidance}")
                
                return guidance
                
        except Exception as e:
            print(f">>>>> Error getting smart join guidance: {str(e)}")
            import traceback
            print(f">>>>> Traceback: {traceback.format_exc()}")
            return ""
            
    def _display_join_interface(self) -> bool:
        """Display join interface and handle user input"""
        try:
            # Get current join suggestion
            join_suggestion = self.session.get('current_join_suggestion', {})
            available_tables = self.session.get('available_tables', [])
            tables_data = self.session.get('tables_data', {})
            joined_tables = self.session.get('joined_tables', {})
            
            # Combine original and joined tables data
            all_tables_data = {**tables_data, **joined_tables}
            
            # Display header
            self.view.display_subheader("Join Tables")
            
            # Display available tables
            self.view.display_markdown("**Available Tables:**")
            table_info = []
            for table_name in available_tables:
                if table_name in all_tables_data:
                    df = all_tables_data[table_name]
                    table_info.append({
                        "Table": table_name,
                        "Rows": df.shape[0],
                        "Columns": df.shape[1]
                    })
            
            self.view.display_table(table_info)
            self.view.display_markdown("---")
            
            # Display important table if set
            important_table = self.session.get('important_table')
            if important_table:
                self.view.display_markdown(f"**Important Table:** {important_table}")
            
            # Display custom instructions if set
            custom_instructions = self.session.get('custom_join_instructions')
            if custom_instructions:
                self.view.display_markdown("**Custom Join Instructions:**")
                self.view.display_markdown(custom_instructions)
            self.view.display_markdown("---")
            
            # Display AI suggestion
            self.view.display_markdown("**🤖 AI Join Suggestion:**")
            tables_to_join = join_suggestion.get("tables_to_join", [])
            join_type = join_suggestion.get("type_of_join", "inner")
            joining_fields = join_suggestion.get("joining_fields", [])
            explanation = join_suggestion.get("explanation", "")
            
            # Use the 5-5-1 column layout for displaying the suggestion
            col1, col2, col3 = self.view.create_columns([5, 5, 1])
            
            with col1:
                self.view.display_markdown("**Tables to Join:**")
                self.view.display_markdown(f"{', '.join(tables_to_join)}")
            
            with col2:
                self.view.display_markdown("**Join Type:**")
                self.view.display_markdown(f"{join_type}")
            
            with col3:
                # Empty column to match layout
                self.view.display_markdown("&nbsp;")
            
            # Display joining fields with better formatting
            self.view.display_markdown("**Joining Fields:**")
            for field_pair in joining_fields:
                if len(field_pair) == 2:
                    # Create columns for each join condition
                    col1, col2, col3 = self.view.create_columns([5, 5, 1])
                    
                    with col1:
                        self.view.display_markdown(f"**Left:** {field_pair[0]}")
                    
                    with col2:
                        self.view.display_markdown(f"**Right:** {field_pair[1]}")
                    
                    with col3:
                        # Empty column to match layout
                        self.view.display_markdown("&nbsp;")
            
            self.view.display_markdown("**Explanation:**")
            self.view.display_markdown(explanation)
            self.view.display_markdown("---")
            
            # Let user modify the suggestion
            self.view.display_markdown("**Customize Join:**")
            
            # Get initial values from suggestion
            left_table = tables_to_join[0] if len(tables_to_join) > 0 else None
            right_table = tables_to_join[1] if len(tables_to_join) > 1 else None

            # Create side-by-side columns for table selection with the same ratio as join conditions
            col1, col2, col3 = self.view.create_columns([5, 5, 1])

            # Left table selection
            with col1:
                self.view.display_markdown("**Left Table:**")
                left_table = self.view.select_box(
                    "Select left table:",
                    options=available_tables,
                    index=available_tables.index(left_table) if left_table in available_tables else 0,
                    key="left_table"
                )
            
            # Right table selection
            with col2:
                self.view.display_markdown("**Right Table:**")
                # Filter out the left table from options
                right_options = [t for t in available_tables if t != left_table]
                right_table = self.view.select_box(
                    "Select right table:",
                    options=right_options,
                    index=right_options.index(right_table) if right_table in right_options else 0,
                    key="right_table"
                )
            
            # Empty third column for consistency
            with col3:
                self.view.display_markdown("&nbsp;")

            # Join type selection (common for both tables)
            self.view.display_markdown("**Join type:**")
            join_type = self.view.select_box(
                "",  # Empty label to save space
                options=["inner", "left"],
                index=0 if join_type == "inner" else 1,
                key="join_type"
            )
            
            # Get columns for both tables
            left_columns = []
            right_columns = []
            
            if left_table and left_table in all_tables_data:
                left_columns = all_tables_data[left_table].columns.tolist()
            
            if right_table and right_table in all_tables_data:
                right_columns = all_tables_data[right_table].columns.tolist()
            
            # Initialize joining fields from suggestion or session
            current_joining_fields = self.session.get('current_joining_fields')
            if not current_joining_fields:
                # Convert suggestion to the format we need
                current_joining_fields = []
                for field_pair in joining_fields:
                    if len(field_pair) == 2 and field_pair[0] in left_columns and field_pair[1] in right_columns:
                        current_joining_fields.append(field_pair)
            
            # Display existing join conditions with side-by-side columns
            self.view.display_markdown("**Join Conditions:**")

            # Initialize new joining fields list
            new_joining_fields = []

            # Display existing join conditions with better layout
            for i, (left_field, right_field) in enumerate(current_joining_fields):
                # Create a row with two main columns for fields and one small column for the remove button
                col1, col2, col3 = self.view.create_columns([5, 5, 1])
                
                with col1:
                    self.view.display_markdown(f"**Left field #{i+1}:**")
                    selected_left = self.view.select_box(
                        "",  # Empty label to save space
                        options=left_columns,
                        index=left_columns.index(left_field) if left_field in left_columns else 0,
                        key=f"left_field_{i}"
                    )
                
                with col2:
                    self.view.display_markdown(f"**Right field #{i+1}:**")
                    selected_right = self.view.select_box(
                        "",  # Empty label to save space
                        options=right_columns,
                        index=right_columns.index(right_field) if right_field in right_columns else 0,
                        key=f"right_field_{i}"
                    )
                
                with col3:
                    # Add some vertical spacing to align the button with the dropdowns
                    self.view.display_markdown("&nbsp;")  # Non-breaking space for vertical alignment
                    if self.view.display_button("❌", key=f"remove_condition_{i}"):
                        # Skip this condition by not adding it to new_joining_fields
                        continue
                
                # Add to new joining fields
                new_joining_fields.append([selected_left, selected_right])

            # Add a separator between conditions
            if current_joining_fields:
                self.view.display_markdown("---")

            # Add button for new condition
            if self.view.display_button("➕ Add Join Condition", key="add_condition"):
                # Add a new empty condition
                if left_columns and right_columns:
                    new_joining_fields.append([left_columns[0], right_columns[0]])
                    # Save to session and rerun to refresh UI
                self.session.set('current_joining_fields', new_joining_fields)
                self.view.rerun_script()
                return False
            
            # Save current joining fields to session
            self.session.set('current_joining_fields', new_joining_fields)
            
            # Preview the join if possible
            if left_table and right_table and new_joining_fields:
                self.view.display_markdown("---")
                self.view.display_markdown("**Join Preview:**")
                
                try:
                    # Get dataframes
                    left_df = all_tables_data[left_table]
                    right_df = all_tables_data[right_table]
                    
                    # Extract left and right joining fields for the join operation
                    left_joining_fields = [pair[0] for pair in new_joining_fields]
                    right_joining_fields = [pair[1] for pair in new_joining_fields]
                    
                    # Perform the join
                    joined_df, result_table_name = self._perform_join_operation(left_table, right_table, left_joining_fields, right_joining_fields, join_type)
                    
                    # Display preview
                    self.view.display_markdown(f"**Result Shape:** {joined_df.shape[0]} rows × {joined_df.shape[1]} columns")
                    self.view.display_dataframe(joined_df.head(5))
                    
                except Exception as e:
                    self.view.show_message(f"Error previewing join: {str(e)}", "error")
            
            # Confirm button
            self.view.display_markdown("---")
            if self.view.display_button("✅ Confirm Join", key="confirm_join"):
                # Validate join
                if not left_table or not right_table:
                    self.view.show_message("❌ Both tables must be selected", "error")
                    return False
                    
                if not new_joining_fields:
                    self.view.show_message("❌ At least one join condition must be specified", "error")
                    return False
                
                # Perform the join
                try:
                    # Get dataframes
                    left_df = all_tables_data[left_table]
                    right_df = all_tables_data[right_table]
                    
                    # Extract left and right joining fields for the join operation
                    left_joining_fields = [pair[0] for pair in new_joining_fields]
                    right_joining_fields = [pair[1] for pair in new_joining_fields]
                    
                    # Perform the join
                    joined_df, result_table_name = self._perform_join_operation(left_table, right_table, left_joining_fields, right_joining_fields, join_type)
                    
                    # Create new table name
                    new_table_name = result_table_name
                    
                    # Update session data
                    joined_tables = self.session.get('joined_tables', {})
                    joined_tables[new_table_name] = joined_df
                    self.session.set('joined_tables', joined_tables)
                    
                    # Update available tables
                    available_tables = self.session.get('available_tables', [])
                    available_tables.remove(left_table)
                    available_tables.remove(right_table)
                    available_tables.append(new_table_name)
                    self.session.set('available_tables', available_tables)
                    
                    # Add to join history
                    join_history = self.session.get('join_history', [])
                    join_history.append({
                        "left_table": left_table,
                        "right_table": right_table,
                        "join_type": join_type,
                        "joining_fields": new_joining_fields,
                        "result_table": new_table_name,
                        "result_shape": joined_df.shape
                    })
                    self.session.set('join_history', join_history)
                    
                    # Save join to database
                    self._save_join_to_db(
                        left_table, 
                        right_table, 
                        join_type, 
                        new_joining_fields, 
                        new_table_name, 
                        joined_df
                    )
                    
                    # Clear current join suggestion
                    self.session.set('current_join_suggestion', None)
                    self.session.set('current_joining_fields', None)
                    
                    # Show success message
                    self.view.show_message(f"✅ Tables joined successfully! Created {new_table_name}", "success")
                    
                    # Rerun to refresh UI
                    self.view.rerun_script()
                    return False
                    
                except Exception as e:
                    self.view.show_message(f"Error performing join: {str(e)}", "error")
                    return False
            
            return False
            
        except Exception as e:
            self.view.show_message(f"Error displaying join interface: {str(e)}", "error")
            return False
            
    def _perform_join_operation(self, left_table: str, right_table: str, left_columns: List[str], 
                               right_columns: List[str], join_type: str) -> Tuple[pd.DataFrame, str]:
        """Perform the actual join operation between two tables"""
        try:
            # Get tables data
            tables_data = self.session.get('tables_data', {})
            joined_tables = self.session.get('joined_tables', {})
            
            # Combine original and joined tables data
            all_tables_data = {**tables_data, **joined_tables}
            
            # Get dataframes
            left_df = all_tables_data[left_table]
            right_df = all_tables_data[right_table]
            
            # Create a copy to avoid modifying the original
            left_df_copy = left_df.copy()
            right_df_copy = right_df.copy()
            
            # Perform the join
            if join_type == 'inner':
                result_df = pd.merge(left_df_copy, right_df_copy, left_on=left_columns, right_on=right_columns, how='inner')
            elif join_type == 'left':
                result_df = pd.merge(left_df_copy, right_df_copy, left_on=left_columns, right_on=right_columns, how='left')
            else:
                # Default to inner join
                result_df = pd.merge(left_df_copy, right_df_copy, left_on=left_columns, right_on=right_columns, how='inner')
            
            # Clean up table names for the result table name
            # Remove .csv extension if present
            clean_left_name = left_table.replace('.csv', '')
            clean_right_name = right_table.replace('.csv', '')
            
            # Create a name for the joined table
            result_table_name = f"joined_{clean_left_name}_{clean_right_name}"
            
            return result_df, result_table_name
            
        except Exception as e:
            logger.error(f"Error performing join: {str(e)}")
            raise e
        
    def _save_join_to_db(self, left_table: str, right_table: str, join_type: str, 
                         joining_fields: List[List[str]], result_table: str, 
                         joined_df: pd.DataFrame) -> bool:
        """Save join information to database"""
        try:
            session_id = self.session.get('session_id')
            
            # Connect to database
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                
                # Create join_details table if it doesn't exist (renamed from join_results)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS join_details (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        left_table TEXT,
                        right_table TEXT,
                        join_type TEXT,
                        joining_fields TEXT,
                        result_table TEXT,
                        data TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Insert join result
                cursor.execute(
                    """
                    INSERT INTO join_details 
                    (session_id, left_table, right_table, join_type, joining_fields, result_table, data) 
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        left_table,
                        right_table,
                        join_type,
                        json.dumps(joining_fields),
                        result_table,
                        joined_df.to_json(orient='records')
                    )
                )
                
                conn.commit()
                
            return True
        
        except Exception as e:
            logger.error(f"Error saving join to database: {str(e)}")
            return False
            
    def _show_final_summary(self) -> bool:
        """Show final summary and proceed button"""
        try:
            # Get join history
            join_history = self.session.get('join_history', [])
            
            # Get final joined table
            available_tables = self.session.get('available_tables', [])
            joined_tables = self.session.get('joined_tables', {})
            
            if not available_tables or not joined_tables:
                self.view.show_message("❌ No joined tables found", "error")
                return False
                
            final_table_name = available_tables[0]
            final_table = joined_tables.get(final_table_name)
            
            if final_table is None:
                self.view.show_message("❌ Final joined table not found", "error")
                return False
                
            # Create a standardized final table name based on session ID
            session_id = self.session.get('session_id')
            short_session = session_id.split('-')[0] if session_id else 'joined'
            final_joined_table_name = f"joined_table_{short_session}"
            
            # Display header
            self.view.display_subheader("Join Process Complete")
            
            # Display join history
            self.view.display_markdown("**Join History:**")
            
            for i, join in enumerate(join_history):
                self.view.display_markdown(f"**Join {i+1}:**")
                self.view.display_markdown(f"- Left Table: {join.get('left_table', 'Unknown')}")
                self.view.display_markdown(f"- Right Table: {join.get('right_table', 'Unknown')}")
                self.view.display_markdown(f"- Join Type: {join.get('join_type', 'inner')}")
                self.view.display_markdown("- Join Conditions:")
                
                # Handle different formats of joining_fields
                joining_fields = join.get('joining_fields', [])
                if isinstance(joining_fields, list):
                    for field_pair in joining_fields:
                        if isinstance(field_pair, list) and len(field_pair) >= 2:
                            self.view.display_markdown(f"  - {field_pair[0]} = {field_pair[1]}")
                
                # Display result information if available
                result_table = join.get('result_table')
                result_shape = join.get('result_shape')
                if result_table:
                    shape_info = f" ({result_shape[0]} rows × {result_shape[1]} columns)" if result_shape else ""
                    self.view.display_markdown(f"- Result: {result_table}{shape_info}")
                
                self.view.display_markdown("---")
            
            # Display final table
            self.view.display_markdown("**Final Joined Table:**")
            self.view.display_markdown(f"**Name:** {final_joined_table_name}")
            self.view.display_markdown(f"**Shape:** {final_table.shape[0]} rows × {final_table.shape[1]} columns")
            self.view.display_markdown("**Preview:**")
            self.view.display_dataframe(final_table.head(5))
            
            # Save the final table to the database with the standardized name
            try:
                with sqlite3.connect(self.db.db_path) as conn:
                    # Save the final table with the standardized name
                    final_table.to_sql(final_joined_table_name, conn, if_exists='replace', index=False)
                    print(f">>> Saved final joined table as {final_joined_table_name}")
            except Exception as e:
                print(f">>> Error saving final joined table: {str(e)}")
            
            # Create state summary with the standardized table name
            self._create_state_summary(final_joined_table_name, final_table, join_history)
            
            # Get problem type from mapping summary
            mapping_summary = self.session.get('mapping_summary', {})
            problem_type = mapping_summary.get('confirmed_mapping', {}).get('problem_type', '').lower()
            
            # Store the final table name and data in session
            self.session.set('final_table_name', final_joined_table_name)
            self.session.set('final_table', final_table)
            
            # Check if we need to get final mappings
            if not self.session.get('final_mappings_complete'):
                # Set a flag to indicate we're ready for final mappings
                self.session.set('ready_for_final_mappings', True)
                
                # Get final mappings for the joined table
                mappings_result = self._get_final_mappings()
                
                if mappings_result is None:
                    # Still waiting for user input on mappings
                    return False
                elif not mappings_result:
                    # Error occurred
                    return False
                
                # Mark final mappings as complete
                self.session.set('final_mappings_complete', True)
            
            # Show proceed button with dynamic text based on problem type
            self.view.display_markdown("---")
            
            button_text = "▶️ Proceed to Next Step"
            next_state = "next"
            
            # Customize button text based on problem type
            if problem_type == 'clustering':
                button_text = "▶️ Proceed to Clustering"
                next_state = "clustering"
            elif problem_type == 'classification':
                button_text = "▶️ Proceed to Classification"
                next_state = "classification"
            elif problem_type == 'regression':
                button_text = "▶️ Proceed to Regression"
                next_state = "regression"
            elif problem_type == 'forecasting':
                button_text = "▶️ Proceed to Forecasting"
                next_state = "forecasting"
            elif problem_type == 'recommendation':
                button_text = "▶️ Proceed to Recommendation"
                next_state = "recommendation"
            
            if self.view.display_button(button_text):
                self.session.set('join_complete', True)
                self.session.set('next_state', next_state)
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error showing final summary: {str(e)}")
            self.view.show_message(f"Error showing final summary: {str(e)}", "error")
            return False
    
    def _create_state_summary(self, final_table_name: str, final_table: pd.DataFrame, join_history: List[Dict]) -> None:
        """Create a summary of the join state"""
        try:
            session_id = self.session.get('session_id')
            
            summary = f"### Step 4: Table Joins\n\n"
            summary += f"**Number of Join Operations:** {len(join_history)}\n"
            summary += f"**Final Table:** {final_table_name}\n"
            summary += f"**Shape:** {final_table.shape[0]} rows × {final_table.shape[1]} columns\n\n"
            
            # Store the summary in the session
            self.session.set("step_4_summary", summary)
            
            # Save to database
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                
                # Create join_summary table if it doesn't exist
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS join_summary (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        summary TEXT,
                        final_table_name TEXT,
                        join_history TEXT,
                        final_mappings TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Check if a summary already exists
                cursor.execute(
                    "SELECT id FROM join_summary WHERE session_id = ?",
                    (session_id,)
                )
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing summary
                    cursor.execute(
                        "UPDATE join_summary SET summary = ?, final_table_name = ?, join_history = ? WHERE session_id = ?",
                        (summary, final_table_name, json.dumps(join_history), session_id)
                    )
                else:
                    # Insert new summary
                    cursor.execute(
                        "INSERT INTO join_summary (session_id, summary, final_table_name, join_history) VALUES (?, ?, ?, ?)",
                        (session_id, summary, final_table_name, json.dumps(join_history))
                    )
                
                # Make sure the final table is available in the database
                try:
                    # Save to database with the final table name
                    final_table.to_sql(final_table_name, conn, if_exists='replace', index=False)
                    print(f">>> Saved final table as {final_table_name}")
                except Exception as e:
                    print(f">>> Error saving final table: {str(e)}")
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error creating state summary: {str(e)}")
            
    def _show_state_summary(self) -> None:
        """Show state summary when already complete"""
        summary = self.session.get("step_4_summary", "Join process complete.")
        self.view.show_message(summary, "success")

    def _handle_single_table_case(self) -> bool:
        """Handle the case where we only have one table (no joins needed)"""
        try:
            # Get the single table
            tables_data = self.session.get('tables_data', {})
            if not tables_data:
                self.view.show_message("❌ No tables data found", "error")
                return False
            
            # Get the single table name and data
            table_name = list(tables_data.keys())[0]
            table_data = tables_data[table_name]
            
            # Create a standardized final table name based on session ID
            session_id = self.session.get('session_id')
            short_session = session_id.split('-')[0] if session_id else 'single'
            final_table_name = f"joined_table_{short_session}"
            
            # Display header
            self.view.display_header("Single Table Workflow")
            self.view.display_markdown("Only one table is available, so no joins are needed.")
            
            # Display table info
            self.view.display_subheader("Table Information")
            self.view.display_markdown(f"**Table Name:** {table_name}")
            self.view.display_markdown(f"**Shape:** {table_data.shape[0]} rows × {table_data.shape[1]} columns")
            self.view.display_markdown("**Preview:**")
            self.view.display_dataframe(table_data.head(5))
            
            # Create a summary for the single table case
            self._create_single_table_summary(table_name, final_table_name, table_data)
            
            # Store the table in the session as if it was a joined table
            available_tables = [final_table_name]
            joined_tables = {final_table_name: table_data}
            
            self.session.set('available_tables', available_tables)
            self.session.set('joined_tables', joined_tables)
            self.session.set('final_table_name', final_table_name)
            self.session.set('final_table', table_data)
            
            # Create an empty join history
            join_history = []
            self.session.set('join_history', join_history)
            
            # Check if we need to get final mappings
            if not self.session.get('final_mappings_complete'):
                # Set a flag to indicate we're ready for final mappings
                self.session.set('ready_for_final_mappings', True)
                
                # Get final mappings for the single table
                mappings_result = self._get_final_mappings()
                
                if mappings_result is None:
                    # Still waiting for user input on mappings
                    return False
                elif not mappings_result:
                    # Error occurred
                    return False
                
                # Mark final mappings as complete
                self.session.set('final_mappings_complete', True)
            
            # Get problem type from mapping summary
            mapping_summary = self.session.get('mapping_summary', {})
            problem_type = mapping_summary.get('confirmed_mapping', {}).get('problem_type', '').lower()
            
            # Show proceed button with dynamic text based on problem type
            self.view.display_markdown("---")
            
            button_text = "▶️ Proceed to Next Step"
            next_state = "next"
            
            # Customize button text based on problem type
            if problem_type == 'clustering':
                button_text = "▶️ Proceed to Clustering"
                next_state = "clustering"
            elif problem_type == 'classification':
                button_text = "▶️ Proceed to Classification"
                next_state = "classification"
            elif problem_type == 'regression':
                button_text = "▶️ Proceed to Regression"
                next_state = "regression"
            elif problem_type == 'forecasting':
                button_text = "▶️ Proceed to Forecasting"
                next_state = "forecasting"
            elif problem_type == 'recommendation':
                button_text = "▶️ Proceed to Recommendation"
                next_state = "recommendation"
            
            if self.view.display_button(button_text):
                self.session.set('join_complete', True)
                self.session.set('next_state', next_state)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error handling single table case: {str(e)}")
            self.view.show_message(f"Error handling single table case: {str(e)}", "error")
            return False

    def _fetch_aggregation_table_name(self, session_id: str, table_name: str) -> Dict:
        """Fetch aggregated table name from aggregation summary"""
        try:
            # Connect to database
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                
                # Query the aggregation_summary table
                cursor.execute(
                    """
                    SELECT * FROM aggregation_summary 
                    WHERE session_id = ? AND table_name = ?
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (session_id, table_name)
                )
                
                result = cursor.fetchone()
                
                if result:
                    # Convert to dictionary
                    columns = [col[0] for col in cursor.description]
                    summary = {columns[i]: result[i] for i in range(len(columns))}
                    return summary
                
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching aggregation table name: {str(e)}")
            return {}

    def _create_single_table_summary(self, table_name: str, final_table_name: str, table_data: pd.DataFrame) -> None:
        """Create a summary for the single table case"""
        try:
            session_id = self.session.get('session_id')
            
            summary = f"### Step 4: Table Joins\n\n"
            summary += "**No join operations needed - single table workflow**\n\n"
            summary += f"**Original Table:** {table_name}\n"
            summary += f"**Final Table:** {final_table_name}\n"
            summary += f"**Shape:** {table_data.shape[0]} rows × {table_data.shape[1]} columns\n\n"
            
            # Store the summary in the session
            self.session.set("step_4_summary", summary)
            self.session.set("final_joined_table_name", final_table_name)
            
            # Save to database
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                
                # Create join_summary table if it doesn't exist
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS join_summary (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        summary TEXT,
                        final_table_name TEXT,
                        join_history TEXT,
                        final_mappings TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Check if a summary already exists
                cursor.execute(
                    "SELECT id FROM join_summary WHERE session_id = ?",
                    (session_id,)
                )
                existing = cursor.fetchone()
                
                # Create a simplified join history
                join_history = [{
                    "message": "No join operations needed - single table workflow",
                    "table_name": table_name,
                    "final_table_name": final_table_name
                }]
                
                if existing:
                    # Update existing summary
                    cursor.execute(
                        "UPDATE join_summary SET summary = ?, final_table_name = ?, join_history = ? WHERE session_id = ?",
                        (summary, final_table_name, json.dumps(join_history), session_id)
                    )
                else:
                    # Insert new summary
                    cursor.execute(
                        "INSERT INTO join_summary (session_id, summary, final_table_name, join_history) VALUES (?, ?, ?, ?)",
                        (session_id, summary, final_table_name, json.dumps(join_history))
                    )
                
                # Make sure the final table is available in the database
                # If it's not already there, save it
                try:
                    table_data_copy = table_data
                    # Save to database with the final table name
                    table_data_copy.to_sql(final_table_name, conn, if_exists='replace', index=False)
                    print(f">>> Saved final table as {final_table_name}")
                except Exception as e:
                    print(f">>> Error saving final table: {str(e)}")
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error creating single table summary: {str(e)}")

    def _get_final_mappings(self):
        """
        Get final mappings for the joined table based on problem type and requirements.
        This method is called after join is complete to ensure mappings are accurate.
        """
        try:
            # Get session ID and problem type
            session_id = self.session.get('session_id')
            problem_type = self.session.get('problem_type', '')
            
            # Get the final joined table
            final_table_name = self.session.get('final_table_name')
            if not final_table_name:
                self.view.show_message("❌ Final table name not found", "error")
                return False
            
            # Load the final table data
            from orchestrator.storage.db_connector import DatabaseConnector
            import sqlite3
            import pandas as pd
            
            db = DatabaseConnector()
            with sqlite3.connect(db.db_path) as conn:
                # Get the final joined dataframe
                df = pd.read_sql(f"SELECT * FROM {final_table_name}", conn)
                
                if df.empty:
                    self.view.show_message("❌ Final joined table is empty", "error")
                    return False
                
                # Get existing mappings from mapping state
                existing_mappings = self._get_existing_mappings(session_id, conn)
                
                # Get onboarding summary for additional requirements
                onboarding_summary = self._get_onboarding_summary(session_id, conn)
                
                # Determine mandatory and optional columns based on problem type and onboarding
                mandatory_columns, optional_columns = self._get_mapping_requirements(problem_type, onboarding_summary)
                
                # Check which mandatory columns are missing from existing mappings
                missing_mappings = []
                for col in mandatory_columns:
                    if col not in existing_mappings or not existing_mappings[col]:
                        missing_mappings.append(col)
            
            # Always show the user interface for confirming mappings
            # This ensures the user always gets a chance to review and confirm
            return self._get_user_mappings(df, existing_mappings, missing_mappings, mandatory_columns, optional_columns)
            
        except Exception as e:
            import traceback
            print(f"Error getting final mappings: {str(e)}")
            print(traceback.format_exc())
            self.view.show_message(f"❌ Error getting final mappings: {str(e)}", "error")
            return False

    def _get_existing_mappings(self, session_id, conn):
        """Get existing mappings from mapping state"""
        try:
            import json
            cursor = conn.cursor()
            
            # First try to get table-specific mappings
            cursor.execute(
                "SELECT table_name, mappings FROM mappings_summary WHERE session_id = ? AND table_name != '_state_summary'",
                (session_id,)
            )
            results = cursor.fetchall()
            
            # If no table-specific mappings, try to get from state summary
            if not results:
                cursor.execute(
                    "SELECT mappings FROM mappings_summary WHERE session_id = ? AND table_name = '_state_summary'",
                    (session_id,)
                )
                state_result = cursor.fetchone()
                
                if state_result:
                    # Parse state summary mappings
                    state_mappings = json.loads(state_result[0])
                    
                    # Extract any field mappings from state summary
                    field_mappings = {}
                    # Look for any key that might be a field mapping
                    for key, value in state_mappings.items():
                        if key not in ['tables_mapped', 'mandatory_columns_mapped', 'prediction_level', 
                                      'has_product_mapping', 'problem_type', 'completion_time']:
                            field_mappings[key] = value
                    
                    return field_mappings
                else:
                    # No mappings found at all
                    return {}
            else:
                # Combine mappings from all tables
                field_mappings = {}
                for table_name, mappings_json in results:
                    table_mappings = json.loads(mappings_json)
                    field_mappings.update(table_mappings)
                
                return field_mappings
                
        except Exception as e:
            print(f"Error getting existing mappings: {str(e)}")
            return {}

    def _get_onboarding_summary(self, session_id, conn):
        """Get onboarding summary from database"""
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM onboarding_summary WHERE session_id = ?",
                (session_id,)
            )
            row = cursor.fetchone()
            
            if row:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, row))
            return {}
            
        except Exception as e:
            print(f"Error getting onboarding summary: {str(e)}")
            return {}

    def _get_mapping_requirements(self, problem_type, onboarding_summary):
        """
        Determine mandatory and optional columns based on problem type and onboarding summary
        """
        # Base mandatory columns by problem type
        base_mandatory = {
            'classification': ['id'],
            'regression': ['id'],
            'recommendation': ['product_id'],
            'clustering': ['id'],
            'forecasting': ['timestamp', 'target']
        }
        
        # Get base mandatory columns for this problem type
        mandatory_columns = base_mandatory.get(problem_type, [])
        
        # Add additional mandatory columns based on onboarding summary
        if problem_type in ['regression', 'classification']:
            # If has_target is true, target is mandatory
            # if onboarding_summary.get('has_target') == 'True':
                # if 'target' not in mandatory_columns:
            mandatory_columns.append('target')
                
            # If is_time_series is true, timestamp is mandatory
            if onboarding_summary.get('is_time_series') == 'True':
                if 'timestamp' not in mandatory_columns:
                    mandatory_columns.append('timestamp')
                
            # If prediction_level is "ID+Product Level", product_id is mandatory
            if onboarding_summary.get('prediction_level') == 'ID+Product Level':
                if 'product_id' not in mandatory_columns:
                    mandatory_columns.append('product_id')
        
        # For recommendation, if approach is user-based, id is mandatory
        if problem_type == 'recommendation':
            if onboarding_summary.get('recommendation_approach') == 'user_based':
                if 'id' not in mandatory_columns:
                    mandatory_columns.append('id')
        
        # Get optional columns based on problem type
        optional_mapping = {
            'classification': ['product_id', 'timestamp', 'revenue'],
            'regression': ['product_id', 'timestamp', 'revenue'],
            'recommendation': ['id', 'interaction_value', 'timestamp'],
            'clustering': ['product_id', 'timestamp', 'revenue'],
            'forecasting': ['product_id', 'id', 'revenue']
        }
        
        optional_columns = optional_mapping.get(problem_type, [])
        
        return mandatory_columns, optional_columns

    def _get_user_mappings(self, df, existing_mappings, missing_mappings, mandatory_columns, optional_columns):
        """Get mappings from user for all columns, with existing mappings pre-filled"""
        self.view.display_header("Final Column Mapping")
        
        # Explain to the user what's happening
        self.view.display_markdown(
            "Please review and confirm the column mappings for your selected problem type. "
            "You can modify any mapping by selecting a different column from the dropdown."
        )
        
        # Show the first few rows of the dataframe
        self.view.display_subheader("Preview of Joined Data")
        self.view.display_dataframe(df.head())
        
        # Create a form for user to select mappings
        self.view.display_subheader("Map Columns")
        
        # For each mapping type, create a dropdown
        updated_mappings = existing_mappings.copy()
        columns = df.columns.tolist()
        
        # Combine mandatory and optional columns for display
        all_mapping_types = mandatory_columns + [col for col in optional_columns if col not in mandatory_columns]
        
        for col_type in all_mapping_types:
            # Create a friendly display name
            display_name = col_type.replace('_', ' ').title()
            
            # Add (Required) to mandatory columns
            if col_type in mandatory_columns:
                display_name += " (Required)"
            
            # Create dropdown with None as first option
            options = ["None"] + columns
            default_idx = 0  # Default to None
            
            # If we have an existing mapping that's in the columns, use it as default
            if col_type in existing_mappings and existing_mappings[col_type] in columns:
                default_idx = columns.index(existing_mappings[col_type]) + 1  # +1 because "None" is first
            
            selected = self.view.selectbox(
                f"Select column for {display_name}:",
                options=options,
                index=default_idx,
                key=f"mapping_{col_type}"
            )
            
            # Update mapping if not None
            if selected != "None":
                updated_mappings[col_type] = selected
            else:
                # If None is selected, remove any existing mapping
                if col_type in updated_mappings:
                    updated_mappings.pop(col_type)
        
        # Add a button to confirm mappings
        if self.view.display_button("Confirm Mappings", key="confirm_final_mappings"):
            # Check if all mandatory columns are mapped
            missing = [col for col in mandatory_columns if col not in updated_mappings or not updated_mappings[col]]
            
            if missing:
                # Show error for missing mandatory columns
                missing_names = [col.replace('_', ' ').title() for col in missing]
                self.view.show_message(
                    f"❌ Please map all required columns: {', '.join(missing_names)}",
                    "error"
                )
                return None  # Return None to keep UI state
            else:
                # Save the mappings and continue
                session_id = self.session.get('session_id')
                self._save_final_mappings(updated_mappings, session_id)
                return True
        
        return None  # Return None to keep UI state

    def _save_final_mappings(self, mappings, session_id):
        """Save final mappings to database"""
        try:
            import json
            import sqlite3
            from orchestrator.storage.db_connector import DatabaseConnector
            
            db = DatabaseConnector()
            
            # Convert mappings to JSON
            mappings_json = json.dumps(mappings)
            
            # Save to join_summary table
            with sqlite3.connect(db.db_path) as conn:
                cursor = conn.cursor()
                
                # First, check if the join_summary table exists
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='join_summary'")
                table_exists = cursor.fetchone() is not None
                
                if not table_exists:
                    # Create the table with the final_mappings column
                    cursor.execute("""
                        CREATE TABLE join_summary (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id TEXT,
                            summary TEXT,
                            final_table_name TEXT,
                            join_history TEXT,
                            final_mappings TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    print("Created join_summary table with final_mappings column")
                else:
                    # Check if final_mappings column exists
                    cursor.execute("PRAGMA table_info(join_summary)")
                    columns = [column[1] for column in cursor.fetchall()]
                    
                    # Add the column if it doesn't exist
                    if 'final_mappings' not in columns:
                        cursor.execute("ALTER TABLE join_summary ADD COLUMN final_mappings TEXT")
                        print("Added final_mappings column to join_summary table")
                
                # Check if there's an existing row for this session
                cursor.execute(
                    "SELECT id FROM join_summary WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
                    (session_id,)
                )
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing row
                    cursor.execute(
                        """
                        UPDATE join_summary 
                        SET final_mappings = ? 
                        WHERE id = ?
                        """,
                        (mappings_json, existing[0])
                    )
                
                else:
                    # Insert a new row
                    final_table_name = self.session.get('final_table_name', '')
                    cursor.execute(
                        """
                        INSERT INTO join_summary 
                        (session_id, final_mappings, final_table_name) 
                        VALUES (?, ?, ?)
                        """,
                        (session_id, mappings_json, final_table_name)
                    )
                
                
                conn.commit()
            
            # Also save to mapping_summary table for consistency
            # Instead of using a non-existent method, use the existing table structure
            with sqlite3.connect(db.db_path) as conn:
                cursor = conn.cursor()
                
                # Create or update the mapping_summary entry with the final mappings
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS mappings_summary (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        table_name TEXT,
                        mappings TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Save as a special '_final_mappings' entry
                cursor.execute(
                    """
                    INSERT INTO mappings_summary (session_id, table_name, mappings)
                    VALUES (?, '_final_mappings', ?)
                    """,
                    (session_id, mappings_json)
                )
                
                conn.commit()
            
            self.view.show_message("✅ Final mappings saved successfully", "success")
            return True
            
        except Exception as e:
            import traceback
            print(f"Error saving final mappings: {str(e)}")
            print(traceback.format_exc())
            self.view.show_message(f"❌ Error saving final mappings: {str(e)}", "error")
            return False