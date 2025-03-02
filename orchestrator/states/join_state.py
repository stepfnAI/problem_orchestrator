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
        
    def execute(self):
        """Execute the join workflow"""
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
            
            print(f">>> Processing join with {len(available_tables)} available tables")
            print(f">>> Available tables: {available_tables}")
            
            # Combine original and joined tables data
            all_tables_data = {**tables_data, **joined_tables}
            
            # Check if we have a suggested join
            if not self.session.get('current_join_suggestion'):
                # Get AI suggestion for the next join
                print(">>> Getting AI join suggestion")
                with self.view.display_spinner('🤖 AI is analyzing tables for join suggestions...'):
                    join_suggestion = self._get_join_suggestion(available_tables, all_tables_data)
                    
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
            
    def _get_join_suggestion(self, available_tables: List[str], tables_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
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
            
            # Get mappings information
            table_mappings = self.session.get('table_mappings', {})
            
            # Create task data
            task_data = {
                'available_tables': available_tables,
                'tables_metadata': tables_metadata,
                'other_info': f"Table mappings: {json.dumps(table_mappings)}"
            }
            
            # Create task
            join_task = Task("Suggest join", data=task_data)
            
            # Execute the task
            join_suggestion = self.join_agent.execute_task(join_task)
            
            return join_suggestion
            
        except Exception as e:
            logger.error(f"Error getting join suggestion: {str(e)}")
            return {}
            
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
            
            # Display AI suggestion
            self.view.display_markdown("**🤖 AI Join Suggestion:**")
            tables_to_join = join_suggestion.get("tables_to_join", [])
            join_type = join_suggestion.get("type_of_join", "inner")
            joining_fields = join_suggestion.get("joining_fields", [])
            explanation = join_suggestion.get("explanation", "")
            
            self.view.display_markdown(f"**Tables to Join**: {', '.join(tables_to_join)}")
            self.view.display_markdown(f"**Join Type**: {join_type}")
            
            self.view.display_markdown("**Joining Fields:**")
            for field_pair in joining_fields:
                if len(field_pair) == 2:
                    self.view.display_markdown(f"- {field_pair[0]} = {field_pair[1]}")
            
            self.view.display_markdown("**Explanation:**")
            self.view.display_markdown(explanation)
            self.view.display_markdown("---")
            
            # Let user modify the suggestion
            self.view.display_markdown("**Customize Join:**")
            
            # Select left table
            left_table = self.view.select_box(
                "Left Table:",
                options=available_tables,
                default=tables_to_join[0] if len(tables_to_join) > 0 else available_tables[0],
                key="left_table"
            )
            
            # Select right table
            right_table_options = [t for t in available_tables if t != left_table]
            right_table = self.view.select_box(
                "Right Table:",
                options=right_table_options,
                default=tables_to_join[1] if len(tables_to_join) > 1 and tables_to_join[1] in right_table_options else right_table_options[0],
                key="right_table"
            )
            
            # Select join type
            join_type = self.view.select_box(
                "Select join type:",
                options=["left", "inner"],  # Removed right, outer, and cross join options
                key="join_type"
            )
            
            # Get columns for both tables
            left_columns = all_tables_data[left_table].columns.tolist() if left_table in all_tables_data else []
            right_columns = all_tables_data[right_table].columns.tolist() if right_table in all_tables_data else []
            
            # Initialize joining fields if not already set
            if not self.session.get('current_joining_fields'):
                # Use AI suggestion if available
                current_joining_fields = []
                for field_pair in joining_fields:
                    if len(field_pair) == 2:
                        left_col = field_pair[0]
                        right_col = field_pair[1]
                        
                        # Check if columns exist in selected tables
                        if left_col in left_columns and right_col in right_columns:
                            current_joining_fields.append([left_col, right_col])
                
                # If no valid fields found, initialize with empty list
                self.session.set('current_joining_fields', current_joining_fields)
            
            # Get current joining fields
            current_joining_fields = self.session.get('current_joining_fields', [])
            
            # Display joining fields interface
            self.view.display_markdown("**Joining Fields:**")
            
            # Display existing joining fields
            new_joining_fields = []
            removed_indices = []
            
            for i, field_pair in enumerate(current_joining_fields):
                left_col = field_pair[0] if len(field_pair) > 0 else ""
                right_col = field_pair[1] if len(field_pair) > 1 else ""
                
                self.view.display_markdown(f"**Join Condition {i+1}:**")
                
                # Select left column
                left_col = self.view.select_box(
                    "Left Column:",
                    options=left_columns,
                    default=left_col if left_col in left_columns else "",
                    key=f"left_col_{i}"
                )
                
                # Select right column
                right_col = self.view.select_box(
                    "Right Column:",
                    options=right_columns,
                    default=right_col if right_col in right_columns else "",
                    key=f"right_col_{i}"
                )
                
                # Add remove button
                if self.view.display_button("Remove", key=f"remove_join_{i}"):
                    removed_indices.append(i)
                    # Don't add this pair to new_joining_fields
                    continue
                
                # Add to new joining fields if both columns selected
                if left_col and right_col:
                    new_joining_fields.append([left_col, right_col])
            
            # If any fields were removed, update the session and rerun
            if removed_indices:
                self.session.set('current_joining_fields', new_joining_fields)
                self.view.rerun_script()
                return False
            
            # Add button to add new joining field
            if self.view.display_button("+ Add Join Condition", key="add_join_condition"):
                new_joining_fields.append(["", ""])
                self.session.set('current_joining_fields', new_joining_fields)
                self.view.rerun_script()
                return False
            
            # Update joining fields in session
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
                    joined_df = self._perform_join(
                        left_df, 
                        right_df, 
                        left_joining_fields, 
                        right_joining_fields, 
                        join_type
                    )
                    
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
                    joined_df = self._perform_join(
                        left_df, 
                        right_df, 
                        left_joining_fields, 
                        right_joining_fields, 
                        join_type
                    )
                    
                    # Create new table name
                    new_table_name = f"joined_{left_table}_{right_table}"
                    
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
            
    def _perform_join(self, left_df: pd.DataFrame, right_df: pd.DataFrame, 
                     left_on: List[str], right_on: List[str], join_type: str) -> pd.DataFrame:
        """Perform the join operation between two tables"""
        try:
            # Create copies of the dataframes to avoid modifying the originals
            left_df_copy = left_df.copy()
            right_df_copy = right_df.copy()
            
            # Get list of common columns that aren't part of the join keys
            left_cols = set(left_df_copy.columns)
            right_cols = set(right_df_copy.columns)
            common_cols = left_cols.intersection(right_cols)
            
            # Remove join columns from the common columns list if they have the same name
            for l, r in zip(left_on, right_on):
                if l == r and l in common_cols:
                    common_cols.remove(l)
            
            # Rename all common columns in the right dataframe with a unique suffix
            # This prevents any potential conflicts with existing _right or _right_right columns
            if common_cols:
                # Create a unique suffix based on the right table name or a random string
                unique_suffix = f"_r_{str(uuid.uuid4())[:8]}"
                
                # Create a mapping of original column names to renamed column names
                rename_map = {col: f"{col}{unique_suffix}" for col in common_cols}
                
                # Rename columns in the right dataframe
                right_df_copy = right_df_copy.rename(columns=rename_map)
            
            # Perform the join with empty suffixes since we've already handled duplicates
            merged_df = pd.merge(
                left_df_copy, 
                right_df_copy, 
                left_on=left_on, 
                right_on=right_on, 
                how=join_type,
                suffixes=('', '')  # No suffixes needed as we've already renamed
            )
            
            return merged_df
        except Exception as e:
            logger.error(f"Error in _perform_join: {str(e)}")
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
            
            # Create state summary
            self._create_state_summary(final_table_name, final_table, join_history)
            
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
                self.session.set('final_table_name', final_joined_table_name)
                self.session.set('final_table', final_table)
                self.session.set('next_state', next_state)
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error showing final summary: {str(e)}")
            self.view.show_message(f"Error showing final summary: {str(e)}", "error")
            return False
    
    def _create_state_summary(self, final_table_name: str, final_table: pd.DataFrame, join_history: List[Dict[str, Any]]) -> None:
        """Create a summary of the join state for display in the next state"""
        try:
            # Create a standardized final table name based on session ID to avoid long names
            session_id = self.session.get('session_id')
            short_session = session_id.split('-')[0] if session_id else 'joined'
            final_joined_table_name = f"joined_table_{short_session}"
            
            summary = f"### Step 4: Table Joins\n\n"
            
            # Add join history
            summary += f"**Join Operations:** {len(join_history)}\n\n"
            
            for i, join in enumerate(join_history):
                summary += f"**Join {i+1}:** {join['left_table']} {join['join_type']} join {join['right_table']}\n"
                summary += f"- Join Keys: "
                
                join_keys = []
                for field_pair in join['joining_fields']:
                    join_keys.append(f"{field_pair[0]} = {field_pair[1]}")
                
                summary += ", ".join(join_keys) + "\n"
            
            summary += "\n"
            
            # Add final table info with cleaned name
            summary += f"**Final Table:** {final_joined_table_name}\n"
            summary += f"**Final Shape:** {final_table.shape[0]} rows × {final_table.shape[1]} columns\n\n"
            
            # Store the summary in the session
            self.session.set("step_4_summary", summary)
            self.session.set("final_joined_table_name", final_joined_table_name)
            
            # Also save to database
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
                        (summary, final_joined_table_name, json.dumps(join_history), session_id)
                    )
                else:
                    # Insert new summary
                    cursor.execute(
                        "INSERT INTO join_summary (session_id, summary, final_table_name, join_history) VALUES (?, ?, ?, ?)",
                        (session_id, summary, final_joined_table_name, json.dumps(join_history))
                    )
                
                # Save the final joined table to a dedicated table in the database
                try:
                    # Create the final joined table in the database
                    final_table.to_sql(final_joined_table_name, conn, if_exists='replace', index=False)
                    print(f">>> Saved final joined table as {final_joined_table_name}")
                except Exception as e:
                    print(f">>> Error saving final joined table: {str(e)}")
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error creating state summary: {str(e)}")
            
    def _show_state_summary(self) -> None:
        """Show state summary when already complete"""
        summary = self.session.get("step_4_summary", "Join process complete.")
        self.view.show_message(summary, "success")

    def _handle_single_table_case(self) -> bool:
        """Handle the case where only one table is uploaded"""
        try:
            # Get the single table name and data
            tables_data = self.session.get('tables_data', {})
            table_name = list(tables_data.keys())[0]
            table_data = tables_data[table_name]
            
            # Get the aggregation summary to find the correct table name
            session_id = self.session.get('session_id')
            aggregation_summary = self._fetch_aggregation_table_name(session_id, table_name)
            
            # Use the aggregated table name from the summary if available
            if aggregation_summary and 'aggregated_table_name' in aggregation_summary:
                final_table_name = aggregation_summary['aggregated_table_name']
            else:
                # Create a standardized final table name based on session ID
                short_session = session_id.split('-')[0] if session_id else 'single'
                final_table_name = f"single_table_{short_session}"
            
            # Display header
            self.view.display_subheader("Join Process")
            
            # Display message about single table
            self.view.display_markdown("**Single Table Detected**")
            self.view.display_markdown("Only one table was uploaded, so no join operations are needed.")
            self.view.display_markdown("The workflow will proceed with the existing table.")
            
            # Display table info
            self.view.display_markdown("---")
            self.view.display_markdown("**Table Information:**")
            self.view.display_markdown(f"**Name:** {table_name}")
            self.view.display_markdown(f"**Final Table Name:** {final_table_name}")
            self.view.display_markdown(f"**Shape:** {table_data.shape[0]} rows × {table_data.shape[1]} columns")
            self.view.display_markdown("**Preview:**")
            self.view.display_dataframe(table_data.head(5))
            
            # Create a simplified join history
            join_history = [{
                "message": "No join operations needed - single table workflow",
                "table_name": table_name,
                "final_table_name": final_table_name
            }]
            
            # Create state summary
            self._create_single_table_summary(table_name, final_table_name, table_data)
            
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
                self.session.set('final_table_name', final_table_name)
                self.session.set('final_table', table_data)
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