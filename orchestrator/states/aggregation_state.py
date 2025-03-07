from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from sfn_blueprint import Task
from aggregation_agent.agents.aggregation_agent import SFNAggregationAgent
# from orchestrator.agents.temp_aggregation_agent import SFNAggregationAgent
from orchestrator.utils.data_type_utils import DataTypeUtils
from orchestrator.storage.db_connector import DatabaseConnector
from sfn_blueprint.agents.validate_and_retry_agent import SFNValidateAndRetryAgent
from orchestrator.config.model_config import DEFAULT_LLM_PROVIDER
import logging
import json
import sqlite3

logger = logging.getLogger(__name__)

class AggregationState:
    """
    State for handling data aggregation based on the problem type and mapping information.
    This state determines if aggregation is needed and applies appropriate aggregation methods.
    """
    
    def __init__(self, session, view):
        self.name = "Aggregation"
        self.session = session
        self.view = view
        self.db = DatabaseConnector()
        
    def execute(self) -> bool:
        """
        Execute the aggregation state logic.
        
        Returns:
            bool: True if the state execution is complete, False otherwise
        """
        logger.info("Executing Aggregation State")
        
        # Get session ID
        session_id = self.session.get('session_id')
        if not session_id:
            self.view.show_message("Session ID is missing. Please restart the application.", "error")
            return False
        
        # Get mapping summary from database
        mapping_summary = self._get_mapping_summary_from_db(session_id)
        if not mapping_summary:
            self.view.show_message("Mapping information is missing. Please complete the mapping step first.", "error")
            return False
        print(">>mapping_summary", mapping_summary)
        
        # Store mapping summary in session for convenience
        self.session.set("mapping_summary", mapping_summary)
        
        # Get problem type from the confirmed_mapping dictionary
        problem_type = mapping_summary.get("confirmed_mapping", {}).get("problem_type", "").lower()
        print(">>problem_type", problem_type)
        
        # Get all tables from the tables_mapped array
        tables_mapped = mapping_summary.get("confirmed_mapping", {}).get("tables_mapped", [])
        if not tables_mapped:
            self.view.show_message("No tables mapped in the mapping summary.", "error")
            return False
        
        # Store the list of tables in the session
        self.session.set("tables_mapped", tables_mapped)
        
        # Check if all tables have been processed
        if self.session.get("all_tables_processed"):
            # Show final summary and proceed button
            return self._render_final_summary()
        
        # Get current table index (default to 0 if not set)
        current_table_idx = self.session.get("current_table_idx", 0)
        
        # If we've processed all tables, mark as complete and show summary
        if current_table_idx >= len(tables_mapped):
            self.session.set("all_tables_processed", True)
            return self._render_final_summary()
        
        # Process current table
        current_table = tables_mapped[current_table_idx]
        
        # If this table is already processed, move to next
        if self.session.get(f"table_{current_table}_processed"):
            self.session.set("current_table_idx", current_table_idx + 1)
            self.view.rerun_script()
            return False
        
        # Process the current table
        return self._process_single_table(current_table, problem_type)
    
    def _process_single_table(self, table_name: str, problem_type: str) -> bool:
        """
        Process a single table for aggregation
        
        Args:
            table_name: The name of the table to process
            problem_type: The problem type
            
        Returns:
            bool: True if processing is complete, False otherwise
        """
        # try:
        # Fetch table data from database
        session_id = self.session.get('session_id')
        print(f">>> Processing table: {table_name} for problem type: {problem_type}")
        table_data = self._get_table_data_from_db(session_id, table_name)
        
        if table_data is None or table_data.empty:
            print(f">>> Table data is missing for {table_name}")
            self.view.show_message(f"Data for table {table_name} is missing. Skipping.", "warning")
            # Mark this table as processed and move to the next one
            self.session.set(f"table_{table_name}_processed", True)
            current_table_idx = self.session.get("current_table_idx", 0)
            self.session.set("current_table_idx", current_table_idx + 1)
            self.view.rerun_script()
            return False
        
        print(f">>> Table data shape: {table_data.shape}")
        print(f">>> Table data columns: {table_data.columns.tolist()}")
        
        # Store the current table data in the session
        self.session.set("current_table", table_name)
        self.session.set("data", table_data)
        
        # Check if aggregation is already complete for this table
        if self.session.get(f"aggregation_complete_{table_name}"):
            # Show the aggregation results
            self._show_aggregation_results(table_name)
            
            # Mark this table as processed and move to the next one
            self.session.set(f"table_{table_name}_processed", True)
            current_table_idx = self.session.get("current_table_idx", 0)
            self.session.set("current_table_idx", current_table_idx + 1)
            self.view.rerun_script()
            return False
        
        # Initialize aggregation for this table if not already done
        if not self.session.get(f"aggregation_initialized_{table_name}"):
            # Skip aggregation for recommendation and forecasting problem types
            if problem_type in ["recommendation", "forecasting"]:
                print(f">>> Skipping aggregation for problem type: {problem_type}")
                self.session.set(f"aggregation_needed_{table_name}", False)
                self.session.set(f"aggregation_complete_{table_name}", True)
                self.session.set(f"aggregated_data_{table_name}", table_data)
                self.session.set(f"aggregation_initialized_{table_name}", True)
                
                # Save aggregation summary to database
                self._save_aggregation_summary_to_db(session_id, {
                    "aggregation_needed": False,
                    "problem_type": problem_type,
                    "table_name": table_name,
                    "message": f"Aggregation skipped for {problem_type} problem type"
                })
                
                # Show the aggregation results
                self._show_aggregation_results(table_name)
                
                # Mark this table as processed and move to the next one
                self.session.set(f"table_{table_name}_processed", True)
                current_table_idx = self.session.get("current_table_idx", 0)
                self.session.set("current_table_idx", current_table_idx + 1)
                self.view.rerun_script()
                return False
            
            # For classification, regression, and clustering, check if aggregation is needed
            if problem_type in ["classification", "regression", "clustering"]:
                # Get mapping columns
                mapping_columns = self._get_mapping_columns(self.session.get("mapping_summary"), problem_type, table_name)
                print(f">>> Mapping columns for {table_name}: {mapping_columns}")
                
                # Get aggregation suggestions using the aggregation agent
                with self.view.display_spinner(f'🤖 AI is analyzing aggregation needs for {table_name}...'):
                    aggregation_needed, aggregation_suggestions = self._get_aggregation_suggestions(
                        table_data, mapping_columns
                    )
                
                print(f">>> Aggregation needed for {table_name}: {aggregation_needed}")
                print(f">>> Aggregation suggestions: {aggregation_suggestions}")
                
                self.session.set(f"aggregation_needed_{table_name}", aggregation_needed)
                self.session.set(f"aggregation_suggestions_{table_name}", aggregation_suggestions)
                self.session.set(f"aggregation_initialized_{table_name}", True)
                
                # If aggregation is not needed, mark as complete and move to next table
                if not aggregation_needed:
                    self.session.set(f"aggregation_complete_{table_name}", True)
                    self.session.set(f"aggregated_data_{table_name}", table_data)
                    
                    # Save aggregation summary to database
                    self._save_aggregation_summary_to_db(session_id, {
                        "aggregation_needed": False,
                        "problem_type": problem_type,
                        "table_name": table_name,
                        "message": "No aggregation needed - data is already at the desired granularity"
                    })
                    
                    # Show the aggregation results
                    self._show_aggregation_results(table_name)
                    
                    # Mark this table as processed and move to the next one
                    self.session.set(f"table_{table_name}_processed", True)
                    current_table_idx = self.session.get("current_table_idx", 0)
                    self.session.set("current_table_idx", current_table_idx + 1)
                    self.view.rerun_script()
                    return False
            else:
                # Unknown problem type, skip aggregation
                print(f">>> Unknown problem type: {problem_type}")
                logger.warning(f"Unknown problem type: {problem_type}")
                self.view.show_message(f"Unknown problem type: {problem_type}. Skipping aggregation.", "warning")
                self.session.set(f"aggregation_needed_{table_name}", False)
                self.session.set(f"aggregation_complete_{table_name}", True)
                self.session.set(f"aggregated_data_{table_name}", table_data)
                self.session.set(f"aggregation_initialized_{table_name}", True)
                
                # Save aggregation summary to database
                self._save_aggregation_summary_to_db(session_id, {
                    "aggregation_needed": False,
                    "problem_type": problem_type,
                    "table_name": table_name,
                    "message": f"Aggregation skipped for unknown problem type: {problem_type}"
                })
                
                # Show the aggregation results
                self._show_aggregation_results(table_name)
                
                # Mark this table as processed and move to the next one
                self.session.set(f"table_{table_name}_processed", True)
                current_table_idx = self.session.get("current_table_idx", 0)
                self.session.set("current_table_idx", current_table_idx + 1)
                self.view.rerun_script()
                return False
        
        # If we get here, aggregation is needed for the current table but not yet complete
        return self._render_aggregation_ui(table_name)
            
        # except Exception as e:
        #     print(f">>> Error processing table {table_name}: {str(e)}")
        #     self.view.show_message(f"Error processing table {table_name}: {str(e)}", "error")
        #     return False
    
    def _get_mapping_summary_from_db(self, session_id: str) -> Dict:
        """
        Fetch mapping summary from the database
        
        Args:
            session_id: The session ID
            
        Returns:
            Dict containing the mapping summary
        """
        try:
            # Connect to the database
            with sqlite3.connect(self.db.db_path) as conn:
                # Convert rows to dictionaries
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Query the mappings_summary table
                cursor.execute(
                    "SELECT * FROM mappings_summary WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
                    (session_id,)
                )
                mapping_record = cursor.fetchone()
                
                if not mapping_record:
                    logger.warning(f"No mapping summary found for session {session_id}")
                    return {}
                
                # Parse the mappings JSON field
                mappings = json.loads(mapping_record['mappings']) if mapping_record['mappings'] else {}
                
                # Create a complete mapping summary
                mapping_summary = {
                    "table_name": mapping_record['table_name'],
                    "confirmed_mapping": mappings,
                    "created_at": mapping_record['created_at']
                }
                
                # Check if there's a _state_summary field with additional info
                if '_state_summary' in mappings:
                    state_summary = mappings['_state_summary']
                    # Add state summary fields to the mapping summary
                    for key, value in state_summary.items():
                        if key not in mapping_summary:
                            mapping_summary[key] = value
                print(">>> mapping_summary in _get_mapping_summary_from_db", mapping_summary)
                return mapping_summary
        except Exception as e:
            logger.error(f"Error fetching mapping summary: {str(e)}")
            return {}
    
    def _save_aggregation_summary_to_db(self, session_id: str, summary: Dict) -> bool:
        """
        Save aggregation summary to the database with enhanced information
        
        Args:
            session_id: The session ID
            summary: The aggregation summary to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                
                # Create a table for aggregation summary if it doesn't exist
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS aggregation_summary (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        table_name TEXT,
                        aggregation_needed BOOLEAN,
                        aggregation_methods TEXT,
                        aggregated_table_shape TEXT,
                        aggregated_table_name TEXT,
                        summary TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Determine the aggregated table name
                base_table_name = summary.get("table_name", "").split('.')[0] if '.' in summary.get("table_name", "") else summary.get("table_name", "")
                
                # If aggregation is needed, use the aggregated table name
                # If not needed, use the original onboarding table name
                if summary.get("aggregation_needed", False):
                    aggregated_table_name = f"aggregated_table_{base_table_name.lower().replace('-', '_')}"
                else:
                    # Use the original onboarding table
                    aggregated_table_name = f"onboarding_table_{base_table_name.lower().replace('-', '_')}"
                
                # Insert the aggregation summary
                cursor.execute(
                    """
                    INSERT INTO aggregation_summary 
                    (session_id, table_name, aggregation_needed, aggregation_methods, aggregated_table_shape, aggregated_table_name, summary)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        summary.get("table_name", ""),
                        summary.get("aggregation_needed", False),
                        json.dumps(summary.get("aggregation_methods", {})),
                        json.dumps(summary.get("aggregated_table_shape", [])),
                        aggregated_table_name,
                        summary.get("message", "")
                    )
                )
                
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving aggregation summary: {str(e)}")
            return False
                
    def _get_mapping_columns(self, mapping_summary: Dict, problem_type: str, current_table: str) -> Dict:
        """
        Get mapping columns based on problem type, prediction level, and time series flag
        
        Args:
            mapping_summary: The mapping summary
            problem_type: The problem type
            current_table: The current table being processed
            
        Returns:
            Dict mapping column types to column names in the format expected by the aggregation agent
        """
        mapping_columns = {}
        print(f">>> Getting mapping columns for table: {current_table}")
        
        # Get confirmed mapping from the summary
        confirmed_mapping = mapping_summary.get("confirmed_mapping", {})
        
        # Get prediction level and time series flag
        prediction_level = self.session.get('prediction_level', 'ID Level')
        is_time_series = self.session.get('is_time_series', False)
        
        print(f">>> Prediction level: {prediction_level}")
        print(f">>> Is time series: {is_time_series}")
        
        # We need to fetch the actual column mappings from the database
        # The mapping_summary only contains metadata, not the actual mappings
        session_id = self.session.get('session_id')
        
        try:
            # Connect to the database
            with sqlite3.connect(self.db.db_path) as conn:
                # Convert rows to dictionaries
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Query the mappings_summary table for this specific table only
                print(f">>> Querying mappings for table: {current_table}")
                cursor.execute(
                    "SELECT * FROM mappings_summary WHERE session_id = ? AND table_name = ? ORDER BY created_at DESC LIMIT 1",
                    (session_id, current_table)
                )
                table_mapping_record = cursor.fetchone()
                
                if not table_mapping_record:
                    print(f">>> No mapping record found for table: {current_table}")
                    return mapping_columns
                
                # Parse the mappings JSON field
                table_mappings = json.loads(table_mapping_record['mappings']) if table_mapping_record['mappings'] else {}
                print(f">>> Table mappings for {current_table}: {table_mappings}")
                
                # Always include ID field (required for all scenarios)
                if 'id' in table_mappings and table_mappings['id']:
                    id_field = table_mappings['id']
                    mapping_columns['customer_id'] = id_field
                    print(f">>> Found customer_id mapping: id -> {id_field}")
                
                # Include timestamp field if it's time series data
                if is_time_series:
                    if 'timestamp' in table_mappings and table_mappings['timestamp']:
                        date_field = table_mappings['timestamp']
                        mapping_columns['date'] = date_field
                        print(f">>> Found date mapping: timestamp -> {date_field}")
                
                # Include product_id field if prediction level is ID+Product
                if prediction_level == 'ID+Product Level':
                    if 'product_id' in table_mappings and table_mappings['product_id']:
                        product_field = table_mappings['product_id']
                        mapping_columns['product_id'] = product_field
                        print(f">>> Found product_id mapping: product_id -> {product_field}")
        
        except Exception as e:
            print(f">>> Error fetching mapping columns: {str(e)}")
            logger.error(f"Error fetching mapping columns: {str(e)}")
        
        # If we still don't have a customer_id mapping, try to use the first column as customer_id
        if 'customer_id' not in mapping_columns and 'data' in self.session:
            df = self.session.get('data')
            if isinstance(df, pd.DataFrame) and not df.empty:
                first_col = df.columns[0]
                mapping_columns['customer_id'] = first_col
                print(f">>> Using first column as customer_id: {first_col}")
        
        print(f">>> Aggregation will be on mappings {current_table}: {mapping_columns}")
        return mapping_columns
    
    def _get_aggregation_suggestions(self, df: pd.DataFrame, mapping_columns: Dict) -> Tuple[bool, Dict]:
        """
        Get aggregation suggestions from the aggregation agent
        
        Args:
            df: The dataframe to analyze
            mapping_columns: Dictionary mapping column types to column names
            
        Returns:
            Tuple of (aggregation_needed, aggregation_suggestions)
        """
        # If no mapping columns, we can't determine if aggregation is needed
        if not mapping_columns:
            print(">>> No mapping columns found, cannot determine if aggregation is needed")
            logger.warning("No mapping columns found, cannot determine if aggregation is needed")
            return False, {}
        
        # Print mapping columns for debugging
        print(f">>> Mapping columns for aggregation: {mapping_columns}")
        
        print(">>> Creating aggregation agent...")
        # Create an instance of the aggregation agent
        aggregation_agent = SFNAggregationAgent(llm_provider=DEFAULT_LLM_PROVIDER)
        
        # Prepare the task data
        task_data = {
            'table': df,
            'mapping_columns': mapping_columns
        }
        
        print(">>> Preparing task for aggregation agent...")
        # Create a task for the agent
        agg_task = Task("Analyze aggregation", data=task_data)
        
        print(">>> Executing aggregation agent task...")
        # Execute the aggregation agent - this will internally check if aggregation is needed
        result = aggregation_agent.execute_task(agg_task)
        print(f">>> Aggregation agent result: {result}")
        
        # Check if the result indicates no aggregation is needed
        if isinstance(result, dict) and result.get("__no_aggregation_needed__", False):
            print(">>> Agent determined no aggregation is needed")
            return False, {}
        
        # If we got here, check if we have actual suggestions
        if isinstance(result, dict) and len(result) > 0:
            print(">>> Aggregation is needed, returning suggestions")
            return True, result
        else:
            print(">>> No valid aggregation suggestions returned")
            return False, {}
    
    def _apply_aggregation(self, table_name: str, selected_methods: Dict[str, List]) -> Dict[str, Any]:
        """
        Apply the selected aggregation methods to the table and save to database
        
        Args:
            table_name: The name of the table to aggregate
            selected_methods: Dictionary mapping feature names to lists of selected methods
            
        Returns:
            Dict with status and message
        """
        try:
            print(f">>> Applying aggregation to {table_name} with methods: {selected_methods}")
            
            # Get the table data
            table_data = self.session.get("data")
            
            # Get mapping columns
            mapping_summary = self.session.get("mapping_summary", {})
            problem_type = mapping_summary.get("confirmed_mapping", {}).get("problem_type", "").lower()
            mapping_columns = self._get_mapping_columns(mapping_summary, problem_type, table_name)
            
            # Get the groupby columns
            groupby_cols = list(mapping_columns.values())
            print(f">>> Groupby columns: {groupby_cols}")
            
            if not groupby_cols:
                return {
                    "success": False,
                    "error": "No groupby columns found"
                }
            
            # Create a dictionary to map pandas aggregation functions with proper naming
            agg_function_map = {
                'Min': 'min',
                'Max': 'max',
                'Sum': 'sum',
                'Mean': 'mean',
                'Median': 'median',
                'Mode': 'mode',  # We'll handle this specially
                'Unique Count': 'nunique',  # Use pandas nunique directly
                'Last Value': 'last'  # We'll handle this specially
            }
            
            # Create the aggregation dictionary for pandas
            agg_dict = {}
            for feature, methods in selected_methods.items():
                for method in methods:
                    # Get the pandas function name
                    pandas_func = agg_function_map.get(method, method.lower())
                    
                    # Create a column name for this aggregation
                    col_name = f"{feature}_{method.lower().replace(' ', '_')}"
                    
                    # Add to aggregation dictionary with explicit naming
                    if method == 'Mode':
                        # Handle mode specially
                        agg_dict[col_name] = pd.NamedAgg(column=feature, aggfunc=lambda x: x.mode().iloc[0] if not x.mode().empty else None)
                    elif method == 'Last Value':
                        # Handle last value specially
                        agg_dict[col_name] = pd.NamedAgg(column=feature, aggfunc=lambda x: x.iloc[-1] if len(x) > 0 else None)
                    else:
                        # Standard aggregation
                        agg_dict[col_name] = pd.NamedAgg(column=feature, aggfunc=pandas_func)
            
            print(f">>> Aggregation dictionary: {agg_dict}")
            
            # Group by the specified columns and apply the aggregation
            grouped = table_data.groupby(groupby_cols)
            
            # Apply the aggregation methods with named aggregations
            aggregated_data = grouped.agg(**agg_dict)
            
            # Reset the index to make the groupby columns regular columns again
            aggregated_data = aggregated_data.reset_index()
            
            # Store the aggregated data in the session
            self.session.set(f"aggregated_data_{table_name}", aggregated_data)
            
            # Store the confirmed aggregation methods
            self.session.set(f"confirmed_aggregation_{table_name}", selected_methods)
            
            # Save the aggregated table to the database
            session_id = self.session.get('session_id')
            
            # Create a standardized table name without extension
            base_table_name = table_name.split('.')[0] if '.' in table_name else table_name
            aggregated_table_name = f"aggregated_table_{base_table_name.lower().replace('-', '_')}"
            
            # Save to database
            try:
                with sqlite3.connect(self.db.db_path) as conn:
                    # Add session_id column to the dataframe
                    # aggregated_data['session_id'] = session_id
                    
                    # Save to database
                    aggregated_data.to_sql(aggregated_table_name, conn, if_exists='replace', index=False)
                    print(f">>> Saved aggregated table to database: {aggregated_table_name}")
            except Exception as e:
                print(f">>> Error saving aggregated table to database: {str(e)}")
                # Continue even if saving fails - we still have the data in the session
            
            # Return success
            return {
                "success": True,
                "message": f"Successfully aggregated {table_name}",
                "shape": aggregated_data.shape,
                "aggregated_table_name": aggregated_table_name
            }
        except Exception as e:
            print(f">>> Error applying aggregation: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_state_summary(self):
        """Create a summary of the aggregation state for display in the next state"""
        tables_mapped = self.session.get("tables_mapped", [])
        
        summary = f"### Step 3: Data Aggregation\n\n"
        
        for table_name in tables_mapped:
            aggregation_needed = self.session.get(f"aggregation_needed_{table_name}", False)
            
            if not aggregation_needed:
                summary += f"✅ No aggregation needed for {table_name} - data is already at the desired granularity.\n\n"
            else:
                aggregated_data = self.session.get(f"aggregated_data_{table_name}")
                shape = aggregated_data.shape if isinstance(aggregated_data, pd.DataFrame) else None
                
                summary += f"✅ Aggregation performed on {table_name}\n\n"
                
                if shape:
                    summary += f"**Shape after aggregation:** {shape[0]} rows × {shape[1]} columns\n\n"
                
                # Add aggregation methods
                aggregation_methods = self.session.get(f"confirmed_aggregation_{table_name}", {})
                if aggregation_methods:
                    summary += "**Applied aggregation methods:**\n\n"
                    for feature, methods in aggregation_methods.items():
                        # Clean up method names for display
                        clean_methods = []
                        for method in methods:
                            if callable(method):
                                method_name = method.__name__
                            else:
                                method_name = 'unique count' if method == 'nunique' else method
                            clean_methods.append(method_name)
                        
                        methods_str = ', '.join(clean_methods)
                        summary += f"- {feature}: {methods_str}\n"
                
                summary += "\n---\n\n"
        
        # Store the summary in the session
        self.session.set("step_3_summary", summary)
        
        # Also save to database
        session_id = self.session.get('session_id')
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.cursor()
            
            # Create aggregation_summary table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS aggregation_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Check if a summary already exists
            cursor.execute(
                "SELECT id FROM aggregation_summary WHERE session_id = ?",
                (session_id,)
            )
            existing = cursor.fetchone()
            
            if existing:
                # Update existing summary
                cursor.execute(
                    "UPDATE aggregation_summary SET summary = ? WHERE session_id = ?",
                    (summary, session_id)
                )
            else:
                # Insert new summary
                cursor.execute(
                    "INSERT INTO aggregation_summary (session_id, summary) VALUES (?, ?)",
                    (session_id, summary)
                )
            
            conn.commit()
        
    def _show_state_summary(self):
        """Display the state summary"""
        summary = self.session.get("step_3_summary")
        if summary:
            self.view.display_markdown(summary)
    
    def _get_table_data_from_db(self, session_id: str, table_name: str) -> Optional[pd.DataFrame]:
        """
        Fetch table data from the database using the correct table name format
        
        Args:
            session_id: The session ID
            table_name: The table name from the mapping summary (includes extension)
            
        Returns:
            DataFrame containing the table data or None if not found
        """
        try:
            # Remove file extension if present
            base_table_name = table_name.split('.')[0] if '.' in table_name else table_name
            
            # Format the table name according to the database convention
            db_table_name = f"onboarding_table_{base_table_name.lower().replace('-', '_')}"
            
            # Connect to the database
            with sqlite3.connect(self.db.db_path) as conn:
                # Try to fetch the data
                query = f"SELECT * FROM {db_table_name} WHERE session_id = ?"
                df = pd.read_sql(query, conn, params=(session_id,))
                
                if df.empty:
                    logger.warning(f"No data found for table {db_table_name}")
                    
                    # Try alternative naming convention
                    alt_table_name = f"onboarding_{base_table_name.lower().replace('-', '_')}"
                    query = f"SELECT * FROM {alt_table_name} WHERE session_id = ?"
                    df = pd.read_sql(query, conn, params=(session_id,))
                    
                    if df.empty:
                        logger.warning(f"No data found for alternative table {alt_table_name}")
                        return None
                
                return df
        except Exception as e:
            logger.error(f"Error fetching table data: {str(e)}")
            
            # Try using the DatabaseConnector's method as a fallback
            try:
                return self.db.fetch_table("onboarding", table_name, session_id)
            except Exception as inner_e:
                logger.error(f"Error using fallback method: {str(inner_e)}")
                return None
    
    def _render_aggregation_ui(self, table_name: str) -> bool:
        """
        Render the UI for selecting aggregation methods for a specific table
        
        Args:
            table_name: The name of the table to render UI for
            
        Returns:
            bool: True if the state execution is complete, False otherwise
        """
        # Display header
        self.view.display_header(f"Step 3: Data Aggregation - {table_name}")
        
        self.view.display_markdown("This dataset requires aggregation to ensure each entity has a single row.")
        
        # Get aggregation suggestions for this table
        aggregation_suggestions = self.session.get(f"aggregation_suggestions_{table_name}", {})
        
        # Get table data
        table_data = self.session.get("data")
        
        # Get mapping columns
        mapping_summary = self.session.get("mapping_summary", {})
        problem_type = mapping_summary.get("confirmed_mapping", {}).get("problem_type", "").lower()
        mapping_columns = self._get_mapping_columns(mapping_summary, problem_type, table_name)
        
        # Display aggregation selection interface
        self.view.display_subheader("Select Aggregation Methods")
        self.view.display_markdown("Choose how to aggregate each feature when multiple rows exist for the same entity:")
        
        # Get all column info excluding groupby columns
        groupby_cols = list(mapping_columns.values())
        column_info = DataTypeUtils.get_column_info(table_data, exclude_columns=groupby_cols)
        
        # Create DataFrame for aggregation methods
        method_names = ['Min', 'Max', 'Sum', 'Unique Count', 'Mean', 'Median', 'Mode', 'Last Value']
        agg_rows = []
        explanations_dict = {}
        
        # Process all columns
        for feature, info in column_info.items():
            row = {'Feature': feature}
            # Ensure allowed_methods is a list of strings matching method_names
            allowed_methods = [m for m in method_names if m.lower() in [am.lower() for am in info.get('allowed_methods', [])]]
            
            # Get suggestions for this feature if they exist
            feature_suggestions = aggregation_suggestions.get(feature, [])
            # Normalize method names to match method_names
            suggested_methods = []
            for suggestion in feature_suggestions:
                method = suggestion.get('method', '')
                # Find the matching method name in method_names (case insensitive)
                for m in method_names:
                    if m.lower() == method.lower():
                        suggested_methods.append(m)
                        break
            
            # Store explanations if they exist
            if feature_suggestions:
                explanations_dict[feature] = {}
                for suggestion in feature_suggestions:
                    method = suggestion.get('method', '')
                    explanation = suggestion.get('explanation', '')
                    # Find the matching method name in method_names (case insensitive)
                    for m in method_names:
                        if m.lower() == method.lower():
                            explanations_dict[feature][m] = explanation
                            break
            
            # For each method, determine if it should be enabled and/or pre-ticked
            for method in method_names:
                row[method] = {
                    'enabled': method in allowed_methods,
                    'checked': method in suggested_methods
                }
            
            agg_rows.append(row)
        
        # Create columns for the header
        col_feature, *method_cols = self.view.create_columns([2] + [1]*8)
        
        # Header row
        col_feature.markdown("**Feature**")
        for col, method in zip(method_cols, method_names):
            col.markdown(f"**{method}**")
        
        # Feature rows with checkboxes
        selected_methods = {}
        for row in agg_rows:
            feature = row['Feature']
            row_cols = self.view.create_columns([2] + [1]*8)
            
            # Feature name with data type
            dtype = column_info[feature].get('dtype', 'unknown')
            row_cols[0].markdown(f"**{feature}** ({dtype})")
            
            # Checkboxes for each method
            selected_methods[feature] = []
            for i, method in enumerate(method_names):
                method_info = row[method]  # Get method info from row
                checkbox_key = f"{table_name}_{feature}_{method}"
                
                if method_info['enabled']:
                    if row_cols[i+1].checkbox(
                        label=f"Select {method} for {feature}",
                        key=checkbox_key,
                        value=method_info['checked'],
                        label_visibility="collapsed"
                    ):
                        selected_methods[feature].append(method)
                else:
                    # Use a disabled checkbox instead of the red cross
                    row_cols[i+1].checkbox(
                        label=f"{method} for {feature} (disabled)",
                        key=f"{checkbox_key}_disabled",
                        value=False,
                        disabled=True,
                        label_visibility="collapsed"
                    )
        
        # Explanations section
        if explanations_dict:
            with self.view.create_expander("Show Aggregation Method Explanations"):
                for feature, explanations in explanations_dict.items():
                    self.view.display_subheader(feature)
                    for method, explanation in explanations.items():
                        self.view.display_markdown(f"• **{method}**: {explanation}")
        
        # Confirm button
        if self.view.display_button(f"Apply Aggregation Methods for {table_name}"):
            # Filter out features with no selected methods
            selected_methods = {k: v for k, v in selected_methods.items() if v}
            
            if not selected_methods:
                self.view.show_message("Please select at least one aggregation method.", "warning")
                return False
            
            # Apply the selected aggregation methods
            aggregation_result = self._apply_aggregation(table_name, selected_methods)
            
            if aggregation_result.get("success"):
                # Mark aggregation as complete for this table
                self.session.set(f"aggregation_complete_{table_name}", True)
                
                # Save aggregation summary to database
                session_id = self.session.get('session_id')
                self._save_aggregation_summary_to_db(session_id, {
                    "aggregation_needed": True,
                    "problem_type": problem_type,
                    "table_name": table_name,
                    "aggregation_methods": selected_methods,
                    "aggregated_table_shape": aggregation_result.get("shape", []),
                    "aggregated_table_name": aggregation_result.get("aggregated_table_name", "")
                })
                
                # Show the aggregation results
                self._show_aggregation_results(table_name)
                
                # Mark this table as processed and move to the next one
                self.session.set(f"table_{table_name}_processed", True)
                current_table_idx = self.session.get("current_table_idx", 0)
                self.session.set("current_table_idx", current_table_idx + 1)
                self.view.rerun_script()
                return False
            else:
                self.view.show_message(f"Error applying aggregation: {aggregation_result.get('error')}", "error")
        
        return False
    
    def _show_aggregation_results(self, table_name: str) -> None:
        """
        Show the results of aggregation for a specific table
        
        Args:
            table_name: The name of the table to show results for
        """
        aggregation_needed = self.session.get(f"aggregation_needed_{table_name}", False)
        
        self.view.display_header(f"Step 3: Data Aggregation - {table_name}")
        
        if not aggregation_needed:
            self.view.show_message(f"✅ No aggregation needed for {table_name} - data is already at the desired granularity.", "success")
        else:
            aggregated_data = self.session.get(f"aggregated_data_{table_name}")
            if isinstance(aggregated_data, pd.DataFrame):
                shape = aggregated_data.shape
                self.view.show_message(f"✅ Aggregation applied successfully to {table_name}", "success")
                self.view.display_markdown(f"**Shape after aggregation:** {shape[0]} rows × {shape[1]} columns")
            
            # Show aggregation methods
            aggregation_methods = self.session.get(f"confirmed_aggregation_{table_name}", {})
            if aggregation_methods:
                self.view.display_markdown("**Applied aggregation methods:**")
                
                # Create a table to display methods
                method_data = []
                for feature, methods in aggregation_methods.items():
                    # Clean up method names for display
                    clean_methods = []
                    for method in methods:
                        if callable(method):
                            method_name = method.__name__
                        else:
                            method_name = 'unique count' if method == 'nunique' else method
                        clean_methods.append(method_name)
                    
                    methods_str = ', '.join(clean_methods)
                    method_data.append({"Feature": feature, "Aggregation Methods": methods_str})
                
                self.view.display_table(method_data)
        
        # Show button to continue to next table
        self.view.display_markdown("---")
        if self.view.display_button("Continue to Next Table"):
            # Mark this table as processed and move to the next one
            self.session.set(f"table_{table_name}_processed", True)
            current_table_idx = self.session.get("current_table_idx", 0)
            self.session.set("current_table_idx", current_table_idx + 1)
            self.view.rerun_script()
    
    def _render_final_summary(self) -> bool:
        """
        Render the final summary of all tables processed with more detailed information
        
        Returns:
            bool: True to indicate completion of the state
        """
        # Display header
        self.view.display_header("Step 3: Data Aggregation - Summary")
        
        # Get all tables
        tables_mapped = self.session.get("tables_mapped", [])
        
        # Create summary for each table
        for table_name in tables_mapped:
            self.view.display_subheader(f"Table: {table_name}")
            
            aggregation_needed = self.session.get(f"aggregation_needed_{table_name}", False)
            
            if not aggregation_needed:
                self.view.display_markdown(f"✅ No aggregation needed - data is already at the desired granularity.")
                
                # Show original data shape
                original_data = self.session.get("data")
                if isinstance(original_data, pd.DataFrame):
                    self.view.display_markdown(f"**Original data shape:** {original_data.shape[0]} rows × {original_data.shape[1]} columns")
                    
                    # Show sample of columns
                    self.view.display_markdown(f"**Columns:** {', '.join(original_data.columns[:5])}{'...' if len(original_data.columns) > 5 else ''}")
            else:
                aggregated_data = self.session.get(f"aggregated_data_{table_name}")
                if isinstance(aggregated_data, pd.DataFrame):
                    # Get original data shape for comparison
                    original_data = self.session.get("data")
                    original_shape = original_data.shape if isinstance(original_data, pd.DataFrame) else None
                    
                    # Show shapes
                    self.view.display_markdown(f"✅ Aggregation applied")
                    if original_shape:
                        self.view.display_markdown(f"**Original shape:** {original_shape[0]} rows × {original_shape[1]} columns")
                    self.view.display_markdown(f"**Shape after aggregation:** {aggregated_data.shape[0]} rows × {aggregated_data.shape[1]} columns")
                    
                    # Show reduction percentage if applicable
                    if original_shape and original_shape[0] > 0:
                        reduction = ((original_shape[0] - aggregated_data.shape[0]) / original_shape[0]) * 100
                        self.view.display_markdown(f"**Row reduction:** {reduction:.1f}% ({original_shape[0] - aggregated_data.shape[0]} rows)")
                    
                    # Show sample of columns
                    self.view.display_markdown(f"**Columns after aggregation:** {', '.join(aggregated_data.columns[:5])}{'...' if len(aggregated_data.columns) > 5 else ''}")
                
                # Show aggregation methods
                aggregation_methods = self.session.get(f"confirmed_aggregation_{table_name}", {})
                if aggregation_methods:
                    self.view.display_markdown("**Applied aggregation methods:**")
                    
                    # Create a table to display methods
                    method_data = []
                    for feature, methods in aggregation_methods.items():
                        # Clean up method names for display
                        clean_methods = []
                        for method in methods:
                            if callable(method):
                                method_name = method.__name__
                            else:
                                method_name = 'unique count' if method == 'nunique' else method
                            clean_methods.append(method_name)
                        
                        methods_str = ', '.join(clean_methods)
                        method_data.append({"Feature": feature, "Aggregation Methods": methods_str})
                    
                    self.view.display_table(method_data)
        
        self.view.display_markdown("---")
        
        # Create overall summary for the next state
        self._create_state_summary()
        
        # Show button to proceed to next step
        if self.view.display_button("Proceed to Join State"):
            self.session.set("aggregation_complete", True)
            return True
            
        return False 