import sqlite3
from typing import Dict, List, Optional, Any
import pandas as pd
import json
from pathlib import Path
import os

class DatabaseConnector:
    """Handles database connections and operations"""
    
    def __init__(self, db_path: str = "storage/db/orchestrator.db"):
        # Get absolute path to project root
        project_root = Path(__file__).parent.parent.parent
        self.db_path = project_root / db_path
        self._ensure_db_path()
        self._init_db()
    
    def _ensure_db_path(self):
        """Ensure database directory exists"""
        db_dir = self.db_path.parent
        db_dir.mkdir(parents=True, exist_ok=True)
    
    def _init_db(self):
        """Initialize database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create table registry with more detailed information
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS table_registry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    table_name TEXT NOT NULL,      -- Actual table name in DB
                    original_name TEXT NOT NULL,   -- Original file name
                    row_count INTEGER NOT NULL,
                    column_count INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(session_id, original_name)
                )
            """)
            
            # Create other necessary tables - use consistent naming convention
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS onboarding_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    num_tables TEXT NOT NULL,
                    problem_type TEXT NOT NULL,
                    target_column TEXT,
                    has_target TEXT,
                    table_names TEXT,
                    table_names_clean TEXT,
                    table_rows TEXT,
                    table_columns TEXT,
                    completion_time TEXT,
                    is_time_series TEXT,
                    recommendation_approach TEXT
                )
            """)
            
            conn.commit()
    
    def save_uploaded_tables(self, session_id: str, tables: List[Dict]) -> bool:
        """Save uploaded tables to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Save each table's data
                for table in tables:
                    print(f"Processing table: {table['name']}")  # Debug log
                    
                    # Read the DataFrame
                    df = pd.read_csv(table['path']) if table['name'].endswith('.csv') \
                        else pd.read_excel(table['path'])
                    
                    # Create safe table name (remove extension and special characters)
                    safe_table_name = f"table_{table['name'].split('.')[0].lower().replace('-', '_')}"
                    db_table_name = f"onboarding_{safe_table_name}"
                    
                    print(f"DB table name: {db_table_name}")  # Debug log
                    
                    # Add session_id to DataFrame
                    # df['session_id'] = session_id
                    
                    # Save DataFrame to database
                    df.to_sql(
                        db_table_name,
                        conn,
                        if_exists='replace',
                        index=False
                    )
                    
                    print(f"Saved data to table: {db_table_name}")  # Debug log
                    
                    # Register table in registry
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO table_registry 
                        (session_id, table_name, original_name, row_count, column_count)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        session_id,
                        db_table_name,  # Store the actual db table name
                        table['name'],   # Store original file name
                        table['rows'],
                        table['columns']
                    ))
                    
                    print(f"Registered table in registry: {table['name']} -> {db_table_name}")  # Debug log
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Error saving tables to database: {e}")
            return False
    
    def save_state_summary(self, state_name: str, summary: Dict, session_id: str) -> bool:
        """Save state summary to database
        
        Args:
            state_name: Name of the state
            summary: Dictionary containing summary data
            session_id: Session ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create table if not exists with the new is_time_series column
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {state_name}_summary (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        num_tables TEXT,
                        problem_type TEXT,
                        target_column TEXT,
                        has_target TEXT,
                        is_time_series TEXT,
                        table_names TEXT,
                        table_rows TEXT,
                        table_columns TEXT,
                        completion_time TEXT,
                        recommendation_approach TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Check if recommendation_approach column exists, add it if not
                cursor.execute(f"PRAGMA table_info({state_name}_summary)")
                columns = [column[1] for column in cursor.fetchall()]
                
                if 'recommendation_approach' not in columns:
                    print(f"Adding recommendation_approach column to {state_name}_summary table")
                    cursor.execute(f"ALTER TABLE {state_name}_summary ADD COLUMN recommendation_approach TEXT")
                
                # Insert summary data
                cursor.execute(
                    f"""
                    INSERT INTO {state_name}_summary 
                    (session_id, num_tables, problem_type, target_column, has_target, is_time_series, 
                    table_names, table_rows, table_columns, completion_time, recommendation_approach)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        summary.get('num_tables', ''),
                        summary.get('problem_type', ''),
                        summary.get('target_column', ''),
                        summary.get('has_target', ''),
                        summary.get('is_time_series', ''),
                        summary.get('table_names', ''),
                        summary.get('table_rows', ''),
                        summary.get('table_columns', ''),
                        summary.get('completion_time', ''),
                        summary.get('recommendation_approach', '')
                    )
                )
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Error in save_state_summary: {str(e)}")
            return False
    
    def get_table_data(self, table_name: str) -> Optional[pd.DataFrame]:
        """Retrieve table data from database
        
        Args:
            table_name (str): Name of the table to retrieve
            
        Returns:
            Optional[pd.DataFrame]: Retrieved data or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                return pd.read_sql(f"SELECT * FROM {table_name}", conn)
        except Exception as e:
            print(f"Error retrieving table data: {e}")
            return None
    
    def get_state_summary(self, session_id: str) -> Optional[Dict]:
        """Retrieve state summary from database
        
        Args:
            session_id (str): Session identifier
            
        Returns:
            Optional[Dict]: State summary or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM onboarding_summary WHERE session_id = ?",
                    (session_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    columns = [desc[0] for desc in cursor.description]
                    return dict(zip(columns, row))
                return None
                
        except Exception as e:
            print(f"Error retrieving state summary: {e}")
            return None
    
    def save_table_to_db(self, state_name: str, table_name: str, df: pd.DataFrame, session_id: str) -> bool:
        """Save DataFrame to database with state-specific naming
        
        Args:
            state_name: Name of the state (for schema organization)
            table_name: Name for the table
            df: DataFrame to save
            session_id: Session identifier
        """
        try:
            # Add session_id to DataFrame
            # df['session_id'] = session_id
            
            # Create full table name with state prefix
            full_table_name = f"{state_name}_{table_name}"
            
            # Save to database
            with sqlite3.connect(self.db_path) as conn:
                df.to_sql(full_table_name, conn, if_exists='append', index=False)
            return True
            
        except Exception as e:
            print(f"Error saving table to database: {e}")
            return False

    def fetch_state_summary(self, state_name: str, session_id: str) -> Optional[Dict]:
        """Fetch state summary from database
        
        Args:
            state_name (str): Name of the state (for table naming)
            session_id (str): Session identifier
            
        Returns:
            Optional[Dict]: State summary or None if not found
        """
        try:
            table_name = f"{state_name}_summary"
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SELECT * FROM {table_name} WHERE session_id = ?",
                    (session_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    columns = [desc[0] for desc in cursor.description]
                    return dict(zip(columns, row))
                return None
                
        except Exception as e:
            print(f"Error fetching state summary: {e}")
            return None

    def fetch_table(self, state_name: str, table_name: str, session_id: str) -> Optional[pd.DataFrame]:
        """Fetch table data from database"""
        try:
            # First get the actual table name from registry
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT table_name FROM table_registry 
                    WHERE session_id = ? AND original_name = ?
                """, (session_id, table_name))
                
                result = cursor.fetchone()
                if not result:
                    print(f"No table found in registry for {table_name}")
                    return None
                    
                db_table_name = result[0]
                
                # Now fetch the actual data without session_id filter
                query = f"SELECT * FROM {db_table_name}"
                df = pd.read_sql(query, conn)
                
                return df if not df.empty else None
                
        except Exception as e:
            print(f"Error fetching table: {e}")
            return None

    def save_table_mappings(self, mapping_data: Dict) -> bool:
        """Save table mappings to database
        
        Args:
            mapping_data (Dict): Dictionary containing mapping information
            
        Returns:
            bool: Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create table if not exists
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS mappings_summary (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        table_name TEXT NOT NULL,
                        mappings TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Insert mapping data
                cursor.execute("""
                    INSERT INTO mappings_summary (session_id, table_name, mappings)
                    VALUES (?, ?, ?)
                """, (
                    mapping_data['session_id'],
                    mapping_data['table_name'],
                    mapping_data['mappings']
                ))
                
                # Register in table registry - include all entries including state summary
                table_name = 'mappings_summary'
                original_name = f"mappings_{mapping_data['table_name']}"
                
                cursor.execute("""
                    INSERT OR REPLACE INTO table_registry 
                    (session_id, table_name, original_name, row_count, column_count)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    mapping_data['session_id'],
                    table_name,
                    original_name,
                    1,
                    3  # session_id, table_name, and mappings
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Error saving table mappings: {e}")
            return False

    def reset_session_data(self, session_id: str) -> bool:
        """Reset all data for a specific session
        
        Args:
            session_id (str): Session identifier
            
        Returns:
            bool: Success status
        """
        try:
            # For a complete reset, simply delete and recreate the database file
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
                print(f"Deleted database file: {self.db_path}")
                
            # Reinitialize the database
            self._ensure_db_path()
            self._init_db()
            print("Reinitialized database")
            
            return True
                
        except Exception as e:
            print(f"Error resetting session data: {e}")
            return False

    def reset_state_data(self, session_id: str, state_name: str) -> bool:
        """Reset data for a specific state in a session
        
        Args:
            session_id (str): Session identifier
            state_name (str): Name of the state to reset
            
        Returns:
            bool: Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Drop state summary tables - try both naming conventions
                try:
                    cursor.execute(f"DROP TABLE IF EXISTS {state_name}_state_summary")
                except sqlite3.OperationalError:
                    pass
                    
                try:
                    cursor.execute(f"DROP TABLE IF EXISTS {state_name}_summary")
                except sqlite3.OperationalError:
                    pass
                
                # If it's the onboarding state, also drop data tables
                if state_name == 'onboarding':
                    # Get all tables in the registry for this session
                    cursor.execute(
                        "SELECT table_name FROM table_registry WHERE session_id = ?",
                        (session_id,)
                    )
                    tables = [row[0] for row in cursor.fetchall()]
                    
                    # Drop each table
                    for table in tables:
                        try:
                            cursor.execute(f"DROP TABLE IF EXISTS {table}")
                        except sqlite3.OperationalError:
                            # Table might not exist, continue
                            pass
                    
                    # Delete from registry
                    cursor.execute("DELETE FROM table_registry WHERE session_id = ?", (session_id,))
                
                # If it's the mapping state, delete mappings
                elif state_name == 'mapping':
                    cursor.execute("DELETE FROM table_mappings WHERE session_id = ?", (session_id,))
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Error resetting state data: {e}")
            return False

    def get_mapping_summary(self, session_id):
        """
        Get mapping summary for a session from the database.
        
        Args:
            session_id (str): The session ID
            
        Returns:
            dict: Mapping summary or None if not found
        """
        try:
            import json
            
            with sqlite3.connect(self.db_path) as conn:
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
                        print("No mappings found in database")
                        return None
                else:
                    # Combine mappings from all tables
                    field_mappings = {}
                    for table_name, mappings_json in results:
                        table_mappings = json.loads(mappings_json)
                        field_mappings.update(table_mappings)
                    
                    return field_mappings
                    
        except Exception as e:
            print(f"Error getting mapping summary: {str(e)}")
            return None

    def get_onboarding_summary(self, session_id):
        """
        Get onboarding summary for a session from the database.
        
        Args:
            session_id (str): The session ID
            
        Returns:
            dict: Onboarding summary or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM onboarding_summary WHERE session_id = ?",
                    (session_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    columns = [desc[0] for desc in cursor.description]
                    return dict(zip(columns, row))
                return None
                
        except Exception as e:
            print(f"Error getting onboarding summary: {str(e)}")
            return None

    def get_joined_dataframe(self, session_id):
        """
        Get the joined dataframe for a session from the database.
        
        Args:
            session_id (str): The session ID
            
        Returns:
            pd.DataFrame: Joined dataframe or None if not found
        """
        try:
            import pandas as pd
            import json
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # First try to get the final table name from join summary
                cursor.execute(
                    "SELECT final_table_name FROM join_summary WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
                    (session_id,)
                )
                result = cursor.fetchone()
                
                if not result:
                    print("No join summary found in database")
                    return None
                    
                final_table_name = result[0]
                print(f"Found final table name: {final_table_name}")
                
                # Load the final table data
                df = pd.read_sql(f"SELECT * FROM {final_table_name}", conn)
                
                if df.empty:
                    print("Final table is empty")
                    return None
                    
                print(f"Loaded final table with shape: {df.shape}")
                return df
                
        except Exception as e:
            print(f"Error getting joined dataframe: {str(e)}")
            return None 