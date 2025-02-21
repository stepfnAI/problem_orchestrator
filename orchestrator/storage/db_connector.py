import sqlite3
from typing import Dict, List, Optional, Any
import pandas as pd
import json
from pathlib import Path

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
            
            # Create state summary table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS onboarding_state_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    num_tables INTEGER NOT NULL,
                    tables_in_db BOOLEAN NOT NULL,
                    problem_type TEXT NOT NULL,
                    target_column_present BOOLEAN NOT NULL,
                    tables_uploaded BOOLEAN NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create table registry
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS table_registry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    table_name TEXT NOT NULL,
                    original_name TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    column_count INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def save_uploaded_tables(self, session_id: str, tables: List[Dict]) -> bool:
        """Save uploaded tables to database
        
        Args:
            session_id (str): Current session identifier
            tables (List[Dict]): List of table information dictionaries
            
        Returns:
            bool: Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Save each table's data
                for table in tables:
                    # Read the DataFrame
                    df = pd.read_csv(table['path']) if table['name'].endswith('.csv') \
                        else pd.read_excel(table['path'])
                    
                    # Create table name for database
                    safe_table_name = f"raw_table_{table['name'].split('.')[0].lower()}"
                    
                    # Save DataFrame to database
                    df.to_sql(
                        safe_table_name,
                        conn,
                        if_exists='replace',
                        index=False
                    )
                    
                    # Register table in registry
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO table_registry 
                        (session_id, table_name, original_name, row_count, column_count)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        session_id,
                        safe_table_name,
                        table['name'],
                        table['rows'],
                        table['columns']
                    ))
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Error saving tables to database: {e}")
            return False
    
    def save_state_summary(self, state_name: str, summary_data: Dict[str, Any], session_id: str) -> bool:
        """Save state summary dynamically based on provided data
        
        Args:
            state_name: Name of the state (for table naming)
            summary_data: Dictionary containing summary data
            session_id: Session identifier
        """
        try:
            # Add session_id to summary data
            summary_data['session_id'] = session_id
            
            # Create column definitions from dictionary keys
            columns = list(summary_data.keys())
            column_defs = [f"{col} TEXT" for col in columns]
            
            # Create table if not exists
            table_name = f"{state_name}_summary"
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                {', '.join(column_defs)}
            )
            """
            
            # Insert data
            placeholders = ','.join(['?' for _ in columns])
            insert_sql = f"""
            INSERT INTO {table_name} ({','.join(columns)})
            VALUES ({placeholders})
            """
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(create_table_sql)
                conn.execute(insert_sql, [str(summary_data[col]) for col in columns])
                conn.commit()
            return True
            
        except Exception as e:
            print(f"Error saving state summary: {e}")
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
                    "SELECT * FROM onboarding_state_summary WHERE session_id = ?",
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
            df['session_id'] = session_id
            
            # Create full table name with state prefix
            full_table_name = f"{state_name}_{table_name}"
            
            # Save to database
            with sqlite3.connect(self.db_path) as conn:
                df.to_sql(full_table_name, conn, if_exists='append', index=False)
            return True
            
        except Exception as e:
            print(f"Error saving table to database: {e}")
            return False 