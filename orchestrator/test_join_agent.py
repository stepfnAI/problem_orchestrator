import streamlit as st
import pandas as pd
import os
import json
import sys
from pathlib import Path

# Add the parent directory to sys.path to make orchestrator importable
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

from sfn_blueprint import Task
from orchestrator.agents.join_suggestion_agent import SFNJoinSuggestionAgent

def main():
    st.title("Join Suggestion Agent Test")
    st.write("Upload multiple CSV files to test the join suggestion agent")
    
    # File uploader for multiple files
    uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)
    
    # Store uploaded dataframes and their metadata
    tables = {}
    tables_metadata = []
    
    # Process uploaded files
    if uploaded_files:
        st.subheader("Uploaded Tables")
        
        for uploaded_file in uploaded_files:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Display table info
            table_name = os.path.splitext(uploaded_file.name)[0]
            st.write(f"**Table**: {table_name} ({df.shape[0]} rows, {df.shape[1]} columns)")
            
            # Store the dataframe
            tables[table_name] = df
            
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
            
            # Show a sample of the data
            with st.expander(f"Preview of {table_name}"):
                st.dataframe(df.head())
    
    # Button to trigger join suggestion
    if len(tables) >= 2 and st.button("Get Join Suggestions"):
        with st.spinner("🤖 AI is analyzing tables for join suggestions..."):
            # Initialize the join suggestion agent
            join_agent = SFNJoinSuggestionAgent()
            
            # Create task data
            task_data = {
                'available_tables': list(tables.keys()),
                'tables_metadata': tables_metadata,
                'other_info': 'Test run for join suggestion agent'
            }
            
            # Create task
            join_task = Task("Suggest join", data=task_data)
            
            try:
                # Execute the task
                join_suggestion = join_agent.execute_task(join_task)
                
                # Display the results
                st.subheader("Join Suggestion Results")
                
                # Format the output for display
                st.json(join_suggestion)
                
                # Show a more user-friendly version
                st.subheader("Suggested Join")
                tables_to_join = join_suggestion.get("tables_to_join", [])
                join_type = join_suggestion.get("type_of_join", "")
                joining_fields = join_suggestion.get("joining_fields", [])
                explanation = join_suggestion.get("explanation", "")
                
                st.write(f"**Tables to Join**: {', '.join(tables_to_join)}")
                st.write(f"**Join Type**: {join_type}")
                
                st.write("**Joining Fields**:")
                for field_pair in joining_fields:
                    if len(field_pair) == 2:
                        st.write(f"- {field_pair[0]} = {field_pair[1]}")
                
                st.write("**Explanation**:")
                st.write(explanation)
                
                # Optional: Show what the joined data would look like
                if len(tables_to_join) == 2 and len(joining_fields) > 0:
                    try:
                        st.subheader("Preview of Joined Data")
                        table1, table2 = tables_to_join
                        df1 = tables[table1]
                        df2 = tables[table2]
                        
                        # Create join condition
                        join_conditions = []
                        for left_col, right_col in joining_fields:
                            if left_col in df1.columns and right_col in df2.columns:
                                join_conditions.append(f"df1['{left_col}'] == df2['{right_col}']")
                        
                        if join_conditions:
                            # Execute the join
                            if join_type.lower() == "inner":
                                joined_df = pd.merge(df1, df2, left_on=[pair[0] for pair in joining_fields], 
                                                    right_on=[pair[1] for pair in joining_fields], how='inner')
                            elif join_type.lower() == "left":
                                joined_df = pd.merge(df1, df2, left_on=[pair[0] for pair in joining_fields], 
                                                    right_on=[pair[1] for pair in joining_fields], how='left')
                            elif join_type.lower() == "right":
                                joined_df = pd.merge(df1, df2, left_on=[pair[0] for pair in joining_fields], 
                                                    right_on=[pair[1] for pair in joining_fields], how='right')
                            else:
                                joined_df = pd.merge(df1, df2, left_on=[pair[0] for pair in joining_fields], 
                                                    right_on=[pair[1] for pair in joining_fields], how='outer')
                            
                            st.dataframe(joined_df.head())
                            st.write(f"Joined result: {joined_df.shape[0]} rows, {joined_df.shape[1]} columns")
                    except Exception as e:
                        st.error(f"Error previewing joined data: {str(e)}")
                
            except Exception as e:
                st.error(f"Error getting join suggestions: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

if __name__ == "__main__":
    main() 