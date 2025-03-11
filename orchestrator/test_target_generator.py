import pandas as pd
import streamlit as st
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.target_generator_agent import SFNTargetGeneratorAgent
from sfn_blueprint import Task

def main():
    st.title("Target Generator Agent Test")
    
    # Create a sample dataframe
    if 'df' not in st.session_state:
        data = {
            'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005'],
            'age': [25, 40, 35, 22, 55],
            'income': [50000, 75000, 60000, 30000, 90000],
            'purchase_count': [10, 5, 15, 2, 20],
            'last_purchase_date': ['2023-01-15', '2023-02-20', '2023-01-05', '2023-03-10', '2022-12-01'],
            'product_category': ['Electronics', 'Clothing', 'Electronics', 'Books', 'Furniture']
        }
        st.session_state.df = pd.DataFrame(data)
    
    # Display the dataframe
    st.subheader("Sample DataFrame")
    st.dataframe(st.session_state.df)
    
    # Problem type selection
    problem_type = st.selectbox(
        "Select Problem Type",
        ["classification", "regression"]
    )
    
    # User instructions
    user_instructions = st.text_area(
        "Enter Instructions for Target Creation",
        "Create a target column where customers with income > 60000 are labeled as 'high_value' (1) and others as 'standard' (0)"
    )
    
    # Previous error message (for testing error handling)
    error_message = st.text_area(
        "Previous Error (if any)",
        ""
    )
    
    # Generate button
    if st.button("Generate Target Column"):
        with st.spinner("Generating target column..."):
            # Create agent
            agent = SFNTargetGeneratorAgent()
            
            # Create task
            task_data = {
                'user_instructions': user_instructions,
                'df': st.session_state.df,
                'problem_type': problem_type,
                'error_message': error_message
            }
            task = Task("Generate Target", data=task_data)
            
            # Execute task
            result = agent.execute_task(task)
            
            # Display results
            st.subheader("Generated Code")
            st.code(result['code'], language='python')
            
            st.subheader("Explanation")
            st.write(result['explanation'])
            
            st.subheader("Preview")
            st.write(result['preview'])
            
            # Execute the code
            st.subheader("Execution Result")
            try:
                # Create a copy of the dataframe
                df = st.session_state.df.copy()
                
                # Execute the code
                exec_locals = {'df': df, 'pd': pd}
                exec(result['code'], {}, exec_locals)
                df = exec_locals['df']
                
                # Display the result
                st.success("Code executed successfully!")
                st.dataframe(df)
                
                # Show target distribution
                st.subheader("Target Distribution")
                if 'target' in df.columns:
                    if problem_type == 'classification':
                        # For classification, show value counts
                        st.write(df['target'].value_counts())
                    else:
                        # For regression, show histogram
                        st.write(df['target'].describe())
                else:
                    st.error("No 'target' column was created by the code")
                    
            except Exception as e:
                st.error(f"Error executing code: {str(e)}")
                
                # Offer retry with error context
                if st.button("Retry with Error Context"):
                    st.experimental_rerun()

if __name__ == "__main__":
    main() 