from sfn_blueprint.views.streamlit_view import SFNStreamlitView
from typing import Any, List, Optional
import streamlit as st
import os
from pathlib import Path
import pandas as pd

class StreamlitView(SFNStreamlitView):
    def __init__(self, title: str):
        self.title = title

    def file_uploader(self, label: str, accepted_types: List[str], key: Optional[str] = None) -> Any:
        """Override file_uploader to include key parameter"""
        return st.file_uploader(label, type=accepted_types, key=key)

    def select_box(self, label: str, options: List[str], default: Optional[str] = None, key: Optional[str] = None) -> str:
        """Add select box functionality with default value support
        
        Args:
            label (str): Label for the select box
            options (List[str]): List of options to choose from
            default (str, optional): Default value to select. Defaults to None.
            key (str, optional): Unique key for the component. Defaults to None.
        
        Returns:
            str: Selected option
        """
        # Find index of default value if provided
        index = options.index(default) if default in options else 0
        return st.selectbox(label, options, index=index, key=key)
    
    def save_uploaded_file(self, uploaded_file: Any) -> Optional[str]:
        """Save uploaded file temporarily"""
        if uploaded_file is not None:
            # Create temp directory if not exists
            temp_dir = Path('./temp_files')
            temp_dir.mkdir(exist_ok=True)

            # Save file
            file_path = temp_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            return str(file_path)
        return None
    
    def delete_uploaded_file(self, file_path: str) -> bool:
        """Delete temporary file"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
        except Exception as e:
            self.show_message(f"Error deleting file: {e}", "error")
        return False 

    def display_radio(self, label: str, options: list, key: str = None) -> str:
        """Display a radio button group
        
        Args:
            label (str): Label for the radio group
            options (list): List of options to display
            key (str): Unique key for the component
            
        Returns:
            str: Selected option
        """
        return st.radio(label, options, key=key) 

    def text_area(
        self,
        label: str,
        value: str = "",
        height: int = None,
        help: str = None,
        placeholder: str = None,
        key: str = None
    ) -> str:
        """Display a multi-line text input widget
        
        Args:
            label (str): Label for the text area
            value (str, optional): Default text value. Defaults to "".
            height (int, optional): Height of the text area in pixels. Defaults to None.
            help (str, optional): Tooltip help text. Defaults to None.
            placeholder (str, optional): Placeholder text shown when empty. Defaults to None.
            key (str, optional): Unique key for the component. Defaults to None.
            
        Returns:
            str: Text entered by the user
        """
        return st.text_area(
            label=label,
            value=value,
            height=height,
            help=help,
            placeholder=placeholder,
            key=key
        ) 

    def multiselect(self, label: str, options: List[Any], default: Optional[List[Any]] = None, key: Optional[str] = None) -> List[Any]:
        """Display a multi-select widget
        
        Args:
            label (str): Label for the multi-select
            options (List[Any]): List of options to choose from
            default (List[Any], optional): Default selected values. Defaults to None.
            key (str, optional): Unique key for the component. Defaults to None.
            
        Returns:
            List[Any]: List of selected options
        """
        return st.multiselect(
            label=label,
            options=options,
            default=default,
            key=key
        ) 

    def create_columns(self, ratios: List[int]) -> tuple:
        """Create columns with specified width ratios
        
        Args:
            ratios (List[int]): List of relative widths for columns
            
        Returns:
            tuple: Tuple of column objects
        """
        return st.columns(ratios)

    def get_column(self, index: int):
        """Get a specific column
        
        Args:
            index (int): Index of the column to get
            
        Returns:
            streamlit.delta_generator.DeltaGenerator: Column object
        """
        return self.columns[index] 

    def plot_bar_chart(self, data: pd.DataFrame, x_col: str, y_col: str, title: str = None):
        """Display a bar chart using streamlit with plotly
        
        Args:
            data (pd.DataFrame): DataFrame containing the data to plot
            x_col (str): Column name for x-axis
            y_col (str): Column name for y-axis
            title (str, optional): Title for the chart. Defaults to None.
        """
        # Format dates to YYYY-MM format
        data = data.copy()
        data[x_col] = pd.to_datetime(data[x_col]).dt.strftime('%Y-%m')
        
        # Extract field name from title for y-axis label
        y_axis_label = 'Forecasted Value'
        if title and 'Forecasted' in title:
            y_axis_label = title.replace('by Month', '').strip()
        
        # Create plotly figure
        fig = {
            'data': [{
                'type': 'bar',
                'x': data[x_col],
                'y': data[y_col],
                'marker': {'color': '#00A4EF'},
            }],
            'layout': {
                'title': title,
                'xaxis': {'title': 'Month', 'tickangle': 45},
                'yaxis': {'title': y_axis_label},
                'template': 'plotly_dark',
                'height': 400,
                'margin': {'b': 100}  # Add bottom margin for rotated labels
            }
        }
        
        st.plotly_chart(fig, use_container_width=True) 

    def display_markdown(self, text: str, unsafe_allow_html: bool = False):
        """Display markdown text with optional HTML support
        
        Args:
            text (str): Markdown text to display
            unsafe_allow_html (bool): Whether to allow HTML in markdown
        """
        st.markdown(text, unsafe_allow_html=unsafe_allow_html) 