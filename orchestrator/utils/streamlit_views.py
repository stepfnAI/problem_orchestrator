from sfn_blueprint.views.streamlit_view import SFNStreamlitView
from typing import Any, List, Optional, Dict, Union
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

    def select_box(self, label: str, options: List[str], default: Optional[str] = None, index: Optional[int] = None, key: Optional[str] = None) -> str:
        """Add select box functionality with default value or index support
        
        Args:
            label (str): Label for the select box
            options (List[str]): List of options to choose from
            default (str, optional): Default value to select. Defaults to None.
            index (int, optional): Index of the default option to select. Defaults to None.
            key (str, optional): Unique key for the component. Defaults to None.
        
        Returns:
            str: Selected option
        """
        # If index is provided, use it directly
        if index is not None:
            return st.selectbox(label, options, index=index, key=key)
        
        # Otherwise, find index of default value if provided
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

    def create_expander(self, label: str, expanded: bool = False) -> Any:
        """Create an expandable/collapsible container
        
        Args:
            label (str): Label for the expander
            expanded (bool, optional): Whether the expander is initially expanded. Defaults to False.
            
        Returns:
            streamlit.delta_generator.DeltaGenerator: Expander object
        """
        return st.expander(label=label, expanded=expanded) 

    def display_table(self, data: List[Dict], use_container_width: bool = True):
        """Display a table from a list of dictionaries
        
        Args:
            data (List[Dict]): List of dictionaries where each dict represents a row
            use_container_width (bool, optional): Whether to expand table to container width. Defaults to True.
        """
        if data:
            # Convert list of dicts to DataFrame
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=use_container_width)
        else:
            # Show a message if no data
            st.info("No data to display") 

    def display_dataframe(self, df: pd.DataFrame, use_container_width: bool = True):
        """Safely display a pandas DataFrame
        
        Args:
            df (pd.DataFrame): DataFrame to display
            use_container_width (bool, optional): Whether to expand table to container width. Defaults to True.
        """
        if df is not None and not df.empty:
            # Display the dataframe
            st.dataframe(df, use_container_width=use_container_width)
        else:
            # Show a message if no data
            st.info("No data to display")

    def radio_select(self, label: str, options: List[str], default: Optional[str] = None, key: Optional[str] = None) -> str:
        """Display a radio button selection with default value support
        
        Args:
            label (str): Label for the radio group
            options (List[str]): List of options to choose from
            default (str, optional): Default value to select. Defaults to None.
            key (str, optional): Unique key for the component. Defaults to None.
            
        Returns:
            str: Selected option
        """
        # Find index of default value if provided
        index = options.index(default) if default and default in options else 0
        return st.radio(label, options, index=index, key=key) 
    

    def slider(self, 
              label: str, 
              min_value: Union[int, float] = 0.0, 
              max_value: Union[int, float] = 1.0, 
              value: Union[int, float] = 0.5, 
              step: Union[int, float] = 0.1,
              help: Optional[str] = None,
              key: Optional[str] = None) -> Union[int, float]:
        """Display a slider widget
        
        Args:
            label (str): Label for the slider
            min_value (Union[int, float]): Minimum value allowed
            max_value (Union[int, float]): Maximum value allowed
            value (Union[int, float]): Default value
            step (Union[int, float]): Step size for the slider
            help (str, optional): Help text to display. Defaults to None.
            key (str, optional): Unique key for the component
            
        Returns:
            Union[int, float]: Selected value
        """
        # Ensure consistent types
        if isinstance(min_value, int) and isinstance(max_value, int):
            # Convert step to int if it's a whole number
            if isinstance(step, float) and step.is_integer():
                step = int(step)
            # Convert value to int if it's a float with no decimal part
            if isinstance(value, float) and value.is_integer():
                value = int(value)
        
        return st.slider(
            label=label,
            min_value=min_value,
            max_value=max_value,
            value=value,
            step=step,
            help=help,
            key=key
        )
    
    def metric(self, 
              label: str, 
              value: str,
              delta: Optional[str] = None,
              help: Optional[str] = None) -> None:
        """Display a metric value with optional delta and help text
        
        Args:
            label (str): Metric label
            value (str): Metric value to display
            delta (str, optional): Delta value to display. Defaults to None.
            help (str, optional): Help text to display. Defaults to None.
        """
        st.metric(
            label=label,
            value=value,
            delta=delta,
            help=help
        ) 

    def selectbox(self, 
                 label: str, 
                 options: List[Any], 
                 index: int = 0, 
                 key: Optional[str] = None,
                 help: Optional[str] = None) -> Any:
        """Display a selectbox widget
        
        Args:
            label (str): Label for the selectbox
            options (List[Any]): List of options to choose from
            index (int, optional): Index of the default selected option. Defaults to 0.
            key (str, optional): Unique key for the component. Defaults to None.
            help (str, optional): Help text to display. Defaults to None.
            
        Returns:
            Any: Selected option
        """
        return st.selectbox(
            label=label,
            options=options,
            index=index,
            key=key,
            help=help
        ) 

    def select_option(self, label: str, options: List[str], key: Optional[str] = None) -> str:
        """Display a radio button group for selecting an option
        
        Args:
            label (str): Label for the selection
            options (List[str]): List of options to choose from
            key (str, optional): Unique key for the component. Defaults to None.
            
        Returns:
            str: Selected option
        """
        return st.radio(label, options, key=key) 