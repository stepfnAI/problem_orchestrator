import pandas as pd
from typing import List, Tuple, Optional, Any, Dict
from sfn_blueprint.views.streamlit_view import SFNStreamlitView
import streamlit as st

class SFNStreamlitView(SFNStreamlitView):
    def __init__(self, title: str):
        self.title = title

    def display_title(self):
        """Display the main title of the app"""
        st.title(self.title)

    def create_columns(self, ratios: List[int]) -> List[Any]:
        """Create columns with specified width ratios"""
        return st.columns(ratios)

    def display_header(self, text: str):
        """Display a header"""
        st.header(text)

    def display_subheader(self, text: str):
        """Display a subheader"""
        st.subheader(text)

    def display_markdown(self, text: str):
        """Display markdown text"""
        st.markdown(text)

    def display_dataframe(self, df: pd.DataFrame):
        """Display a pandas DataFrame"""
        # Create a copy to avoid modifying the original DataFrame
        display_df = df.copy()
        
        # If index is not default numeric, add it as a column while preserving the original
        if not isinstance(display_df.index, pd.RangeIndex):
            # Reset index but keep it as a column
            display_df = display_df.reset_index()
        
        st.dataframe(display_df)

    def file_uploader(self, label: str, key: Optional[str] = None, accepted_types: List[str] = None) -> Any:
        """Create a file uploader"""
        return st.file_uploader(label, type=accepted_types, key=key)

    def display_spinner(self, text: str):
        """Create a spinner with custom text"""
        return st.spinner(text)

    def show_message(self, message: str, message_type: str = "info"):
        """Display a message with specified type (info, success, warning, error)"""
        if message_type == "success":
            st.success(message)
        elif message_type == "warning":
            st.warning(message)
        elif message_type == "error":
            st.error(message)
        else:
            st.info(message)

    def display_button(self, label: str, key: Optional[str] = None) -> bool:
        """Create a button and return its clicked state"""
        return st.button(label, key=key)

    def radio_select(self, label: str, options: List[str], key: str = None) -> str:
        """Create a radio button selection"""
        return st.radio(label, options, key=key)

    def select_box(self, label: str, options: List[str], key: Optional[str] = None, default: Optional[str] = None) -> str:
        """Create a select box"""
        index = 0 if default is None else options.index(default)
        return st.selectbox(label, options, index=index, key=key)

    def checkbox(self, label=None, key=None, value=False, disabled=False, label_visibility="visible"):
        """Create a checkbox with a default hidden label if none provided"""
        if label is None or label == "":
            # Generate a label based on the key if no label is provided
            label = key if key else "checkbox"
        return st.checkbox(
            label=label,
            key=key,
            value=value,
            disabled=disabled,
            label_visibility=label_visibility
        )

    def create_download_button(self, label: str, data: Any, file_name: str, mime_type: str):
        """Create a download button"""
        st.download_button(label=label, data=data, file_name=file_name, mime=mime_type)

    def display_progress_bar(self, progress: float):
        """Display a progress bar"""
        st.progress(progress)

    def create_progress_container(self) -> Tuple[Any, Any]:
        """Create a progress bar and status text container"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        return progress_bar, status_text

    def update_progress(self, progress_bar: Any, value: float):
        """Update progress bar value"""
        progress_bar.progress(value)

    def update_text(self, text_container: Any, text: str):
        """Update text in a container"""
        text_container.text(text)

    def rerun_script(self):
        """Rerun the Streamlit script"""
        st.rerun()

    def stop_execution(self):
        """Stop script execution"""
        st.stop()

    def text_area(self, label: str, value: str = "", height: int = None, help: str = None, key: str = None) -> str:
        """Display a multi-line text input widget
        
        Args:
            label (str): Label for the text area
            value (str, optional): Default text value. Defaults to "".
            height (int, optional): Height of the text area in pixels. Defaults to None.
            help (str, optional): Tooltip help text. Defaults to None.
            key (str, optional): Unique key for the component. Defaults to None.
            
        Returns:
            str: Text entered by the user
        """
        return st.text_area(
            label=label,
            value=value,
            height=height,
            help=help,
            key=key
        )

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