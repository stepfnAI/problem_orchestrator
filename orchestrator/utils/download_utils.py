import pandas as pd
import io
from typing import Union, Dict, List, Any, Optional

class DownloadUtils:
    """Utility class for generating downloadable data files using Streamlit view methods"""
    
    @staticmethod
    def create_download_button(view, 
                              data: Union[pd.DataFrame, Dict, List], 
                              filename: str, 
                              file_format: str = 'csv',
                              index: bool = False,
                              button_text: Optional[str] = None) -> None:
        """
        Display a download button in the Streamlit view
        
        Args:
            view: The view object to display the button
            data: The data to download (DataFrame, Dict, or List)
            filename: Name of the file to download
            file_format: Format of the file (csv, excel, json)
            index: Whether to include index in CSV/Excel output
            button_text: Custom text for the button
        """
        if file_format not in ['csv', 'excel', 'json']:
            raise ValueError("file_format must be one of 'csv', 'excel', or 'json'")
            
        # Convert data to appropriate format if needed
        if not isinstance(data, pd.DataFrame) and file_format in ['csv', 'excel']:
            data = pd.DataFrame(data)
        
        # Create in-memory buffer
        buffer = io.BytesIO()
        
        # Determine file extension and mime type
        if file_format == 'csv':
            data.to_csv(buffer, index=index)
            mime_type = 'text/csv'
            file_ext = 'csv'
        elif file_format == 'excel':
            data.to_excel(buffer, index=index)
            mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            file_ext = 'xlsx'
        elif file_format == 'json':
            if isinstance(data, pd.DataFrame):
                data.to_json(buffer, orient='records')
            else:
                import json
                buffer.write(json.dumps(data).encode())
            mime_type = 'application/json'
            file_ext = 'json'
        
        # Reset buffer position
        buffer.seek(0)
        
        # If no custom button text, use default
        if not button_text:
            button_text = f"Download {filename}.{file_ext}"
        
        # Use the view's create_download_button method
        view.create_download_button(
            label=button_text,
            data=buffer,
            file_name=f"{filename}.{file_ext}",
            mime_type=mime_type
        )
    
    @staticmethod
    def download_multiple_formats(view, 
                                 data: Union[pd.DataFrame, Dict, List],
                                 filename: str,
                                 formats: List[str] = ['csv', 'excel'],
                                 index: bool = False) -> None:
        """
        Display multiple download buttons for different formats
        
        Args:
            view: The view object to display the buttons
            data: The data to download
            filename: Base name for the downloaded files
            formats: List of formats to provide (csv, excel, json)
            index: Whether to include index in output
        """
        # Create columns for the buttons
        cols = view.create_columns(len(formats))
        
        # Add a download button for each format
        for i, fmt in enumerate(formats):
            with cols[i]:
                icon = "📥"
                ext = "xlsx" if fmt == "excel" else fmt
                DownloadUtils.create_download_button(
                    view=view,
                    data=data,
                    filename=filename,
                    file_format=fmt,
                    index=index,
                    button_text=f"{icon} Download ({ext.upper()})"
                ) 