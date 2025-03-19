"""
Serialization - Utilities for serializing and deserializing objects.
"""

import json
import pickle
from typing import Any, Dict, List, Union
import os
import logging

logger = logging.getLogger(__name__)

def serialize_to_json(obj: Any, file_path: str = None, indent: int = 2) -> Union[str, None]:
    """
    Serialize an object to JSON.
    
    Args:
        obj: Object to serialize
        file_path: Path to save JSON file, or None to return JSON string
        indent: Indentation level for JSON formatting
        
    Returns:
        JSON string if file_path is None, otherwise None
    """
    try:
        if file_path:
            # Create directory if it doesn't exist
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            
            # Write to file
            with open(file_path, 'w') as f:
                json.dump(obj, f, indent=indent)
            logger.debug(f"Serialized object to {file_path}")
            return None
        else:
            # Return JSON string
            return json.dumps(obj, indent=indent)
    except Exception as e:
        logger.error(f"Error serializing to JSON: {e}")
        raise


def deserialize_from_json(file_path: str = None, json_str: str = None) -> Any:
    """
    Deserialize an object from JSON.
    
    Args:
        file_path: Path to JSON file, or None to use json_str
        json_str: JSON string, or None to use file_path
        
    Returns:
        Deserialized object
    """
    try:
        if file_path:
            with open(file_path, 'r') as f:
                obj = json.load(f)
            logger.debug(f"Deserialized object from {file_path}")
            return obj
        elif json_str:
            return json.loads(json_str)
        else:
            raise ValueError("Either file_path or json_str must be provided")
    except Exception as e:
        logger.error(f"Error deserializing from JSON: {e}")
        raise


def serialize_to_pickle(obj: Any, file_path: str) -> None:
    """
    Serialize an object to a pickle file.
    
    Args:
        obj: Object to serialize
        file_path: Path to save pickle file
    """
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # Write to file
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        logger.debug(f"Serialized object to {file_path}")
    except Exception as e:
        logger.error(f"Error serializing to pickle: {e}")
        raise


def deserialize_from_pickle(file_path: str) -> Any:
    """
    Deserialize an object from a pickle file.
    
    Args:
        file_path: Path to pickle file
        
    Returns:
        Deserialized object
    """
    try:
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        logger.debug(f"Deserialized object from {file_path}")
        return obj
    except Exception as e:
        logger.error(f"Error deserializing from pickle: {e}")
        raise


def is_json_serializable(obj: Any) -> bool:
    """
    Check if an object is JSON serializable.
    
    Args:
        obj: Object to check
        
    Returns:
        True if the object is JSON serializable, False otherwise
    """
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False


def make_json_serializable(obj: Any) -> Any:
    """
    Convert an object to a JSON serializable form.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON serializable form of the object
    """
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, set):
        return [make_json_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return make_json_serializable(obj.__dict__)
    elif hasattr(obj, 'to_dict'):
        return make_json_serializable(obj.to_dict())
    elif hasattr(obj, 'to_json'):
        return make_json_serializable(obj.to_json())
    else:
        # Try to convert to string if not serializable
        if not is_json_serializable(obj):
            return str(obj)
        return obj 