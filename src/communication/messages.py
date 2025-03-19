"""
Messages - Defines message types for communication between components.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class BaseMessage:
    """Base class for all messages."""
    message_type: str
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class FlowRequestMessage(BaseMessage):
    """Message for requesting flow execution."""
    flow_id: str
    input_tables: List[str]
    expected_output: str
    parameters: Dict[str, Any] = None
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.message_type = "flow_request"
        if self.parameters is None:
            self.parameters = {}
        if self.context is None:
            self.context = {}


@dataclass
class FlowResponseMessage(BaseMessage):
    """Message for flow execution response."""
    flow_id: str
    status: str  # initializing, in_progress, needs_input, completed, failed
    input_tables: List[str]
    output_table: Optional[str] = None
    summary: str = ""
    details: Dict[str, Any] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.message_type = "flow_response"
        if self.details is None:
            self.details = {}


@dataclass
class UserInputRequestMessage(BaseMessage):
    """Message for requesting user input."""
    requester_id: str
    question: str
    priority: int = 5
    options: Optional[List[str]] = None
    context: Optional[str] = None
    response_format: str = "free_text"  # single_select, multi_select, free_text, yes_no
    
    def __post_init__(self):
        super().__post_init__()
        self.message_type = "user_input_request"


@dataclass
class UserInputResponseMessage(BaseMessage):
    """Message for user input response."""
    requester_id: str
    question: str
    response: Any
    response_format: str
    
    def __post_init__(self):
        super().__post_init__()
        self.message_type = "user_input_response"


@dataclass
class DataLineageRecordMessage(BaseMessage):
    """Message for data lineage record."""
    table_id: str
    parent_tables: List[str]
    creating_flow: str
    flow_history: List[Dict[str, Any]]
    schema: Dict[str, Dict[str, str]]
    child_tables: List[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.message_type = "data_lineage_record"
        if self.child_tables is None:
            self.child_tables = []


def create_message_from_dict(message_dict: Dict[str, Any]) -> BaseMessage:
    """
    Create a message object from a dictionary.
    
    Args:
        message_dict: Dictionary representation of a message
        
    Returns:
        Message object
    """
    message_type = message_dict.get("message_type")
    
    if message_type == "flow_request":
        return FlowRequestMessage(**message_dict)
    elif message_type == "flow_response":
        return FlowResponseMessage(**message_dict)
    elif message_type == "user_input_request":
        return UserInputRequestMessage(**message_dict)
    elif message_type == "user_input_response":
        return UserInputResponseMessage(**message_dict)
    elif message_type == "data_lineage_record":
        return DataLineageRecordMessage(**message_dict)
    else:
        raise ValueError(f"Unknown message type: {message_type}") 