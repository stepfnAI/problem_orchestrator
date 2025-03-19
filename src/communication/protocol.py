"""
Protocol - Defines the communication protocols between components.
"""

from typing import Dict, List, Any, Optional
import json
from datetime import datetime

class Protocol:
    """
    Defines the communication protocols between components.
    
    This class provides methods for creating standardized messages
    for communication between the Meta Agent and flows.
    """
    
    @staticmethod
    def create_flow_request(flow_id: str, input_tables: List[str], 
                           expected_output: str, parameters: Dict[str, Any],
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a flow request message.
        
        Args:
            flow_id: ID of the flow to execute
            input_tables: List of input table IDs
            expected_output: ID of the expected output table
            parameters: Parameters for the flow
            context: Context information for the flow
            
        Returns:
            Flow request message
        """
        return {
            "message_type": "flow_request",
            "flow_id": flow_id,
            "timestamp": datetime.now().isoformat(),
            "input_tables": input_tables,
            "expected_output": expected_output,
            "parameters": parameters,
            "context": context
        }
    
    @staticmethod
    def create_flow_response(flow_id: str, status: str, input_tables: List[str],
                            output_table: Optional[str], summary: str,
                            details: Dict[str, Any] = None,
                            error: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a flow response message.
        
        Args:
            flow_id: ID of the flow
            status: Status of the flow (initializing, in_progress, needs_input, completed, failed)
            input_tables: List of input table IDs
            output_table: ID of the output table
            summary: Concise description of actions and results
            details: Additional details about the flow execution
            error: Error message if status is failed
            
        Returns:
            Flow response message
        """
        return {
            "message_type": "flow_response",
            "flow_id": flow_id,
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "input_tables": input_tables,
            "output_table": output_table,
            "summary": summary,
            "details": details or {},
            "error": error
        }
    
    @staticmethod
    def create_user_input_request(requester_id: str, question: str,
                                 options: Optional[List[str]] = None,
                                 context: Optional[str] = None,
                                 priority: int = 5,
                                 response_format: str = "free_text") -> Dict[str, Any]:
        """
        Create a user input request message.
        
        Args:
            requester_id: ID of the component requesting input
            question: Question to ask the user
            options: List of options for the user to choose from
            context: Additional context for the question
            priority: Priority of the request (1-10, 10 being highest)
            response_format: Format of the expected response
            
        Returns:
            User input request message
        """
        return {
            "message_type": "user_input_request",
            "requester_id": requester_id,
            "timestamp": datetime.now().isoformat(),
            "priority": priority,
            "question": question,
            "options": options,
            "context": context,
            "response_format": response_format
        }
    
    @staticmethod
    def create_data_lineage_record(table_id: str, parent_tables: List[str],
                                  creating_flow: str, flow_history: List[Dict[str, Any]],
                                  schema: Dict[str, Dict[str, str]],
                                  child_tables: List[str] = None) -> Dict[str, Any]:
        """
        Create a data lineage record.
        
        Args:
            table_id: ID of the table
            parent_tables: List of parent table IDs
            creating_flow: ID of the flow that created the table
            flow_history: List of flow history records
            schema: Schema of the table
            child_tables: List of child table IDs
            
        Returns:
            Data lineage record
        """
        return {
            "table_id": table_id,
            "created_at": datetime.now().isoformat(),
            "parent_tables": parent_tables,
            "creating_flow": creating_flow,
            "flow_history": flow_history,
            "schema": schema,
            "child_tables": child_tables or []
        }
    
    @staticmethod
    def serialize_message(message: Dict[str, Any]) -> str:
        """
        Serialize a message to JSON.
        
        Args:
            message: Message to serialize
            
        Returns:
            JSON string
        """
        return json.dumps(message)
    
    @staticmethod
    def deserialize_message(message_str: str) -> Dict[str, Any]:
        """
        Deserialize a JSON message.
        
        Args:
            message_str: JSON string
            
        Returns:
            Deserialized message
        """
        return json.loads(message_str) 