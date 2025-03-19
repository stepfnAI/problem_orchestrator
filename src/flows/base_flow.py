"""
Base Flow - Abstract base class for all specialized flows.
"""

from typing import Dict, List, Any
import logging
from abc import ABC, abstractmethod
import uuid
import time

logger = logging.getLogger(__name__)

class BaseFlow(ABC):
    """
    Abstract base class for all specialized flows.
    
    All flows must inherit from this class and implement the required methods.
    """
    
    # Class attributes to be overridden by subclasses
    flow_id = None  # Unique identifier for the flow
    input_requirements = []  # List of required input fields
    output_format = {}  # Description of output format
    parameters = {}  # Description of configurable parameters
    
    def __init__(self, flow_id: str):
        """
        Initialize the Base Flow.
        
        Args:
            flow_id: Unique identifier for the flow
        """
        self.flow_id = flow_id
        self.status = "initialized"
        self.config = {}
        self.input_data = None
        self.output_data = None
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the flow with the given configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        logger.debug(f"Configured {self.__class__.__name__} with {config}")
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the flow.
        
        This method handles common pre- and post-processing for all flows.
        The actual implementation is delegated to _execute_implementation.
        
        Args:
            input_data: Input data for the flow
            
        Returns:
            Flow execution results
        """
        self.status = "running"
        start_time = time.time()
        
        try:
            # Validate input
            self._validate_input(input_data)
            
            # Execute the flow implementation
            result = self._execute_implementation(input_data)
            
            # Add execution metadata
            if "metadata" not in result:
                result["metadata"] = {}
            
            result["metadata"]["flow_id"] = self.flow_id
            result["metadata"]["execution_time"] = time.time() - start_time
            
            self.status = "completed"
            return result
        except Exception as e:
            logger.exception(f"Error executing flow {self.flow_id}: {str(e)}")
            self.status = "failed"
            return {
                "status": "failed",
                "error": str(e),
                "metadata": {
                    "flow_id": self.flow_id,
                    "execution_time": time.time() - start_time
                }
            }
    
    def get_status(self) -> str:
        """
        Get the current status of the flow.
        
        Returns:
            Current status
        """
        return self.status
    
    def terminate(self) -> None:
        """Terminate the flow execution."""
        self.status = "terminated"
    
    def _validate_input(self, input_data: Dict[str, Any]) -> None:
        """
        Validate the input data against the flow's requirements.
        
        Args:
            input_data: Input data to validate
            
        Raises:
            ValueError: If the input data is invalid
        """
        for req in self.input_requirements:
            if req not in input_data:
                raise ValueError(f"Missing required input: {req}")
    
    @abstractmethod
    def _execute_implementation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implementation of the flow execution.
        
        This method must be implemented by all subclasses.
        
        Args:
            input_data: Input data for the flow
            
        Returns:
            Flow execution results
        """
        pass
    
    def _generate_id(self) -> str:
        """Generate a unique ID."""
        return str(uuid.uuid4())[:8]
    
    def _get_timestamp(self) -> str:
        """Get the current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _generate_summary(self) -> str:
        """
        Generate a summary of the flow execution.
        
        Returns:
            Summary string
        """
        return f"Executed {self.__class__.__name__}" 