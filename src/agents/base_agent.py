"""
Base Agent - Abstract base class for all specialized agents.
"""

from typing import Dict, Any
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """
    Abstract base class for all specialized agents.
    
    Agents are responsible for performing specific tasks within flows.
    """
    
    def __init__(self, agent_id: str):
        """
        Initialize the agent.
        
        Args:
            agent_id: Unique identifier for the agent
        """
        self.agent_id = agent_id
        logger.debug(f"Initialized agent: {agent_id}")
    
    @abstractmethod
    def execute(self, task_input: Dict[str, Any]) -> Any:
        """
        Execute the agent's task.
        
        Args:
            task_input: Input data for the task
            
        Returns:
            Result of the task execution
        """
        pass 