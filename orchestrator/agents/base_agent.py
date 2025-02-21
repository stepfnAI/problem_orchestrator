from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """Base class for all agents"""
    
    @abstractmethod
    def execute(self):
        """Execute the agent's main functionality"""
        pass 