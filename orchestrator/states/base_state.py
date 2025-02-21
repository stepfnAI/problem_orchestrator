from abc import ABC, abstractmethod

class BaseState(ABC):
    """Base class for all states"""
    
    @abstractmethod
    def execute(self):
        """Execute the state's main functionality"""
        pass 