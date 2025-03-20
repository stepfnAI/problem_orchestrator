"""
User Input Service - Manages all user interactions across the system.
"""

from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class UserInputService:
    """
    Manages all user interactions across the system.
    
    This service standardizes user interaction patterns, queues and prioritizes
    interaction requests, and routes user responses to appropriate components.
    """
    
    def __init__(self, interactive=True):
        """Initialize the User Input Service."""
        self.interaction_queue = []
        self.interaction_history = []
        self.interactive = interactive
        logger.info(f"User Input Service initialized with interactive mode: {interactive}")
    
    def request_input(self, question: str, options: Optional[List[str]] = None, 
                     response_format: str = "free_text", 
                     context: Optional[str] = None,
                     requester_id: str = "system",
                     priority: int = 5) -> Any:
        """
        Request input from the user.
        
        Args:
            question: Question to ask the user
            options: List of options for the user to choose from
            response_format: Format of the expected response
            context: Additional context for the question
            requester_id: ID of the component requesting input
            priority: Priority of the request (1-10, 10 being highest)
            
        Returns:
            User's response
        """
        request = {
            "question": question,
            "options": options,
            "response_format": response_format,
            "context": context,
            "requester_id": requester_id,
            "priority": priority,
            "timestamp": self._get_timestamp()
        }
        
        logger.info(f"Input request from {requester_id}: {question} (interactive={self.interactive})")
        
        # Get the response
        if self.interactive:
            response = self._get_interactive_response(request)
        else:
            response = self._get_default_response(request)
        
        # Record the interaction
        self.interaction_history.append({
            "request": request,
            "response": response
        })
        
        return response
    
    def get_interaction_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of user interactions.
        
        Returns:
            List of interaction records
        """
        return self.interaction_history
    
    def _get_timestamp(self) -> str:
        """Get the current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _get_interactive_response(self, request: Dict[str, Any]) -> Any:
        """Get response from the user interactively."""
        question = request["question"]
        response_format = request["response_format"]
        options = request["options"]
        
        print(f"\n[USER INPUT REQUIRED] {question}")
        
        if response_format == "single_select" and options:
            print("Available options:")
            for i, option in enumerate(options):
                print(f"{i+1}. {option}")
            
            while True:
                try:
                    choice = input("\nEnter your choice (number): ")
                    index = int(choice) - 1
                    if 0 <= index < len(options):
                        selected = options[index]
                        print(f"You selected: {selected}")
                        return selected
                    else:
                        print(f"Please enter a number between 1 and {len(options)}")
                except ValueError:
                    print("Please enter a valid number")
        
        elif response_format == "multi_select" and options:
            for i, option in enumerate(options):
                print(f"{i+1}. {option}")
            
            while True:
                try:
                    choices = input("Enter your choices (comma-separated numbers): ")
                    indices = [int(c.strip()) - 1 for c in choices.split(",")]
                    if all(0 <= i < len(options) for i in indices):
                        return [options[i] for i in indices]
                    else:
                        print(f"Please enter numbers between 1 and {len(options)}")
                except ValueError:
                    print("Please enter valid numbers")
        
        elif response_format == "yes_no":
            while True:
                response = input("[yes/no] > ").lower()
                if response in ["yes", "y"]:
                    return "yes"
                elif response in ["no", "n"]:
                    return "no"
                else:
                    print("Please enter 'yes' or 'no'")
        
        else:  # free_text
            return input("Enter your response: ")
    
    def _get_default_response(self, request: Dict[str, Any]) -> Any:
        """Get a default response when not in interactive mode."""
        question = request["question"]
        response_format = request["response_format"]
        options = request["options"]
        
        logger.info(f"Non-interactive mode, using default response for: {question}")
        
        if response_format == "single_select" and options:
            return options[0]
        elif response_format == "multi_select" and options:
            return [options[0]]
        elif response_format == "yes_no":
            return "yes"
        else:
            return "Default response" 