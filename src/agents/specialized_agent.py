"""
Specialized Agent - Base class for specialized agents.
"""

from typing import Dict, Any, Optional
import logging
import os
import json

logger = logging.getLogger(__name__)

class SpecializedAgent:
    """
    Base class for specialized agents.
    
    Specialized agents are focused on specific tasks like mapping,
    feature engineering, etc.
    """
    
    def __init__(self, model_name: str = "default"):
        """
        Initialize the Specialized Agent.
        
        Args:
            model_name: Name of the model to use
        """
        self.model_name = model_name
        logger.info(f"Initialized {self.__class__.__name__} with model {model_name}")
    
    def get_prompt(self, prompt_type: str, **kwargs) -> Optional[str]:
        """
        Get a formatted prompt for a specific task.
        
        Args:
            prompt_type: Type of prompt to retrieve
            **kwargs: Variables to format the prompt with
            
        Returns:
            Formatted prompt string, or None if not found
        """
        # Look for prompt template in prompts directory
        prompt_dir = os.path.join(os.path.dirname(__file__), "..", "prompts")
        prompt_file = os.path.join(prompt_dir, f"{prompt_type}.txt")
        
        if not os.path.exists(prompt_file):
            logger.warning(f"Prompt file not found: {prompt_file}")
            return None
        
        try:
            with open(prompt_file, "r") as f:
                prompt_template = f.read()
            
            # Format the prompt with the provided variables
            return prompt_template.format(**kwargs)
        except Exception as e:
            logger.error(f"Error formatting prompt: {str(e)}")
            return None
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load a configuration file.
        
        Args:
            config_name: Name of the configuration to load
            
        Returns:
            Configuration dictionary, or empty dict if not found
        """
        config_dir = os.path.join(os.path.dirname(__file__), "..", "configs")
        config_file = os.path.join(config_dir, f"{config_name}.json")
        
        if not os.path.exists(config_file):
            logger.warning(f"Config file not found: {config_file}")
            return {}
        
        try:
            with open(config_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            return {} 