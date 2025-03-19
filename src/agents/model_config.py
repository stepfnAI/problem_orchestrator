"""
Model Config - Configuration manager for language models.
"""

from typing import Dict, Any
import os
import json
import logging

logger = logging.getLogger(__name__)

class ModelConfig:
    """
    Configuration manager for language models.
    
    This class manages configurations for different language models,
    including parameters like temperature, max tokens, etc.
    """
    
    # Default configurations
    _default_configs = {
        "default": {
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        },
        "gpt-4": {
            "temperature": 0.7,
            "max_tokens": 2000,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        },
        "claude-3-opus": {
            "temperature": 0.5,
            "max_tokens": 4000,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
    }
    
    # Task-specific configurations
    _task_configs = {
        "field_mapping": {
            "default": {
                "temperature": 0.2,  # Lower temperature for more deterministic mappings
                "max_tokens": 500
            }
        },
        "feature_suggestion": {
            "default": {
                "temperature": 0.8,  # Higher temperature for more creative suggestions
                "max_tokens": 1000
            }
        },
        "schema_analysis": {
            "default": {
                "temperature": 0.3,
                "max_tokens": 800
            }
        }
    }
    
    @classmethod
    def get_config(cls, model_name: str, task: str = None) -> Dict[str, Any]:
        """
        Get configuration for a model and task.
        
        Args:
            model_name: Name of the model
            task: Name of the task, or None for general configuration
            
        Returns:
            Configuration dictionary
        """
        # Start with default config
        if model_name in cls._default_configs:
            config = cls._default_configs[model_name].copy()
        else:
            logger.warning(f"Model {model_name} not found, using default config")
            config = cls._default_configs["default"].copy()
        
        # Apply task-specific overrides if applicable
        if task and task in cls._task_configs:
            task_config = cls._task_configs[task]
            if model_name in task_config:
                config.update(task_config[model_name])
            elif "default" in task_config:
                config.update(task_config["default"])
        
        return config
    
    @classmethod
    def load_configs(cls, config_path: str = None) -> None:
        """
        Load configurations from a JSON file.
        
        Args:
            config_path: Path to the configuration file
        """
        if not config_path:
            config_path = os.path.join(os.path.dirname(__file__), "config", "model_config.json")
        
        try:
            with open(config_path, "r") as f:
                configs = json.load(f)
            
            # Update default configs
            if "default_configs" in configs:
                cls._default_configs.update(configs["default_configs"])
            
            # Update task configs
            if "task_configs" in configs:
                for task, task_config in configs["task_configs"].items():
                    if task not in cls._task_configs:
                        cls._task_configs[task] = {}
                    cls._task_configs[task].update(task_config)
            
            logger.info(f"Loaded model configurations from {config_path}")
        except Exception as e:
            logger.warning(f"Error loading model configurations: {e}")
    
    @classmethod
    def save_configs(cls, config_path: str = None) -> None:
        """
        Save configurations to a JSON file.
        
        Args:
            config_path: Path to the configuration file
        """
        if not config_path:
            config_path = os.path.join(os.path.dirname(__file__), "config", "model_config.json")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        configs = {
            "default_configs": cls._default_configs,
            "task_configs": cls._task_configs
        }
        
        try:
            with open(config_path, "w") as f:
                json.dump(configs, f, indent=2)
            
            logger.info(f"Saved model configurations to {config_path}")
        except Exception as e:
            logger.error(f"Error saving model configurations: {e}") 