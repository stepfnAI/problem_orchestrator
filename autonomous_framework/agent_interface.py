from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

@dataclass
class AgentCapability:
    """Defines what an agent can do"""
    name: str
    description: str
    required_inputs: List[str]
    optional_inputs: List[Dict[str, Any]]
    output_schema: Dict[str, Any]
    example_use_case: str

class AgentCategory(Enum):
    """Categories of agents"""
    DATA_LOADING = "data_loading"
    DATA_ANALYSIS = "data_analysis"
    DATA_CLEANING = "data_cleaning"
    DATA_TRANSFORMATION = "data_transformation"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_SELECTION = "model_selection"
    MODEL_TRAINING = "model_training"
    MODEL_EVALUATION = "model_evaluation"
    INTERPRETATION = "interpretation"
    AGGREGATION = "aggregation"
    JOIN = "join"
    TARGET_GENERATION = "target_generation"

@dataclass
class AgentMetadata:
    """Metadata about an agent"""
    name: str
    description: str
    category: AgentCategory
    capabilities: List[AgentCapability]
    output_type: str = "DIRECT"  # DIRECT, EXECUTABLE, ADVISORY, COMPOSITE
    dependencies: List[str] = None 