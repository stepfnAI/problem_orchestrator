"""
Strategy module for the Meta Agent.
Contains problem-solving strategies for different problem types.
"""

from typing import Dict, List, Any

class BaseStrategy:
    """Base class for problem-solving strategies."""
    
    def __init__(self, problem_context: Dict[str, Any]):
        self.problem_context = problem_context
    
    def determine_workflow(self) -> List[str]:
        """
        Determine the workflow sequence for the problem.
        
        Returns:
            List of flow IDs representing the workflow sequence
        """
        raise NotImplementedError("Subclasses must implement determine_workflow")


class RegressionStrategy(BaseStrategy):
    """Strategy for regression problems."""
    
    def determine_workflow(self) -> List[str]:
        """Determine workflow for regression problems."""
        return [
            "mapping_flow",
            "feature_suggestion_flow",
            "data_splitting_flow",
            "model_training_flow"
        ]


class ClassificationStrategy(BaseStrategy):
    """Strategy for classification problems."""
    
    def determine_workflow(self) -> List[str]:
        """Determine workflow for classification problems."""
        return [
            "mapping_flow",
            "feature_suggestion_flow",
            "data_splitting_flow",
            "model_training_flow"
        ]


class ClusteringStrategy(BaseStrategy):
    """Strategy for clustering problems."""
    
    def determine_workflow(self) -> List[str]:
        """Determine workflow for clustering problems."""
        return [
            "mapping_flow",
            "feature_suggestion_flow",
            "model_training_flow"
        ]


class ForecastingStrategy(BaseStrategy):
    """Strategy for forecasting problems."""
    
    def determine_workflow(self) -> List[str]:
        """Determine workflow for forecasting problems."""
        return [
            "mapping_flow",
            "feature_suggestion_flow",
            "target_preparation_flow",
            "data_splitting_flow",
            "model_training_flow"
        ]


def get_strategy(problem_type: str, problem_context: Dict[str, Any]) -> BaseStrategy:
    """
    Factory function to get the appropriate strategy for a problem type.
    
    Args:
        problem_type: Type of problem (regression, classification, etc.)
        problem_context: Context information about the problem
        
    Returns:
        Strategy instance for the problem type
    """
    strategies = {
        "regression": RegressionStrategy,
        "classification": ClassificationStrategy,
        "clustering": ClusteringStrategy,
        "forecasting": ForecastingStrategy
    }
    
    strategy_class = strategies.get(problem_type.lower())
    if not strategy_class:
        raise ValueError(f"Unknown problem type: {problem_type}")
    
    return strategy_class(problem_context) 