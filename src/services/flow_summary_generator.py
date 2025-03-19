"""
Flow Summary Generator - Creates concise summaries of flow execution results.
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class FlowSummaryGenerator:
    """
    Creates concise summaries of flow execution results for the Meta Agent.
    
    This service distills complex flow outputs into actionable summaries,
    formats information in a standardized way, and highlights key decisions,
    transformations, and statistics.
    """
    
    def __init__(self):
        """Initialize the Flow Summary Generator."""
        logger.info("Initialized Flow Summary Generator")
    
    def generate_summary(self, flow_id: str, flow_result: Dict[str, Any]) -> str:
        """
        Generate a summary of a flow execution result.
        
        Args:
            flow_id: ID of the flow
            flow_result: Result of the flow execution
            
        Returns:
            Summary string
        """
        if flow_result.get("status") != "completed":
            return f"Flow {flow_id} {flow_result.get('status', 'unknown')}"
        
        # Use flow-specific summary if available
        if "summary" in flow_result:
            return flow_result["summary"]
        
        # Generate a generic summary
        summary_parts = [f"Flow {flow_id} completed successfully"]
        
        if "input_table_name" in flow_result and "output_table_name" in flow_result:
            summary_parts.append(
                f"Transformed {flow_result['input_table_name']} to {flow_result['output_table_name']}"
            )
        
        if "mappings" in flow_result:
            summary_parts.append(f"Applied {len(flow_result['mappings'])} mappings")
        
        if "features" in flow_result:
            summary_parts.append(f"Created {len(flow_result['features'])} features")
        
        if "model" in flow_result:
            model_info = flow_result.get("model", {})
            model_type = model_info.get("type", "unknown")
            metrics = model_info.get("metrics", {})
            
            summary_parts.append(f"Trained {model_type} model")
            
            if metrics:
                metric_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                summary_parts.append(f"Performance metrics: {metric_str}")
        
        return ". ".join(summary_parts) 