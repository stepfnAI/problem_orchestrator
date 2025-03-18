import argparse
import json
import os
import logging
import pandas as pd
from datetime import datetime
from autonomous_framework.agent_registry import AgentRegistry
from autonomous_framework.context_manager import ContextManager
from autonomous_framework.execution_engine import ExecutionEngine
from autonomous_framework.agent_interface import AgentMetadata, AgentCapability, AgentCategory
import numpy as np

# Import existing agents
from orchestrator.agents.mapping_agent import SFNDataMappingAgent
from orchestrator.agents.join_suggestion_agent import SFNJoinSuggestionAgent
from orchestrator.agents.temp_aggregation_agent import SFNAggregationAgent
from orchestrator.agents.target_generator_agent import SFNTargetGeneratorAgent
from orchestrator.agents.clustering_agent import SFNClusteringAgent
from orchestrator.agents.clustering_strategy_selector import SFNClusterSelectionAgent
from orchestrator.agents.reco_approach_selection_agent import SFNApproachSelectionAgent
from orchestrator.agents.reco_feature_suggestion_agent import SFNFeatureSuggestionAgent
from orchestrator.agents.reco_explanation_agent import SFNRecommendationExplanationAgent
from orchestrator.config.model_config import DEFAULT_LLM_PROVIDER
from autonomous_framework.agents.data_loading_agent import DataLoadingAgent
from autonomous_framework.agents.data_analysis_agent import DataAnalysisAgent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('autonomous_framework.log')
    ]
)
logger = logging.getLogger(__name__)

def register_existing_agents(registry: AgentRegistry):
    """Register existing agents from the orchestrator"""
    
    # Register Mapping Agent
    registry.register(
        SFNDataMappingAgent,
        AgentMetadata(
            name="field_mapper",
            description="Maps columns in a dataset to standard field names based on problem type",
            category=AgentCategory.DATA_TRANSFORMATION,
            output_type="DIRECT",  # Direct mapping output
            capabilities=[
                AgentCapability(
                    name="map_columns",
                    description="Maps columns to standard field names",
                    required_inputs=["df", "problem_type"],
                    optional_inputs=[],
                    output_schema={
                        "field_name": "column_name"
                    },
                    example_use_case="Mapping raw column names to standard field names"
                )
            ]
        )
    )
    
    # Register Join Suggestion Agent
    registry.register(
        SFNJoinSuggestionAgent,
        AgentMetadata(
            name="join_suggester",
            description="Suggests the best way to join multiple tables",
            category=AgentCategory.JOIN,
            output_type="ADVISORY",  # Advisory output
            capabilities=[
                AgentCapability(
                    name="suggest_join",
                    description="Suggests join strategy between tables",
                    required_inputs=["available_tables", "tables_metadata"],
                    optional_inputs=[],
                    output_schema={
                        "tables_to_join": ["table1", "table2"],
                        "type_of_join": "inner/left/right/outer",
                        "joining_fields": [{"table1": "field1", "table2": "field2"}],
                        "explanation": "Explanation of join strategy"
                    },
                    example_use_case="Joining customer and transaction tables"
                )
            ]
        )
    )
    
    # Register Aggregation Agent
    registry.register(
        SFNAggregationAgent,
        AgentMetadata(
            name="aggregation_advisor",
            description="Suggests and performs data aggregation",
            category=AgentCategory.AGGREGATION,
            output_type="ADVISORY",  # Advisory output
            capabilities=[
                AgentCapability(
                    name="suggest_aggregation",
                    description="Suggests aggregation strategy for a table",
                    required_inputs=["table", "mapping_columns"],
                    optional_inputs=[],
                    output_schema={
                        "aggregation_needed": True,
                        "groupby_columns": ["col1", "col2"],
                        "aggregation_functions": {"col3": "sum", "col4": "mean"}
                    },
                    example_use_case="Aggregating transaction data to customer level"
                )
            ]
        )
    )
    
    # Register Target Generator Agent
    registry.register(
        SFNTargetGeneratorAgent,
        AgentMetadata(
            name="target_generator",
            description="Generates target column based on user instructions",
            category=AgentCategory.TARGET_GENERATION,
            output_type="EXECUTABLE",  # Executable code output
            capabilities=[
                AgentCapability(
                    name="generate_target",
                    description="Generates code to create target column",
                    required_inputs=["user_instructions", "df", "problem_type"],
                    optional_inputs=[{"name": "error_message", "description": "Error from previous attempt"}],
                    output_schema={
                        "code": "Python code to generate target",
                        "explanation": "Explanation of the code",
                        "preview": "Preview of the target column"
                    },
                    example_use_case="Creating churn target based on last transaction date"
                )
            ]
        )
    )
    
    # Register Data Loading Agent
    registry.register(
        DataLoadingAgent,
        AgentMetadata(
            name="data_loader",
            description="Loads data from various sources",
            category=AgentCategory.DATA_LOADING,
            capabilities=[
                AgentCapability(
                    name="load_data",
                    description="Loads data from a source",
                    required_inputs=["source_path"],
                    optional_inputs=[
                        {"name": "source_type", "description": "Type of source (csv, excel, database)"},
                        {"name": "options", "description": "Additional options for loading"}
                    ],
                    output_schema={
                        "df": "Loaded DataFrame",
                        "metadata": "Metadata about the loaded data"
                    },
                    example_use_case="Loading customer data from a CSV file"
                )
            ]
        )
    )
    
    # Register Data Analysis Agent
    registry.register(
        DataAnalysisAgent,
        AgentMetadata(
            name="data_analyzer",
            description="Analyzes datasets to understand their structure, relationships, and quality",
            category=AgentCategory.DATA_ANALYSIS,
            capabilities=[
                AgentCapability(
                    name="basic_analysis",
                    description="Performs basic analysis of a dataset",
                    required_inputs=["df"],
                    optional_inputs=[
                        {"name": "analysis_type", "description": "Type of analysis to perform (basic, detailed, relationships)"},
                        {"name": "options", "description": "Additional options for analysis"}
                    ],
                    output_schema={
                        "basic_info": "Basic information about the dataset",
                        "missing_values": "Analysis of missing values",
                        "duplicates": "Analysis of duplicate rows",
                        "column_types": "Categorization of columns by data type"
                    },
                    example_use_case="Analyzing a customer dataset to understand its structure"
                ),
                AgentCapability(
                    name="detailed_analysis",
                    description="Performs detailed analysis of a dataset",
                    required_inputs=["df"],
                    optional_inputs=[
                        {"name": "options", "description": "Additional options for analysis"}
                    ],
                    output_schema={
                        "basic_info": "Basic information about the dataset",
                        "missing_values": "Analysis of missing values",
                        "duplicates": "Analysis of duplicate rows",
                        "column_types": "Categorization of columns by data type",
                        "numeric_stats": "Detailed statistics for numeric columns",
                        "categorical_stats": "Detailed statistics for categorical columns",
                        "datetime_stats": "Detailed statistics for datetime columns",
                        "potential_id_columns": "Columns that might be IDs or keys"
                    },
                    example_use_case="Performing detailed analysis of customer transaction data"
                ),
                AgentCapability(
                    name="relationship_analysis",
                    description="Analyzes relationships between columns in a dataset",
                    required_inputs=["df"],
                    optional_inputs=[
                        {"name": "options", "description": "Additional options for analysis"}
                    ],
                    output_schema={
                        "correlation": "Correlation analysis for numeric columns",
                        "categorical_relationships": "Relationships between categorical columns",
                        "potential_foreign_keys": "Columns that might be foreign keys"
                    },
                    example_use_case="Identifying relationships between customer and transaction tables"
                )
            ]
        )
    )
    
    # Register more agents as needed...

def setup_output_directory(output_dir):
    """Set up output directory for results"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    # Create a timestamped subdirectory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir)
    logger.info(f"Created run directory: {run_dir}")
    
    return run_dir

def save_workflow_results(results, run_dir):
    """Save workflow results to a file"""
    # Create a custom JSON encoder to handle NumPy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient='records')
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            return super(NumpyEncoder, self).default(obj)
    
    # Save results to file
    with open(os.path.join(run_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    # Print summary
    print("\n" + "=" * 80)
    print("WORKFLOW SUMMARY")
    print("=" * 80 + "\n")
    
    # Print steps
    print(f"Completed {len(results['steps'])} steps")
    print(f"Status: {'Completed successfully' if results['completed'] else 'Failed or incomplete'}\n")
    
    print("Workflow steps:")
    for i, step in enumerate(results['steps']):
        print(f"{i+1}. {step['agent']} -> {step.get('next_agent')}")
        print(f"   Reasoning: {step.get('reasoning', '')[:100]}...")
    
    print("\nFinal output:")
    print(json.dumps(results['final_output'], indent=2, cls=NumpyEncoder))
    
    print("\n" + "=" * 80)

def print_workflow_summary(results):
    """Print a summary of the workflow results"""
    # Create a custom JSON encoder to handle NumPy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient='records')
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            return super(NumpyEncoder, self).default(obj)
    
    print("\n" + "="*80)
    print(f"WORKFLOW SUMMARY")
    print("="*80)
    
    print(f"\nCompleted {len(results['steps'])} steps")
    print(f"Status: {'Completed successfully' if results['completed'] else 'Failed or incomplete'}")
    
    print("\nWorkflow steps:")
    for i, step in enumerate(results['steps']):
        print(f"{i+1}. {step['agent']} -> {step.get('next_agent', 'END')}")
        print(f"   Reasoning: {step['reasoning'][:100]}..." if len(step['reasoning']) > 100 else f"   Reasoning: {step['reasoning']}")
    
    if results.get('final_output'):
        print("\nFinal output:")
        print(json.dumps(results['final_output'], indent=2, cls=NumpyEncoder))
    
    print("\n" + "="*80)

def load_data(args, context_manager):
    """Load data from file"""
    if args.data_path:
        try:
            # Load data
            df = pd.read_csv(args.data_path)
            print(f"Loaded data from {args.data_path}: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Get table name from path
            table_name = os.path.basename(args.data_path)
            
            # Store in context
            context_manager.store_table(table_name, df)
            
            # Also set as current dataframe
            context_manager.memory["current_dataframe"] = df
            
            print(f"Added table '{table_name}' to context with {df.shape[0]} rows and {df.shape[1]} columns")
            logger.info(f"Added table '{table_name}' to context")
            
            return df
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            logger.error(f"Error loading data: {str(e)}")
    return None

def print_progress(context_manager):
    """Print workflow progress and results"""
    current_state = context_manager.get_workflow_summary()
    
    print("\n=== Workflow Progress ===")
    print(f"Goal: {current_state['goal']}")
    print(f"Completed Steps: {len(current_state['completed_steps'])}")
    
    # If we have a dataframe with target, show preview
    df = context_manager.get_current_dataframe()
    if df is not None and 'target' in df.columns:
        print("\n=== Target Column Preview ===")
        print("\nFirst few rows:")
        print(df[['target']].head())
        print("\nTarget Distribution:")
        print(df['target'].value_counts())

def run_workflow(goal: str, data_path: str):
    # Initialize components
    context_manager = ContextManager()
    agent_registry = AgentRegistry()
    execution_engine = ExecutionEngine(agent_registry, context_manager)
    
    # Load initial data
    df = pd.read_csv(data_path)
    context_manager.update_dataframe(df)
    
    while True:
        # Get next agent decision
        meta_agent = MetaAgent(context_manager, agent_registry)
        next_action = meta_agent.decide_next_action()
        
        if not next_action:
            break
            
        # Execute the agent
        agent = agent_registry.get_agent(next_action['agent_name'])
        if not agent:
            break
            
        # Get agent output
        output = agent.execute_task(next_action['task'])
        
        # Store output in context
        context_manager.store_agent_output(next_action['agent_name'], output)
        
        # Execute output if needed (like target generation code)
        execution_result = execution_engine.execute_agent_output(
            next_action['agent_name'], 
            output
        )
        
        if execution_result:
            # Store execution results in context
            context_manager.store_execution_result(
                next_action['agent_name'],
                execution_result
            )
            
        # Print progress
        print_progress(context_manager)

def main():
    parser = argparse.ArgumentParser(description="Autonomous Agent Framework")
    parser.add_argument("--goal", type=str, required=True, help="Goal for the workflow (e.g., 'churn prediction')")
    parser.add_argument("--data-path", type=str, help="Path to input data file (CSV, Excel, Parquet)")
    parser.add_argument("--session", type=str, help="Session ID to resume")
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum number of steps to run")
    parser.add_argument("--output-dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--list-agents", action="store_true", help="List available agents and exit")
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create agent registry
    registry = AgentRegistry()
    
    # Register existing agents
    register_existing_agents(registry)
    
    # List agents and exit if requested
    if args.list_agents:
        print("\nAvailable Agents:")
        for agent_name in registry.list_agents():
            metadata = registry.get_agent_metadata(agent_name)
            print(f"- {agent_name}: {metadata.description}")
            print(f"  Category: {metadata.category.value}")
            print(f"  Capabilities:")
            for capability in metadata.capabilities:
                print(f"    - {capability.name}: {capability.description}")
            print()
        return
    
    # Create context manager
    context_manager = ContextManager(args.session)
    
    # Load data if provided
    if args.data_path:
        df = load_data(args, context_manager)
        if df is not None:
            # Add to context manager
            file_name = os.path.basename(args.data_path)
            context_manager.store_table(file_name, df)
            logger.info(f"Added table '{file_name}' to context")
    
    # Create execution engine
    engine = ExecutionEngine(registry, context_manager)
    
    # Set up output directory
    run_dir = setup_output_directory(args.output_dir)
    
    if args.interactive:
        # Interactive mode
        print(f"\nStarting interactive session for goal: {args.goal}")
        context_manager.set_current_goal(args.goal)
        
        steps = []
        while True:
            user_input = input("\nEnter input (or 'quit' to exit): ")
            if user_input.lower() == 'quit':
                break
                
            result = engine.run_step(user_input=user_input)
            steps.append(result)
            
            print(f"\nAgent: {result['agent']}")
            print(f"Reasoning: {result['reasoning']}")
            print(f"Output: {json.dumps(result['output'], indent=2)}")
            
            if not result.get('next_agent'):
                print("\nWorkflow complete or error occurred.")
                break
        
        # Save results
        workflow_results = {
            'steps': steps,
            'final_output': steps[-1]['output'] if steps else None,
            'completed': len(steps) > 0 and 'error' not in steps[-1].get('output', {})
        }
        save_workflow_results(workflow_results, run_dir)
        print_workflow_summary(workflow_results)
    else:
        # Run full workflow
        logger.info(f"Starting workflow with goal: {args.goal}")
        result = engine.run_workflow(args.goal, args.max_steps)
        
        # Save results
        save_workflow_results(result, run_dir)
        
        # Print summary
        print_workflow_summary(result)
        
        logger.info(f"Workflow completed with {len(result['steps'])} steps")

if __name__ == "__main__":
    main() 