#!/usr/bin/env python
"""
Command-line interface for the Meta-Agent framework.
"""

import argparse
import logging
import json
import os
import sys
from typing import Dict, Any

from meta_agent.meta_agent import MetaAgent
from utils.logging_utils import setup_logging, create_timed_rotating_log_file
from services.state_manager import StateManager
import pandas as pd

def setup_parser() -> argparse.ArgumentParser:
    """Set up the argument parser."""
    parser = argparse.ArgumentParser(description="Meta-Agent CLI")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Initialize problem command
    init_parser = subparsers.add_parser("init", help="Initialize a new problem")
    init_parser.add_argument("description", help="Problem description")
    init_parser.add_argument("--model", default="default", help="Model to use for agents")
    
    # Load data command
    load_parser = subparsers.add_parser("load", help="Load data into the system")
    load_parser.add_argument("file_path", help="Path to the data file")
    load_parser.add_argument("--name", help="Name for the loaded table")
    
    # Load multiple data files command
    load_multiple_parser = subparsers.add_parser("load-multiple", help="Load multiple data files")
    load_multiple_parser.add_argument("file_paths", nargs='+', help="Paths to data files")
    load_multiple_parser.add_argument("--prefix", help="Prefix for table names")
    
    # Execute flow command
    flow_parser = subparsers.add_parser("flow", help="Execute a flow")
    flow_parser.add_argument("flow_id", help="ID of the flow to execute")
    flow_parser.add_argument("--input", help="Input tables (comma-separated)")
    flow_parser.add_argument("--params", help="Flow parameters (JSON string)")
    
    # Show state command
    state_parser = subparsers.add_parser("state", help="Show current state")
    state_parser.add_argument("--tables", action="store_true", help="Show tables")
    state_parser.add_argument("--flows", action="store_true", help="Show flows")
    state_parser.add_argument("--lineage", action="store_true", help="Show data lineage")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run the entire workflow")
    run_parser.add_argument("--goal", required=True, help="Problem description/goal")
    run_parser.add_argument("--data-path", nargs='+', required=True, help="Path(s) to data file(s)")
    run_parser.add_argument("--model", default="default", help="Model to use for agents")
    run_parser.add_argument("--interactive", action="store_true", help="Enable interactive mode")
    
    # Common arguments
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Logging level")
    parser.add_argument("--output", help="Output file for results")
    
    return parser

def initialize_problem(args: argparse.Namespace) -> None:
    """Initialize a new problem."""
    config = {
        "model_name": args.model
    }
    
    meta_agent = MetaAgent(config)
    meta_agent.initialize_problem(args.description)
    
    print(f"Problem initialized: {args.description}")
    print(f"Problem type: {meta_agent.problem_context.get('problem_type', 'unknown')}")

def load_data(args: argparse.Namespace) -> None:
    """Load data into the system."""
    state_manager = StateManager()
    
    # Determine table name
    table_name = args.name
    if not table_name:
        base_name = os.path.basename(args.file_path)
        table_name = os.path.splitext(base_name)[0]
    
    # In a real implementation, this would load and process the data file
    # For now, just create a dummy table with a simple schema
    table_schema = {
        "customer_id": "string",
        "product_name": "string",
        "purchase_date": "date",
        "amount": "numeric"
    }
    
    # Update state with the new table
    state = state_manager.get_state()
    if "tables" not in state:
        state["tables"] = {}
    
    state["tables"][table_name] = {
        "schema": table_schema,
        "source_file": args.file_path,
        "row_count": 1000,  # Dummy value
        "flow_history": [
            {
                "flow_id": "data_loading",
                "timestamp": state_manager._get_timestamp(),
                "summary": f"Loaded data from {args.file_path}"
            }
        ]
    }
    
    state_manager.update_state(state)
    
    print(f"Data loaded from {args.file_path} as table '{table_name}'")
    print(f"Schema: {json.dumps(table_schema, indent=2)}")

def load_multiple_data(args: argparse.Namespace) -> None:
    """Load multiple data files into the system."""
    state_manager = StateManager()
    
    for i, file_path in enumerate(args.file_paths):
        # Determine table name
        base_name = os.path.basename(file_path)
        table_name = os.path.splitext(base_name)[0]
        
        if args.prefix:
            table_name = f"{args.prefix}_{table_name}"
        
        # In a real implementation, this would load and process the data file
        # For now, just create a dummy table with a simple schema
        table_schema = {
            "customer_id": "string",
            "product_name": "string",
            "purchase_date": "date",
            "amount": "numeric"
        }
        
        # Update state with the new table
        state = state_manager.get_state()
        if "tables" not in state:
            state["tables"] = {}
        
        state["tables"][table_name] = {
            "schema": table_schema,
            "source_file": file_path,
            "row_count": 1000,  # Dummy value
            "flow_history": [
                {
                    "flow_id": "data_loading",
                    "timestamp": state_manager._get_timestamp(),
                    "summary": f"Loaded data from {file_path}"
                }
            ]
        }
        
        state_manager.update_state(state)
        print(f"Loaded data from {file_path} as table '{table_name}'")
    
    print(f"Successfully loaded {len(args.file_paths)} data files")

def execute_flow(args: argparse.Namespace) -> None:
    """Execute a flow."""
    # Get current state
    state_manager = StateManager()
    current_state = state_manager.get_state()
    
    # Get problem context
    problem_context = current_state.get("problem_context", {})
    
    # Parse input tables
    input_tables = []
    if args.input:
        input_tables = args.input.split(",")
    elif "tables" in current_state:
        # Use all available tables if none specified
        input_tables = list(current_state.get("tables", {}).keys())
    
    # Parse flow parameters
    flow_params = {}
    if args.params:
        try:
            flow_params = json.loads(args.params)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in flow parameters: {args.params}")
            return
    
    # Prepare input data for the flow
    input_data = {
        "problem_context": problem_context,
        "current_state": current_state,
        "input_tables": input_tables,
        "parameters": flow_params
    }
    
    # Create meta agent and execute flow
    config = {"model_name": "default"}
    meta_agent = MetaAgent(config)
    
    # Execute the flow
    print(f"Executing flow: {args.flow_id}")
    print(f"Input tables: {', '.join(input_tables)}")
    
    result = meta_agent.flow_manager.execute_flow(args.flow_id, input_data)
    
    # Update state with flow results
    if "tables" in result:
        if "tables" not in current_state:
            current_state["tables"] = {}
        current_state["tables"].update(result["tables"])
        state_manager.update_state(current_state)
    
    # Print results
    print(f"Flow execution completed with status: {result.get('status', 'unknown')}")
    print(f"Summary: {result.get('summary', 'No summary available')}")
    
    # Output to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")

def show_state(args: argparse.Namespace) -> None:
    """Show current state."""
    state_manager = StateManager()
    state = state_manager.get_state()
    
    if not state:
        print("No state found. Initialize a problem first.")
        return
    
    print("Current State:")
    print(f"Problem: {state.get('problem_context', {}).get('description', 'No problem initialized')}")
    print(f"Problem type: {state.get('problem_context', {}).get('problem_type', 'unknown')}")
    
    if args.tables and "tables" in state:
        print("\nTables:")
        for table_name, table_info in state["tables"].items():
            print(f"  - {table_name}")
            if "schema" in table_info:
                print(f"    Schema: {len(table_info['schema'])} fields")
            if "row_count" in table_info:
                print(f"    Rows: {table_info['row_count']}")
            if "flow_history" in table_info:
                last_flow = table_info["flow_history"][-1]
                print(f"    Last operation: {last_flow.get('summary', 'unknown')}")
    
    if args.flows and "flow_history" in state:
        print("\nFlow History:")
        for flow in state["flow_history"]:
            print(f"  - {flow.get('timestamp', '')}: {flow.get('flow_id', 'unknown')} - {flow.get('summary', 'No summary')}")
    
    if args.lineage:
        print("\nData Lineage:")
        # In a real implementation, this would show the data lineage graph
        print("  Data lineage visualization not implemented in CLI")

def run_workflow(args: argparse.Namespace) -> None:
    """Run the entire workflow."""
    print(f"Starting workflow for goal: {args.goal}")
    
    # 1. Initialize the problem
    config = {
        "model_name": args.model,
        "interactive": args.interactive
    }
    
    meta_agent = MetaAgent(config)
    meta_agent.initialize_problem(args.goal)
    
    problem_type = meta_agent.problem_context.get('problem_type', 'unknown')
    print(f"Problem initialized with type: {problem_type}")
    
    # 2. Load the data files
    state_manager = StateManager()
    input_tables = []
    
    for data_path in args.data_path:
        print(f"\nLoading data from: {data_path}")
        
        # Read the actual CSV file to get schema
        df = pd.read_csv(data_path)
        
        # Determine table name from file path
        base_name = os.path.basename(data_path)
        table_name = os.path.splitext(base_name)[0]
        input_tables.append(table_name)
        
        # Create schema from actual DataFrame columns
        table_schema = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                table_schema[col] = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                table_schema[col] = "date"
            else:
                table_schema[col] = "string"
        
        print("Actual schema from data:")
        print(json.dumps(table_schema, indent=2))
        
        # Update state with the new table
        state = state_manager.get_state()
        if "tables" not in state:
            state["tables"] = {}
        
        state["tables"][table_name] = {
            "schema": table_schema,
            "source_file": data_path,
            "row_count": len(df),
            "flow_history": [
                {
                    "flow_id": "data_loading",
                    "timestamp": state_manager._get_timestamp(),
                    "summary": f"Loaded data from {data_path}"
                }
            ]
        }
        
        state_manager.update_state(state)
        print(f"Data loaded as table '{table_name}'")
    
    # 3. Execute the mapping flow
    print("\n--- Executing Mapping Flow ---")
    mapping_input = {
        "problem_context": meta_agent.problem_context,
        "current_state": state_manager.get_state(),
        "input_tables": input_tables,
        "parameters": {},
        "meta_agent": meta_agent  # Pass the meta agent instance
    }
    
    mapping_result = meta_agent.flow_manager.execute_flow("mapping_flow", mapping_input)
    
    if mapping_result.get("status") != "completed":
        print(f"Mapping flow failed: {mapping_result.get('error', 'Unknown error')}")
        return
    
    # Update state with mapping results
    current_state = state_manager.get_state()
    if "tables" in mapping_result:
        if "tables" not in current_state:
            current_state["tables"] = {}
        current_state["tables"].update(mapping_result["tables"])
        state_manager.update_state(current_state)
    
    mapped_table = mapping_result.get("output_table")
    print(f"Mapping completed: {mapping_result.get('summary')}")
    
    # 4. Execute the feature suggestion flow
    print("\n--- Executing Feature Suggestion Flow ---")
    feature_input = {
        "problem_context": meta_agent.problem_context,
        "current_state": state_manager.get_state(),
        "input_tables": [mapped_table],
        "parameters": {},
        "meta_agent": meta_agent  # Pass the meta agent instance
    }
    
    feature_result = meta_agent.flow_manager.execute_flow("feature_suggestion_flow", feature_input)
    
    if feature_result.get("status") != "completed":
        print(f"Feature suggestion flow failed: {feature_result.get('error', 'Unknown error')}")
        return
    
    # Update state with feature results
    current_state = state_manager.get_state()
    if "tables" in feature_result:
        if "tables" not in current_state:
            current_state["tables"] = {}
        current_state["tables"].update(feature_result["tables"])
        state_manager.update_state(current_state)
    
    print(f"Feature suggestion completed: {feature_result.get('summary')}")
    
    # 5. Show final results
    print("\n--- Workflow Summary ---")
    print(f"Goal: {args.goal}")
    print(f"Problem type: {problem_type}")
    print(f"Data source: {', '.join(args.data_path)}")
    print(f"Mapping results: {mapping_result.get('summary')}")
    print(f"Feature suggestions: {feature_result.get('summary')}")
    
    # Show top features
    if "tables" in feature_result:
        feature_table = feature_result.get("output_table")
        if feature_table and feature_table in feature_result["tables"]:
            features = feature_result["tables"][feature_table].get("features", [])
            if features:
                print("\nTop suggested features:")
                for i, feature in enumerate(features[:5]):  # Show top 5
                    print(f"  {i+1}. {feature.get('name')} (importance: {feature.get('importance', 'N/A')})")
                    print(f"     {feature.get('description')}")
                    print(f"     Formula: {feature.get('formula')}")

def setup_logging(log_level: str, log_file: str) -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # This will show logs in console too
        ]
    )

def main() -> None:
    """Main entry point for the CLI."""
    parser = setup_parser()
    args = parser.parse_args()
    
    # Set up logging
    log_dir = os.path.join(os.getcwd(), "logs")
    log_file = create_timed_rotating_log_file(log_dir, "meta_agent_cli")
    setup_logging(args.log_level, log_file)
    
    # Execute the requested command
    if args.command == "init":
        initialize_problem(args)
    elif args.command == "load":
        load_data(args)
    elif args.command == "load-multiple":
        load_multiple_data(args)
    elif args.command == "flow":
        execute_flow(args)
    elif args.command == "state":
        show_state(args)
    elif args.command == "run":
        run_workflow(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 