import argparse
import json
import os
import pandas as pd
from collections import Counter

def load_workflow(results_file):
    """Load workflow results from file"""
    with open(results_file, 'r') as f:
        return json.load(f)

def analyze_agent_usage(workflow_results):
    """Analyze agent usage in the workflow"""
    agents = [step['agent'] for step in workflow_results['steps']]
    agent_counts = Counter(agents)
    
    print("\nAgent Usage:")
    for agent, count in agent_counts.most_common():
        print(f"- {agent}: {count} times ({count/len(agents)*100:.1f}%)")
    
    return agent_counts

def analyze_transitions(workflow_results):
    """Analyze transitions between agents"""
    transitions = []
    for i in range(len(workflow_results['steps']) - 1):
        current = workflow_results['steps'][i]['agent']
        next_agent = workflow_results['steps'][i]['next_agent']
        if next_agent:
            transitions.append((current, next_agent))
    
    transition_counts = Counter(transitions)
    
    print("\nAgent Transitions:")
    for (source, target), count in transition_counts.most_common():
        print(f"- {source} -> {target}: {count} times")
    
    return transition_counts

def analyze_workflow_efficiency(workflow_results):
    """Analyze workflow efficiency"""
    steps = workflow_results['steps']
    completed = workflow_results['completed']
    
    print("\nWorkflow Efficiency:")
    print(f"- Total steps: {len(steps)}")
    print(f"- Completed successfully: {completed}")
    
    # Check for repeated agent calls
    agents = [step['agent'] for step in steps]
    repeated = [agent for agent, count in Counter(agents).items() if count > 1]
    
    if repeated:
        print(f"- Agents called multiple times: {', '.join(repeated)}")
    else:
        print("- No agents were called multiple times")
    
    # Check for potential loops
    transitions = [(steps[i]['agent'], steps[i+1]['agent']) for i in range(len(steps)-1)]
    transition_counts = Counter(transitions)
    potential_loops = [f"{source}->{target}" for (source, target), count in transition_counts.items() if count > 1]
    
    if potential_loops:
        print(f"- Potential loops detected: {', '.join(potential_loops)}")
    else:
        print("- No potential loops detected")

def export_to_csv(workflow_results, output_file):
    """Export workflow steps to CSV"""
    steps_data = []
    for i, step in enumerate(workflow_results['steps']):
        steps_data.append({
            'step': i+1,
            'agent': step['agent'],
            'next_agent': step.get('next_agent', 'END'),
            'reasoning': step['reasoning']
        })
    
    df = pd.DataFrame(steps_data)
    df.to_csv(output_file, index=False)
    print(f"\nExported workflow steps to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Analyze Autonomous Workflow")
    parser.add_argument("results_file", type=str, help="Path to workflow_results.json file")
    parser.add_argument("--export-csv", type=str, help="Export analysis to CSV file")
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.results_file):
        print(f"Error: Results file not found: {args.results_file}")
        return
    
    # Load workflow results
    workflow_results = load_workflow(args.results_file)
    
    print(f"\nAnalyzing workflow from {args.results_file}")
    print("="*80)
    
    # Perform analysis
    analyze_agent_usage(workflow_results)
    analyze_transitions(workflow_results)
    analyze_workflow_efficiency(workflow_results)
    
    # Export to CSV if requested
    if args.export_csv:
        export_to_csv(workflow_results, args.export_csv)

if __name__ == "__main__":
    main() 