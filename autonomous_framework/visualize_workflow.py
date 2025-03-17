import argparse
import json
import os
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap

def load_workflow(results_file):
    """Load workflow results from file"""
    with open(results_file, 'r') as f:
        return json.load(f)

def create_workflow_graph(workflow_results):
    """Create a directed graph of the workflow"""
    G = nx.DiGraph()
    
    # Add nodes and edges
    for i, step in enumerate(workflow_results['steps']):
        agent_name = step['agent']
        next_agent = step.get('next_agent')
        
        # Add current agent node
        G.add_node(agent_name, step=i+1)
        
        # Add edge to next agent if there is one
        if next_agent:
            G.add_edge(agent_name, next_agent)
    
    return G

def visualize_workflow(G, output_file=None):
    """Visualize the workflow graph"""
    plt.figure(figsize=(12, 8))
    
    # Create a custom colormap
    colors = [(0.8, 0.8, 1.0), (0.2, 0.2, 0.8)]
    cmap = LinearSegmentedColormap.from_list('agent_cmap', colors, N=100)
    
    # Get positions for nodes
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color=list(range(len(G.nodes))), 
                          cmap=cmap, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=2, edge_color='gray', arrowsize=20)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Add step numbers to nodes
    node_labels = {node: f"Step {G.nodes[node]['step']}" for node in G.nodes}
    pos_labels = {k: (v[0], v[1]-0.1) for k, v in pos.items()}
    nx.draw_networkx_labels(G, pos_labels, labels=node_labels, font_size=8)
    
    plt.title("Workflow Execution Graph", fontsize=16)
    plt.axis('off')
    
    if output_file:
        plt.savefig(output_file, bbox_inches='tight')
        print(f"Saved workflow visualization to {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize Autonomous Workflow")
    parser.add_argument("results_file", type=str, help="Path to workflow_results.json file")
    parser.add_argument("--output", type=str, help="Output file path for visualization (PNG, PDF, SVG)")
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.results_file):
        print(f"Error: Results file not found: {args.results_file}")
        return
    
    # Load workflow results
    workflow_results = load_workflow(args.results_file)
    
    # Create graph
    G = create_workflow_graph(workflow_results)
    
    # Visualize
    visualize_workflow(G, args.output)

if __name__ == "__main__":
    main() 