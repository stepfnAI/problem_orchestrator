#!/usr/bin/env python
"""
Main entry point for the Meta-Agent framework.
"""

import sys
import argparse
from cli import setup_parser, run_workflow

def main():
    """Main entry point."""
    # Check if using the old command format
    if len(sys.argv) > 1 and sys.argv[1] == '--goal':
        # Convert the old format to the new format
        new_args = ['run']
        data_paths = []
        
        i = 1
        while i < len(sys.argv):
            if sys.argv[i] == '--goal':
                new_args.extend(['--goal', sys.argv[i+1]])
                i += 2
            elif sys.argv[i] == '--data-path':
                data_paths.append(sys.argv[i+1])
                i += 2
            else:
                new_args.append(sys.argv[i])
                i += 1
        
        # Add all data paths
        if data_paths:
            new_args.append('--data-path')
            new_args.extend(data_paths)
        
        # Replace sys.argv with the new args
        sys.argv = [sys.argv[0]] + new_args
    
    # Use the standard CLI parser
    parser = setup_parser()
    args = parser.parse_args()
    
    if args.command == 'run':
        run_workflow(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 