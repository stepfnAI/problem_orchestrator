import json
import os

def update_prompt_config():
    """Update the prompt_config.json file with meta_agent prompts"""
    
    # Path to the original prompt config
    parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
    prompt_config_path = os.path.join(parent_path, 'orchestrator', 'config', 'prompt_config.json')
    
    # Path to the additional prompts
    addition_path = os.path.join(parent_path, 'orchestrator', 'config', 'prompt_config_addition.json')
    
    # Load the original config
    with open(prompt_config_path, 'r') as f:
        prompt_config = json.load(f)
    
    # Load the additional prompts
    with open(addition_path, 'r') as f:
        addition_config = json.load(f)
    
    # Merge the configs
    prompt_config.update(addition_config)
    
    # Write the updated config back to the file
    with open(prompt_config_path, 'w') as f:
        json.dump(prompt_config, f, indent=4)
    
    print(f"Updated {prompt_config_path} with meta_agent prompts")

if __name__ == "__main__":
    update_prompt_config() 