from typing import Dict, List, Type, Optional
import inspect
from sfn_blueprint import SFNAgent
from autonomous_framework.agent_interface import AgentMetadata

class AgentRegistry:
    """Registry for all available agents"""
    
    def __init__(self):
        self._agents: Dict[str, Dict] = {}
        
    def register(self, agent_class: Type[SFNAgent], metadata: AgentMetadata):
        """Register an agent with its metadata"""
        self._agents[metadata.name] = {
            'class': agent_class,
            'metadata': metadata
        }
        
    def get_agent_class(self, agent_name: str) -> Optional[Type[SFNAgent]]:
        """Get agent class by name"""
        agent_info = self._agents.get(agent_name)
        return agent_info['class'] if agent_info else None
        
    def get_agent_metadata(self, agent_name: str) -> Optional[AgentMetadata]:
        """Get agent metadata by name"""
        agent_info = self._agents.get(agent_name)
        return agent_info['metadata'] if agent_info else None
    
    def list_agents(self) -> List[str]:
        """List all registered agents"""
        return list(self._agents.keys())
    
    def list_agents_by_category(self, category: str) -> List[str]:
        """List agents by category"""
        return [
            name for name, info in self._agents.items() 
            if info['metadata'].category.value == category
        ]
    
    def get_agent_info_for_llm(self) -> List[Dict]:
        """Get agent information formatted for LLM consumption"""
        agent_info = []
        for name, info in self._agents.items():
            metadata = info['metadata']
            capabilities = []
            for cap in metadata.capabilities:
                capabilities.append({
                    'name': cap.name,
                    'description': cap.description,
                    'required_inputs': cap.required_inputs,
                    'output_schema': cap.output_schema
                })
            
            agent_info.append({
                'name': name,
                'description': metadata.description,
                'category': metadata.category.value,
                'capabilities': capabilities,
                'dependencies': metadata.dependencies
            })
        return agent_info 

    def get_agent(self, agent_name: str) -> Optional[SFNAgent]:
        """Get an instance of an agent by name
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Instance of the agent or None if not found
        """
        agent_class = self.get_agent_class(agent_name)
        if agent_class:
            try:
                # Instantiate the agent
                return agent_class()
            except Exception as e:
                print(f"Error instantiating agent {agent_name}: {str(e)}")
                return None
        return None 