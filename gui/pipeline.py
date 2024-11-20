# pymobi/gui/pipeline.py

from typing import Dict, List, Optional, Any
import numpy as np
from pathlib import Path
import json
from ..core.config import PyMoBIConfig
from ..core.data import PyMoBIData

class PipelineManager:
    """Manager for node-based pipeline processing."""
    
    def __init__(self):
        """Initialize pipeline manager."""
        self.nodes = {}
        self.connections = {}
        self.execution_order = []
        self.data_cache = {}
        
    def add_node(self, node_id: int, node: Any) -> None:
        """
        Add a node to the pipeline.
        
        Parameters
        ----------
        node_id : int
            Unique identifier for the node
        node : Any
            Node instance
        """
        self.nodes[node_id] = node
        self._update_execution_order()
        
    def remove_node(self, node_id: int) -> None:
        """
        Remove a node from the pipeline.
        
        Parameters
        ----------
        node_id : int
            Node identifier to remove
        """
        if node_id in self.nodes:
            del self.nodes[node_id]
            
            # Remove connections
            self.connections = {k: v for k, v in self.connections.items() 
                              if k[0] != node_id and k[1] != node_id}
            
            self._update_execution_order()
            
    def connect_nodes(self, from_node: int, to_node: int, 
                     output_name: str, input_name: str) -> None:
        """
        Connect two nodes.
        
        Parameters
        ----------
        from_node : int
            Source node ID
        to_node : int
            Target node ID
        output_name : str
            Name of output from source node
        input_name : str
            Name of input on target node
        """
        self.connections[(from_node, to_node)] = {
            'output': output_name,
            'input': input_name
        }
        self._update_execution_order()
        
    def disconnect_nodes(self, from_node: int, to_node: int) -> None:
        """
        Remove connection between nodes.
        
        Parameters
        ----------
        from_node : int
            Source node ID
        to_node : int
            Target node ID
        """
        if (from_node, to_node) in self.connections:
            del self.connections[(from_node, to_node)]
            self._update_execution_order()
            
    def process_pipeline(self) -> None:
        """Process all nodes in the pipeline in correct order."""
        self.data_cache.clear()
        
        for node_id in self.execution_order:
            try:
                # Get input data
                inputs = self._get_node_inputs(node_id)
                
                # Process node
                outputs = self.nodes[node_id].process(inputs)
                
                # Cache outputs
                self.data_cache[node_id] = outputs
                
            except Exception as e:
                print(f"Error processing node {node_id}: {str(e)}")
                raise
                
    def save_pipeline(self, filename: Path) -> None:
        """
        Save pipeline configuration.
        
        Parameters
        ----------
        filename : Path
            Path to save configuration
        """
        config = {
            'nodes': {
                str(node_id): {
                    'type': node.__class__.__name__,
                    'parameters': node.parameters
                }
                for node_id, node in self.nodes.items()
            },
            'connections': {
                f"{k[0]}-{k[1]}": v 
                for k, v in self.connections.items()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
            
    def load_pipeline(self, filename: Path) -> None:
        """
        Load pipeline configuration.
        
        Parameters
        ----------
        filename : Path
            Path to configuration file
        """
        with open(filename, 'r') as f:
            config = json.load(f)
            
        # Clear current pipeline
        self.nodes.clear()
        self.connections.clear()
        
        # Load nodes
        for node_id, node_config in config['nodes'].items():
            node_class = self._get_node_class(node_config['type'])
            node = node_class()
            node.parameters = node_config['parameters']
            self.nodes[int(node_id)] = node
            
        # Load connections
        for conn_str, conn_config in config['connections'].items():
            from_node, to_node = map(int, conn_str.split('-'))
            self.connections[(from_node, to_node)] = conn_config
            
        self._update_execution_order()
        
    def _update_execution_order(self) -> None:
        """Update node execution order based on connections."""
        # Build dependency graph
        graph = {node_id: [] for node_id in self.nodes}
        for (from_node, to_node) in self.connections:
            graph[to_node].append(from_node)
            
        # Topological sort
        visited = set()
        temp_mark = set()
        order = []
        
        def visit(node_id):
            if node_id in temp_mark:
                raise ValueError("Circular dependency detected")
            if node_id not in visited:
                temp_mark.add(node_id)
                for dep in graph[node_id]:
                    visit(dep)
                temp_mark.remove(node_id)
                visited.add(node_id)
                order.insert(0, node_id)
                
        for node_id in self.nodes:
            if node_id not in visited:
                visit(node_id)
                
        self.execution_order = order
        
    def _get_node_inputs(self, node_id: int) -> Dict[str, Any]:
        """Get input data for a node."""
        inputs = {}
        
        # Find connections where this node is the target
        for (from_node, to_node), conn_config in self.connections.items():
            if to_node == node_id:
                if from_node in self.data_cache:
                    output_data = self.data_cache[from_node][conn_config['output']]
                    inputs[conn_config['input']] = output_data
                    
        return inputs
        
    @staticmethod
    def _get_node_class(node_type: str) -> type:
        """Get node class by type name."""
        # Implementation to get node class
        pass