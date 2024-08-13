from typing import Union, List, Dict, Any
from collections import defaultdict
import networkx as nx
import random

class DiGraph(nx.DiGraph):
    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)
        self.node2edges_cache = dict() # Dynamic cache
        self.type2edges = defaultdict(set)
        
    def sample_subgraph(self, start_nodes:Union[str, List[str]], hop_num:int=1, edge_sample_size:dict=None):
        if isinstance(start_nodes, str):
            start_nodes = [start_nodes]
        new_nodes = start_nodes
        subgraph = DiGraph()
        subgraph.add_nodes_from(start_nodes)
        for _ in range(hop_num):
            current_nodes = set(subgraph.nodes)
            for node in new_nodes:
                edge_dict:dict
                if node not in self.node2edges_cache:
                    edge_dict = defaultdict(list)
                    for child in self.successors(node):
                        edge_dict[self.get_edge_data(node, child)['type']].append((node, child))
                    for parent in self.predecessors(node):
                        edge_dict['inv_' + self.get_edge_data(parent, node)['type']].append((parent, node))
                    self.node2edges_cache[node] = edge_dict
                else:
                    edge_dict = self.node2edges_cache[node]
                if edge_sample_size:
                    for edge_type, sample_size in edge_sample_size.items():
                        if edge_type in edge_dict:
                            if sample_size <= 0:
                                edge_dict.pop(edge_type)
                            elif sample_size < 1:
                                edge_dict[edge_type] = [edge for edge in edge_dict[edge_type] if random.uniform(0, 1) < sample_size]
                                if len(edge_dict[edge_type]) > 10:
                                    edge_dict[edge_type] = random.sample(edge_dict[edge_type], random.randint(1, 10))
                            else:
                                edge_dict[edge_type] = random.sample(edge_dict[edge_type], min(sample_size, len(edge_dict[edge_type])))
                for edge_type, edge_list in edge_dict.items():
                    subgraph.add_edges_from(edge_list, type=edge_type[4:] if edge_type.startswith('inv_') else edge_type)
            new_nodes = set(subgraph.nodes) - current_nodes
        return subgraph
    
    def add_edges_from(self, ebunch_to_add, **attr):
        super().add_edges_from(ebunch_to_add, **attr)
        for u, v, t in self.edges.data("type"):
            if u in self.node2edges_cache:
                del self.node2edges_cache[u]
            if v in self.node2edges_cache:
                del self.node2edges_cache[v]
            self.type2edges[t].add((u, v))
        return
    
    def remove_edges_from(self, ebunch):
        for u, v in ebunch:
            if u in self.node2edges_cache:
                del self.node2edges_cache[u]
            if v in self.node2edges_cache:
                del self.node2edges_cache[v]
            t = self.get_edge_data(u, v)["type"]
            self.type2edges[t].remove((u, v))
        super().remove_edges_from(ebunch)
        return
    
    def get_typed_edges(self, edge_type:str):
        return list(self.type2edges[edge_type])

"""
def graph2data(graph:Union[nx.DiGraph, DiGraph], attribute_map:Dict[str, Any], prefix2type:Dict[str, str]):
    nodes = defaultdict(list)
    # attributes:Dict[str, Any] = {}
    for node in graph.nodes:
        nodes[prefix2type[node[:2]]].append(node[2:])
        # if node in attribute_map:
        #     attributes[node[2:]] = deepcopy(attribute_map[node])
    data_item = {
        'Nodes': nodes,
        # 'Attributes': attributes,
        'Edges': [(u[2:], v[2:], edge_type) for u, v, edge_type in graph.edges.data('type')]
    }
    return data_item

def form_data_item(graph_info:dict, question:str, answer:str):
    return {
        'Question': question,
        'Answer': answer,
        **graph_info
    }
"""