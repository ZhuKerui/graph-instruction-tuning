import random
from collections import defaultdict, Counter
from typing import Tuple, List, Dict, Any
from tqdm import tqdm
import pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
from networkx import has_path, all_simple_paths, ego_graph, Graph, all_shortest_paths

from graph import DiGraph
from data_utils import *

def build_edge_prediction(G:DiGraph, attribute_type:str, sample_size:int, edge_sample_size:Dict[str, Any]=None):
    # Find candidate pairs
    new_G = DiGraph()
    new_G.add_edges_from([(u, v, {'type': t}) for u, v, t in G.edges.data('type')])
    
    co_edges = G.get_typed_edges(attribute_type)
    head_type = co_edges[0][0][:2]
    tail_type = co_edges[0][1][:2]
    
    # Extract answers and remove answers from graph
    random.shuffle(co_edges)
    pos_samples, neg_samples = [], []
    tqdm_bar = tqdm(total=sample_size)
    for u, v in co_edges:
        if new_G.in_degree(u) + new_G.out_degree(u) > 1 and new_G.in_degree(v) + new_G.out_degree(v) > 1:
            if len(pos_samples) < sample_size // 2:
                pos_samples.append(((u, v), True))
            elif len(neg_samples) < sample_size // 2:
                find_negative = False
                if random.choice([0, 1]) == 0:
                    subgraph = new_G.sample_subgraph(u, 2, edge_sample_size)
                    for n in subgraph.nodes:
                        if n.startswith(tail_type) and not subgraph.has_edge(u, n):
                            neg_samples.append(((u, n), False))
                            find_negative = True
                            break
                else:
                    subgraph = new_G.sample_subgraph(v, 2, edge_sample_size)
                    for n in subgraph.nodes:
                        if n.startswith(head_type) and not subgraph.has_edge(n, v):
                            neg_samples.append(((n, v), False))
                            find_negative = True
                            break
                if not find_negative:
                    continue
            else:
                break
            tqdm_bar.update()
            new_G.remove_edges_from([(u, v)])
    
    # Make train and test splits
    dataset = pos_samples + neg_samples
    dataset = [({'head': edge[0], 'tail': edge[1]}, answer) for edge, answer in dataset]
    random.shuffle(dataset)
    return new_G, dataset

attribute_to_dict = lambda attribute_types: {f'attribute_type_{i}': attribute_type for i, attribute_type in enumerate(attribute_types)}

# ------------------------------ Node tasks ------------------------------

# ---------------------- Single hop ----------------------

# node, edge
def find_neighbor(subgraph:DiGraph, attribute_type:str, class_bin:int=None):
    # Find candidate nodes
    candidate_node_neighbor_pairs = defaultdict(list)
    find_predecessor = attribute_type.startswith('inv_')
    if find_predecessor: attribute_type = attribute_type[4:]
    for u, v, edge_type in subgraph.edges.data('type'):
        if edge_type == attribute_type:
            if not find_predecessor:
                candidate_node_neighbor_pairs[u].append(v)
            else:
                candidate_node_neighbor_pairs[v].append(u)
    
    candidate_nodes = [node for node, neighbors in candidate_node_neighbor_pairs.items() if len(neighbors) == class_bin] if class_bin else list(candidate_node_neighbor_pairs.keys())
    if not candidate_nodes:
        return None
    
    # Build the question
    source = random.choice(candidate_nodes)

    # Find the answer
    answer = candidate_node_neighbor_pairs[source]
    
    return {'source': source, 'attribute_type': attribute_type}, answer

# ---------------------- Multi hop ----------------------

# node, edge
def find_nodes_with_shared_attributes(subgraph:DiGraph, attribute_types:List[str], class_bin:int=None):
    # Find candidate nodes
    edge_type_inverse_pairs = {attribute_type: defaultdict(set) for attribute_type in attribute_types}
    for u, v, edge_type in subgraph.edges.data('type'):
        if edge_type in attribute_types:
            edge_type_inverse_pairs[edge_type][v].add(u)
        elif f'inv_{edge_type}' in attribute_types:
            edge_type_inverse_pairs[f'inv_{edge_type}'][u].add(v)
    inverse_pairs_groups = [list(inverse_pairs.values()) for _, inverse_pairs in edge_type_inverse_pairs.items()]
    
    cur_node_sets = inverse_pairs_groups[0]
    cur_node_sets = {frozenset(node_set) for node_set in cur_node_sets if len(node_set) > 1}
    for next_node_sets in inverse_pairs_groups[1:]:
        next_node_sets = {frozenset(node_set) for node_set in next_node_sets if len(node_set) > 1}
        if not cur_node_sets or not next_node_sets:
            return
        new_cur_node_sets = set()
        for cur_node_set in cur_node_sets:
            for next_node_set in next_node_sets:
                temp_node_set = cur_node_set & next_node_set
                if len(temp_node_set) > 1:
                    new_cur_node_sets.add(temp_node_set)
        cur_node_sets = new_cur_node_sets
    if not cur_node_sets:
        return
    
    cur_node_sets = list(cur_node_sets)
    nodes_with_same_attributes = list({node for node_set in cur_node_sets for node in node_set})

    # Select a target node and build the question
    for source in nodes_with_same_attributes:
        answer_nodes = set()
        for node_set in cur_node_sets:
            if source in node_set:
                answer_nodes.update(node_set)
        if answer_nodes:
            answer_nodes.remove(source)
        
        if class_bin is None or len(answer_nodes) == class_bin:
            return {'source': source, **attribute_to_dict(attribute_types)}, list(answer_nodes)

# node, num
def nodes_within_hops(subgraph:DiGraph, source_type:str, target_type:str, class_bin:int=None):
    source_nodes = [node for node in subgraph.nodes if node.startswith(source_type)]
    if not source_nodes:
        return
    random.shuffle(source_nodes)
    for source in source_nodes:
        hop = random.choice([2,3,4])
        new_subgraph:Graph = ego_graph(subgraph, source, hop, undirected=True)
        nodes:List[str] = [node for node in new_subgraph.nodes if node.startswith(target_type) and node != source]
        if nodes and (class_bin is None or len(nodes) == class_bin):
            return {'source': source, 'hop': hop, 'target_type': target_type}, nodes

# ------------------------------- Pair tasks ------------------------------

# ---------------------- Single hop ----------------------

# node, edge
def find_pairs(subgraph:DiGraph, node_type:str, attribute_types:List[str], class_bin:int=None):
    # Find candidates
    node2pairs = defaultdict(list)
    for u, v, edge_type in subgraph.edges.data('type'):
        if edge_type in attribute_types:
            if u.startswith(node_type):
                node2pairs[u].append((u, v))
            if v.startswith(node_type):
                node2pairs[v].append((u, v))
    if class_bin is not None:
        node2pairs = {node: pairs for node, pairs in node2pairs.items() if len(pairs) == class_bin}
    if not node2pairs:
        return
    
    # Select a target node and build the question
    source = random.choice(list(node2pairs.keys()))
    return {'source': source, **attribute_to_dict(attribute_types)}, node2pairs[source]

# ---------------------- Multi hop ----------------------

def _get_adjacency_matrix_of_edge_types(subgraph:DiGraph, attribute_types:List[str]):
    targeted_pairs = [(u, v) for u, v, edge_type in subgraph.edges.data('type') if edge_type in attribute_types] + [(v, u) for u, v, edge_type in subgraph.edges.data('type') if f'inv_{edge_type}' in attribute_types]
    if not targeted_pairs:
        return None, None, None, None, None
    heads, tails = zip(*targeted_pairs)
    idx2head, idx2tail = list(set(heads)), list(set(tails))
    head2idx, tail2idx = {head: idx for idx, head in enumerate(idx2head)}, {tail: idx for idx, tail in enumerate(idx2tail)}
    
    head2tail_mat = np.zeros((len(head2idx), len(tail2idx)))
    rows, cols = zip(*[(head2idx[u], tail2idx[v]) for u, v in targeted_pairs])
    head2tail_mat[rows, cols] = 1
    
    return head2idx, idx2head, tail2idx, idx2tail, head2tail_mat

# edge, num
def find_pairs_with_n_shared_attributes(subgraph:DiGraph, attribute_types:List[str], class_bin:int=None):
    # Find candidates
    head2idx, idx2head, tail2idx, idx2tail, head2tail_mat = _get_adjacency_matrix_of_edge_types(subgraph, attribute_types)
    if head2tail_mat is None:
        return
    sharing_mat:np.ndarray = np.matmul(head2tail_mat, head2tail_mat.T)
    sharing_mat[np.tri(*sharing_mat.shape, dtype=bool)] = 0
    shared_nums_cnt = Counter(sharing_mat[sharing_mat > 0].tolist())
    shared_nums = [int(num) for num, cnt in shared_nums_cnt.items() if class_bin is None or cnt == class_bin]
    if shared_nums:
        # Select a target node and build the question
        target_num = random.choice(shared_nums)
        
        # Find the answer
        node1s, node2s = np.nonzero(sharing_mat == target_num)
        answer = [(idx2head[node1], idx2head[node2]) for node1, node2 in zip(node1s, node2s)]
        return {'target_num': target_num, **attribute_to_dict(attribute_types)}, answer

# ------------------------------- Count tasks ------------------------------

# ---------------------- Single hop ----------------------

# node, edge
def degree_count(subgraph:DiGraph, attribute_types:List[str], class_bin:int=None):
    # Find candidate nodes
    attribute_dict = defaultdict(set)
    for u, v, edge_type in subgraph.edges.data('type'):
        if edge_type in attribute_types:
            attribute_dict[u].add(v)
        elif f'inv_{edge_type}' in attribute_types:
            attribute_dict[v].add(u)
    if not attribute_dict:
        return None
    
    # Select a source node and build the question
    length_to_source_node = defaultdict(list)
    for source_node, neighbors in attribute_dict.items():
        length_to_source_node[len(neighbors)].append(source_node)
    if class_bin is not None:
        if class_bin not in length_to_source_node:
            return
        answer = class_bin
    else:
        answer = random.choice(list(length_to_source_node.keys()))
    source = random.choice(length_to_source_node[answer])
    return {'source': source, **attribute_to_dict(attribute_types)}, answer

# ---------------------- Multi hop ----------------------

# node, num
def node_num_within_hops(subgraph:DiGraph, source_type:str, target_type:str, class_bin:int=None):
    source_nodes = [node for node in subgraph.nodes if node.startswith(source_type)]
    if not source_nodes:
        return
    random.shuffle(source_nodes)
    for source in source_nodes:
        hop = random.choice([2,3,4])
        new_subgraph:Graph = ego_graph(subgraph, source, hop, undirected=True)
        nodes:List[str] = [node for node in new_subgraph.nodes if node.startswith(target_type) and node != source]
        # if nodes and (class_bin is None or len(nodes) == class_bin):
        return {'source': source, 'hop': hop, 'target_type': target_type}, len(nodes)

# node, num
def count_path_between_nodes(subgraph:DiGraph, source_type:str, target_type:str, class_bin:int=None):
    undirected_subgraph = subgraph.to_undirected()
    source_nodes = [node for node in subgraph.nodes if node.startswith(source_type)]
    target_nodes = [node for node in subgraph.nodes if node.startswith(target_type)]
    if not source_nodes or not target_nodes:
        return
    for source, target in itertools.product(source_nodes, target_nodes):
        if not has_path(undirected_subgraph, source, target) or source == target:
            continue
        hop = random.choice([2,3,4])
        paths = list(all_simple_paths(undirected_subgraph, source, target, hop))
        if paths and (class_bin is None or len(paths) == class_bin):
            return {'source': source, 'target': target, 'hop': hop}, len(paths)

# ------------------------------- Bool tasks ------------------------------

# ---------------------- Single hop ----------------------

# node, edge
def linked_by_edge(subgraph:DiGraph, source_type:str, target_type:str, attribute_type:str, class_bin:int=None):
    edges_with_attribute_type, edges_without_attribute_type, sources, targets = [], [], set(), set()
    for node in subgraph.nodes:
        if node.startswith(source_type):
            sources.add(node)
        if node.startswith(target_type):
            targets.add(node)
    
    for u, v, edge_type in subgraph.edges.data('type'):
        if u.startswith(source_type) and v.startswith(target_type):
            if edge_type == attribute_type:
                edges_with_attribute_type.append((u, v))
            else:
                edges_without_attribute_type.append((u, v))

    is_exist = bool(class_bin) if class_bin else random.choice([True, False])
    target_edge = None
    if is_exist:
        if edges_with_attribute_type:
            target_edge = random.choice(edges_with_attribute_type)
    else:
        if edges_without_attribute_type:
            target_edge = random.choice(edges_without_attribute_type)
        else:
            if sources and targets:
                for source, target in itertools.product(sources, targets):
                    if source != target and (source, target) not in edges_with_attribute_type:
                        target_edge = (source, target)
                        break
                
    if target_edge is not None:
        return {'source': target_edge[0], 'target': target_edge[1], 'edge_type': attribute_type}, is_exist

# ---------------------- Multi hop ----------------------

# node
def has_path_between_nodes(subgraph:DiGraph, source_type:str, target_type:str, class_bin:int=None):
    if class_bin is not None: answer = bool(class_bin)
    else: answer = random.choice([True, False])
    undirected_subgraph = subgraph.to_undirected()
    source_nodes = [node for node in subgraph.nodes if node.startswith(source_type)]
    target_nodes = [node for node in subgraph.nodes if node.startswith(target_type)]
    if not source_nodes or not target_nodes:
        return
    for source, target in itertools.product(source_nodes, target_nodes):
        if source == target or not has_path(undirected_subgraph, source, target) == answer:
            continue
        return {'source': source, 'target': target}, answer

# ------------------------------- Path tasks ------------------------------

# node, num
def find_path_of_length(subgraph:DiGraph, source_type:str, target_type:str, class_bin:int=None):
    undirected_subgraph = subgraph.to_undirected()
    source_nodes = [node for node in subgraph.nodes if node.startswith(source_type)]
    target_nodes = [node for node in subgraph.nodes if node.startswith(target_type)]
    if not source_nodes or not target_nodes:
        return
    for source, target in itertools.product(source_nodes, target_nodes):
        if not has_path(undirected_subgraph, source, target) or source == target:
            continue
        paths = all_simple_paths(undirected_subgraph, source, target, 5)
        hop_map = defaultdict(list)
        for p in paths:
            hop_map[len(p)-1].append(p)
        hops = [hop for hop in hop_map.keys() if hop > 1]
        if not hops:
            continue
        if class_bin is not None:
            path_num_to_hop = {len(hop_map[hop]): hop for hop in hops}
            if class_bin not in path_num_to_hop:
                continue
            hop = path_num_to_hop[class_bin]
        else:
            hop = random.choice(hops)
        return {'source': source, 'target': target, 'hop': hop}, hop_map[hop]

# node
def find_shortest_path(subgraph:DiGraph, source_type:str, target_type:str, class_bin:int=None):
    undirected_subgraph = subgraph.to_undirected()
    source_nodes = [node for node in subgraph.nodes if node.startswith(source_type)]
    target_nodes = [node for node in subgraph.nodes if node.startswith(target_type)]
    if not source_nodes or not target_nodes:
        return
    for source, target in itertools.product(source_nodes, target_nodes):
        if not has_path(undirected_subgraph, source, target) or source == target:
            continue
        paths = list(all_shortest_paths(undirected_subgraph, source, target))
        if class_bin is None or len(paths) == class_bin:
            return {'source': source, 'target': target}, paths

# ------------------------------- Graph tasks ------------------------------

# node, num
def find_ego_graph(subgraph:DiGraph, center_type:str, hop_num:int=1, class_bin:int=None):
    center_nodes = [node for node in subgraph.nodes if node.startswith(center_type)]
    if not center_nodes:
        return
    random.shuffle(center_nodes)
    for center_node in center_nodes:
        subsubgraph:Graph = ego_graph(subgraph, center_node, hop_num, undirected=True)
        answers = list(subsubgraph.edges)
        if class_bin is None or len(answers) == class_bin:
            return {'center': center_node, 'hop': hop_num}, answers

class DatasetMetaData:
    edge_sample_size = None
    attr_sample_ratio = None
    # textwrapper = "<text>{text}</text>"
    NAME = 'name'
    EDGE_ATTRBUTES = []
    ATTRIBUTE_DESCRIPTION = {}
    
    def __init__(self, node_types:List[str]) -> None:
        self.type2prefix = {t: f"{t.upper()[0]}:" for t in node_types}
        self.prefix2type = {v: k for k, v in self.type2prefix.items()}
        self._G:DiGraph = None
        self._attribute_map:Dict[str, Any] = None
        
    def read_data(self, category:str, args) -> Tuple[List[Tuple[str, str, dict]], Dict]:
        '''Return edges and attribute map
        '''
        raise NotImplementedError
    
    def load_global_graph(self, args):
        graph_fn = osp.join(args.domain_dir, "graph.pkl")
        attr_fn = osp.join(args.domain_dir, "attributes.pkl")
        args_fn = osp.join(args.domain_dir, "args.json")
        if osp.exists(args_fn):
            with open(args_fn, 'r') as rf:
                saved_args = json.load(rf)
        else:
            saved_args = None
        if osp.exists(graph_fn) and osp.exists(attr_fn) and vars(args) == saved_args:
            print(f"\nLoading graph from {args.domain_dir}", flush=True)
            with open(graph_fn, "rb") as rf0, open(attr_fn, "rb") as rf1:
                G = pickle.load(rf0)
                attribute_map:Dict[str, Any] = pickle.load(rf1)
        else:
            print(f"\nLoading raw data from {len(args.categories)} categories", flush=True)
            with ProcessPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(self.read_data, category, args) for category in args.categories]

            edges = []
            attribute_map:Dict[str, Any] = {} # Actually this can be maintained by nx
            for future in as_completed(futures):
                sub_edges, sub_attribute_map = future.result()
                edges += sub_edges
                attribute_map.update(sub_attribute_map)
            
            print(f"\nBuilding graph with {len(edges)} edges", flush=True)
            G = DiGraph()
            G.add_edges_from(edges)
            
            print("\nSaving graph and args into disk...", flush=True)
            with open(graph_fn, "wb") as wf0, open(attr_fn, "wb") as wf1:
                pickle.dump(G, wf0)
                pickle.dump(attribute_map, wf1)
            with open(osp.join(args.domain_dir, "args.json"), 'w') as wf:
                json.dump(vars(args), wf)
        
        self._G = G
        self._attribute_map = attribute_map
    
    @property
    def G(self):
        return self._G
    
    @G.setter
    def G(self, G:DiGraph):
        self._G = G
    
    @property
    def attribute_map(self):
        return self._attribute_map
    
    @attribute_map.setter
    def attribute_map(self, attribute_map:Dict[str, Any]):
        self._attribute_map = attribute_map
        
    
class QuestionGenerator:
    def __init__(self) -> None:
        self._metadata:DatasetMetaData = None
        self.seen_tasks = []
        self.unseen_tasks = []
        # Target type
        self.structure_aware_tasks = []
        self.inductive_reasoning_tasks = []
        # Answer type
        self.node_tasks = []
        self.pair_tasks = []
        self.count_tasks = []
        self.bool_tasks = []
        self.path_tasks = []
        self.graph_tasks = []
        # Hop type
        self.single_hop_tasks = []
        self.multi_hop_tasks = []
        # Query type
        self.query_node_tasks = set()
        self.query_edge_tasks = set()
        self.query_num_tasks = set()
        
    @property
    def metadata(self):
        return self._metadata
    
    @metadata.setter
    def metadata(self, metadata:DatasetMetaData):
        self._metadata = metadata
        
    def mask_nodes(self, subgraph:DiGraph):
        type2nodes:Dict[str, List[str]] = defaultdict(list)
        for node in subgraph.nodes:
            type2nodes[self.metadata.prefix2type[node[:2]]].append(node)
        type2base = {t: random.randint(0, 100) for t in type2nodes.keys()} # Create random base number for each node type to avoid bias to small node numbers
        node2rep = {node: t + str(nid + type2base[t]) for t, nodes in type2nodes.items() for nid, node in enumerate(nodes)}
        rep2node = {v: k for k, v in node2rep.items()}
        new_subgraph = DiGraph()
        new_edges = [(node2rep[u], node2rep[v], {'type': t}) for u, v, t in subgraph.edges.data('type')]
        new_subgraph.add_edges_from(new_edges)
        return new_subgraph, rep2node
    
    def sample_attr_edges(self, rep2node:Dict[str, str], attr_sample_ratio:Dict[str, float]=None):
        # Get nodes with attributes
        nodes_with_attr = [(rep, node) for rep, node in rep2node.items() if node in self.metadata.attribute_map and self.metadata.attribute_map[node]['text']]
        if not attr_sample_ratio:
            selected_attributed_nodes = nodes_with_attr
        else:
            complete_attr_sample_ratio = defaultdict(lambda: 1.) # By default, all nodes will keep their attributes
            complete_attr_sample_ratio.update(**attr_sample_ratio)
            selected_attributed_nodes = [(rep, node) for rep, node in nodes_with_attr if random.uniform(0, 1) < complete_attr_sample_ratio[self.metadata.prefix2type[node[:2]]]]
        return [(rep, self.metadata.attribute_map[node]['text'], {'type': self.metadata.NAME}) for rep, node in selected_attributed_nodes]
    
    def structure_aware(self, collect_func):
        def structure_aware_generation(sample_size:int, subgraph_num:int=2, hop_num:int=2, edge_sample_size:Dict[str, Any]=None, attr_sample_ratio:Dict[str, float]=None, class_bin_start:int=None, class_bin_end:int=None):
            nodes:List[str] = list(self.metadata.G.nodes)
            dataset = []
            for sid in tqdm(range(sample_size)):
                generation = None
                rep2node = None
                class_bin = sid % (class_bin_end - class_bin_start + 1) + class_bin_start if class_bin_start is not None and class_bin_end is not None else None
                while generation is None:
                    starting_nodes = []
                    if len(starting_nodes) < subgraph_num:
                        starting_nodes.extend(random.sample(nodes, subgraph_num - len(starting_nodes)))
                    starting_nodes = list(set(starting_nodes))
                    subgraph = self.metadata.G.sample_subgraph(starting_nodes, hop_num, edge_sample_size if edge_sample_size else self.metadata.edge_sample_size)
                    subgraph, rep2node = self.mask_nodes(subgraph)
                    generation = collect_func(subgraph, class_bin)
                question, answer = generation
                attr_edges = self.sample_attr_edges(rep2node, attr_sample_ratio if attr_sample_ratio else self.metadata.attr_sample_ratio)
                subgraph.add_edges_from(attr_edges)
                dataset.append(pack_data(subgraph, question, answer))
            return dataset
        self.structure_aware_tasks.append((collect_func.__name__, structure_aware_generation))
        return structure_aware_generation 
    
    def inductive_reasoning(self, collect_func):
        def inductive_reasoning_generation(sample_size:int, hop_num:int=2, edge_sample_size:Dict[str, Any]=None, attr_sample_ratio:Dict[str, float]=None):
            new_G:DiGraph
            dataset = []
            new_G, dataset_wo_graph = collect_func(self.metadata.G, sample_size, edge_sample_size if edge_sample_size else self.metadata.edge_sample_size)
            args:dict
            for args, answer in dataset_wo_graph:
                # Build a subgraph centered at the target node
                subgraph = new_G.sample_subgraph([v for k, v in args.items()], hop_num, edge_sample_size if edge_sample_size else self.metadata.edge_sample_size)
                subgraph, rep2node = self.mask_nodes(subgraph)
                node2rep = {v: k for k, v in rep2node.items()}
                attr_edges = self.sample_attr_edges(rep2node, attr_sample_ratio if attr_sample_ratio else self.metadata.attr_sample_ratio)
                subgraph.add_edges_from(attr_edges)
                for k, v in args.items():
                    args[k] = node2rep[v]
                dataset.append(pack_data(subgraph, args, answer))
            return dataset
        self.inductive_reasoning_tasks.append((collect_func.__name__, inductive_reasoning_generation))
        return inductive_reasoning_generation
    
    def seen_task(self, collect_func):
        self.seen_tasks.append(collect_func.__name__)
        return collect_func
    
    def unseen_task(self, collect_func):
        self.unseen_tasks.append(collect_func.__name__)
        return collect_func

    def node_task(self, collect_func):
        self.node_tasks.append(collect_func.__name__)
        return collect_func
    
    def pair_task(self, collect_func):
        self.pair_tasks.append(collect_func.__name__)
        return collect_func
    
    def count_task(self, collect_func):
        self.count_tasks.append(collect_func.__name__)
        return collect_func
    
    def bool_task(self, collect_func):
        self.bool_tasks.append(collect_func.__name__)
        return collect_func
    
    def path_task(self, collect_func):
        self.path_tasks.append(collect_func.__name__)
        return collect_func
    
    def graph_task(self, collect_func):
        self.graph_tasks.append(collect_func.__name__)
        return collect_func
    
    def single_hop(self, collect_func):
        self.single_hop_tasks.append(collect_func.__name__)
        return collect_func
    
    def multi_hop(self, collect_func):
        self.multi_hop_tasks.append(collect_func.__name__)
        return collect_func
    
    def query_node(self, collect_func):
        self.query_node_tasks.add(collect_func.__name__)
        return collect_func
    
    def query_edge(self, collect_func):
        self.query_edge_tasks.add(collect_func.__name__)
        return collect_func
    
    def query_num(self, collect_func):
        self.query_num_tasks.add(collect_func.__name__)
        return collect_func
    