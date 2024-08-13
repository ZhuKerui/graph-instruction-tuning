import time
import os.path as osp
from typing import Tuple, List, Dict, Any
import sys
sys.path.append('..')
from question_generator import DatasetMetaData, QuestionGenerator, \
    build_edge_prediction, \
    find_neighbor, \
    nodes_within_hops, find_nodes_with_shared_attributes, \
    find_pairs, find_pairs_with_n_shared_attributes, \
    degree_count, \
    node_num_within_hops, count_path_between_nodes, \
    linked_by_edge, \
    has_path_between_nodes, \
    find_path_of_length, find_shortest_path, \
    find_ego_graph
from data_utils import read_gzip, remove_special_char
from Graph import DiGraph
import random
import itertools
    
class AmazonMeta(DatasetMetaData):
    # Node types
    CATEGORY = 'category'
    BRAND = 'brand'
    PRODUCT = 'product'
    
    # Attributes
    ASIN = 'asin'
    TITLE = 'title'
    PRICE = 'price'

    # Edge types
    PRODUCT_OF = 'product_of'
    BELONG_TO = 'belong_to'
    ALSO_VIEW = 'also_view'
    ALSO_BUY = 'also_buy'
    
    EDGE_ATTRBUTES = [PRODUCT_OF, BELONG_TO, ALSO_VIEW, ALSO_BUY]
    
    ATTRIBUTE_DESCRIPTION = {
        PRODUCT_OF: 'is a product of {tails}', 
        BELONG_TO: 'belongs to following categories: {tails}', 
        ALSO_VIEW: 'is also viewed with {tails}', 
        ALSO_BUY: 'is also bought together with {tails}', 
    }
    
    edge_sample_size = {
        'inv_' + BELONG_TO: 4,
        'inv_' + PRODUCT_OF : 4,
        'inv_' + ALSO_VIEW: 4,
        'inv_' + ALSO_BUY: 4,
        BELONG_TO: 4,
        PRODUCT_OF : 4,
        ALSO_VIEW: 4,
        ALSO_BUY: 4,
    }
    
    def __init__(self) -> None:
        super().__init__([self.CATEGORY, self.BRAND, self.PRODUCT])
        
    def read_data(self, category: str, args) -> Tuple[List[Tuple[str, str, dict]], Dict]:
        edges = []
        attribute_map = {}
        start_time = time.time()
        source_file = osp.join(args.raw_data_dir, f'meta_{category}.json.gz')
        metas = list(read_gzip(source_file))
        data:dict
        for data in metas:
            # Extract nodes
            category_nodes = [self.type2prefix[self.CATEGORY] + remove_special_char(c) for c in data.pop(self.CATEGORY)]
            brand_node = remove_special_char(data.pop(self.BRAND))
            brand_nodes = [self.type2prefix[self.BRAND] + brand_node] if brand_node else []
            current_product_node = self.type2prefix[self.PRODUCT] + data.pop(self.ASIN)
            also_view_product_nodes = [self.type2prefix[self.PRODUCT] + p for p in data.pop(self.ALSO_VIEW)]
            also_buy_product_nodes = [self.type2prefix[self.PRODUCT] + p for p in data.pop(self.ALSO_BUY)]
            # Register attribute for node
            attribute_map[current_product_node] = {'text': data[self.TITLE]}
            for category_node in category_nodes:
                attribute_map[category_node] = {'text': category_node[2:]}
            for brand_node in brand_nodes:
                attribute_map[brand_node] = {'text': brand_node[2:]}
            # Add edges to graph
            edges.extend([(current_product_node, p, {'type': self.ALSO_VIEW}) for p in also_view_product_nodes])
            edges.extend([(current_product_node, p, {'type': self.ALSO_BUY}) for p in also_buy_product_nodes])
            edges.extend([(current_product_node, b, {'type': self.PRODUCT_OF}) for b in brand_nodes])
            edges.extend([(current_product_node, c, {'type': self.BELONG_TO}) for c in category_nodes])

        print(f"{category} done (Elasped time: {time.time() - start_time:.2f} sec)", flush=True)
        
        return edges, attribute_map
        
amazon_tasks = QuestionGenerator()
    
# ---------------------------------------- Structure aware instructions ----------------------------------------

# ------------------------------ Node tasks ------------------------------

# ---------------------- Single hop ----------------------

# --------- find_neighbor ---------

@amazon_tasks.structure_aware
@amazon_tasks.seen_task
@amazon_tasks.node_task
@amazon_tasks.single_hop
@amazon_tasks.query_node
@amazon_tasks.query_edge
def also_view_or_buy_product(subgraph:DiGraph, class_bin:int=None):
    attribute_type = random.choice([AmazonMeta.ALSO_BUY, AmazonMeta.ALSO_VIEW])
    return find_neighbor(subgraph, attribute_type, class_bin)

@amazon_tasks.structure_aware
@amazon_tasks.seen_task
@amazon_tasks.node_task
@amazon_tasks.single_hop
@amazon_tasks.query_node
@amazon_tasks.query_edge
def product_of_brand(subgraph:DiGraph, class_bin:int=None):
    return find_neighbor(subgraph, 'inv_' + AmazonMeta.PRODUCT_OF, class_bin)

@amazon_tasks.structure_aware
@amazon_tasks.unseen_task
@amazon_tasks.node_task
@amazon_tasks.single_hop
@amazon_tasks.query_node
@amazon_tasks.query_edge
def brand_or_category_of_product(subgraph:DiGraph, class_bin:int=None):
    attribute_type = random.choice([AmazonMeta.PRODUCT_OF, AmazonMeta.BELONG_TO])
    return find_neighbor(subgraph, attribute_type, class_bin)

@amazon_tasks.structure_aware
@amazon_tasks.unseen_task
@amazon_tasks.node_task
@amazon_tasks.single_hop
@amazon_tasks.query_node
@amazon_tasks.query_edge
def product_of_category(subgraph:DiGraph, class_bin:int=None):
    return find_neighbor(subgraph, 'inv_' + AmazonMeta.BELONG_TO, class_bin)


# ---------------------- Multi hop ----------------------

# --------- find_nodes_with_shared_attributes ---------

@amazon_tasks.structure_aware
@amazon_tasks.seen_task
@amazon_tasks.node_task
@amazon_tasks.multi_hop
@amazon_tasks.query_node
@amazon_tasks.query_edge
def product_with_shared_also_view_or_buy(subgraph:DiGraph, class_bin:int=None):
    attribute_type = random.choice([AmazonMeta.ALSO_BUY, AmazonMeta.ALSO_VIEW])
    return find_nodes_with_shared_attributes(subgraph, [attribute_type], class_bin)

@amazon_tasks.structure_aware
@amazon_tasks.seen_task
@amazon_tasks.node_task
@amazon_tasks.multi_hop
@amazon_tasks.query_node
@amazon_tasks.query_edge
def product_with_shared_brand_and_category(subgraph:DiGraph, class_bin:int=None):
    return find_nodes_with_shared_attributes(subgraph, [AmazonMeta.PRODUCT_OF, AmazonMeta.BELONG_TO], class_bin)

@amazon_tasks.structure_aware
@amazon_tasks.unseen_task
@amazon_tasks.node_task
@amazon_tasks.multi_hop
@amazon_tasks.query_node
@amazon_tasks.query_edge
def product_with_shared_category_or_brand(subgraph:DiGraph, class_bin:int=None):
    attribute_type = random.choice([AmazonMeta.PRODUCT_OF, AmazonMeta.BELONG_TO])
    return find_nodes_with_shared_attributes(subgraph, [attribute_type], class_bin)

@amazon_tasks.structure_aware
@amazon_tasks.unseen_task
@amazon_tasks.node_task
@amazon_tasks.multi_hop
@amazon_tasks.query_node
@amazon_tasks.query_edge
def product_with_shared_also_buy_and_view(subgraph:DiGraph, class_bin:int=None):
    return find_nodes_with_shared_attributes(subgraph, [AmazonMeta.ALSO_BUY, AmazonMeta.ALSO_VIEW], class_bin)

# --------- nodes_within_hops ---------

@amazon_tasks.structure_aware
@amazon_tasks.seen_task
@amazon_tasks.node_task
@amazon_tasks.multi_hop
@amazon_tasks.query_node
@amazon_tasks.query_num
def brands_or_categories_within_hops_to_product(subgraph:DiGraph, class_bin:int=None):
    target_type = random.choice([AmazonMeta.BRAND, AmazonMeta.CATEGORY])
    return nodes_within_hops(subgraph, AmazonMeta.PRODUCT, target_type, class_bin)

@amazon_tasks.structure_aware
@amazon_tasks.unseen_task
@amazon_tasks.node_task
@amazon_tasks.multi_hop
@amazon_tasks.query_node
@amazon_tasks.query_num
def brands_within_hops_to_category(subgraph:DiGraph, class_bin:int=None):
    return nodes_within_hops(subgraph, AmazonMeta.CATEGORY, AmazonMeta.BRAND, class_bin)

@amazon_tasks.structure_aware
@amazon_tasks.unseen_task
@amazon_tasks.node_task
@amazon_tasks.multi_hop
@amazon_tasks.query_node
@amazon_tasks.query_num
def categories_within_hops_to_brand(subgraph:DiGraph, class_bin:int=None):
    return nodes_within_hops(subgraph, AmazonMeta.BRAND, AmazonMeta.CATEGORY, class_bin)

# ------------------------------- Pair tasks ------------------------------

# ---------------------- Single hop ----------------------

# --------- find_pairs ---------

@amazon_tasks.structure_aware
@amazon_tasks.seen_task
@amazon_tasks.pair_task
@amazon_tasks.single_hop
@amazon_tasks.query_node
@amazon_tasks.query_edge
def also_view_pairs_with_product(subgraph:DiGraph, class_bin:int=None):
    return find_pairs(subgraph, AmazonMeta.PRODUCT, [AmazonMeta.ALSO_VIEW], class_bin)

@amazon_tasks.structure_aware
@amazon_tasks.unseen_task
@amazon_tasks.pair_task
@amazon_tasks.single_hop
@amazon_tasks.query_node
@amazon_tasks.query_edge
def pairs_with_brand_or_category(subgraph:DiGraph, class_bin:int=None):
    node_type, attribute_type = random.choice([(AmazonMeta.BRAND, AmazonMeta.PRODUCT_OF), (AmazonMeta.CATEGORY, AmazonMeta.BELONG_TO)])
    return find_pairs(subgraph, node_type, [attribute_type], class_bin)

@amazon_tasks.structure_aware
@amazon_tasks.unseen_task
@amazon_tasks.pair_task
@amazon_tasks.single_hop
@amazon_tasks.query_node
@amazon_tasks.query_edge
def three_pair_types_with_product(subgraph:DiGraph, class_bin:int=None):
    return find_pairs(subgraph, AmazonMeta.PRODUCT, [AmazonMeta.ALSO_BUY, AmazonMeta.BELONG_TO, AmazonMeta.PRODUCT_OF], class_bin)

# ---------------------- Multi hop ----------------------

# --------- find_pairs_with_n_shared_attributes ---------

@amazon_tasks.structure_aware
@amazon_tasks.seen_task
@amazon_tasks.pair_task
@amazon_tasks.multi_hop
@amazon_tasks.query_num
@amazon_tasks.query_edge
def pairs_with_shared_also_view(subgraph:DiGraph, class_bin:int=None):
    return find_pairs_with_n_shared_attributes(subgraph, [AmazonMeta.ALSO_VIEW], class_bin)

@amazon_tasks.structure_aware
@amazon_tasks.unseen_task
@amazon_tasks.pair_task
@amazon_tasks.multi_hop
@amazon_tasks.query_num
@amazon_tasks.query_edge
def pairs_with_shared_non_also_view(subgraph:DiGraph, class_bin:int=None):
    attribute_type = random.choice([AmazonMeta.ALSO_BUY, AmazonMeta.PRODUCT_OF, AmazonMeta.BELONG_TO])
    return find_pairs_with_n_shared_attributes(subgraph, [attribute_type], class_bin)

@amazon_tasks.structure_aware
@amazon_tasks.unseen_task
@amazon_tasks.pair_task
@amazon_tasks.multi_hop
@amazon_tasks.query_num
@amazon_tasks.query_edge
def pairs_with_2_shared_attributes(subgraph:DiGraph, class_bin:int=None):
    attribute_types = random.choice(list(itertools.combinations([AmazonMeta.ALSO_VIEW, AmazonMeta.ALSO_BUY, AmazonMeta.PRODUCT_OF, AmazonMeta.BELONG_TO], 2)))
    return find_pairs_with_n_shared_attributes(subgraph, list(attribute_types), class_bin)

# ------------------------------- Count tasks ------------------------------

# ---------------------- Single hop ----------------------

# --------- degree_count ---------

@amazon_tasks.structure_aware
@amazon_tasks.seen_task
@amazon_tasks.count_task
@amazon_tasks.single_hop
@amazon_tasks.query_node
@amazon_tasks.query_edge
def also_view_num(subgraph:DiGraph, class_bin:int=None):
    attribute_type = random.choice([AmazonMeta.ALSO_VIEW, AmazonMeta.ALSO_BUY])
    return degree_count(subgraph, [attribute_type], class_bin)

@amazon_tasks.structure_aware
@amazon_tasks.seen_task
@amazon_tasks.count_task
@amazon_tasks.single_hop
@amazon_tasks.query_node
@amazon_tasks.query_edge
def product_num_of_brand(subgraph:DiGraph, class_bin:int=None):
    return degree_count(subgraph, ['inv_' + AmazonMeta.PRODUCT_OF], class_bin)

@amazon_tasks.structure_aware
@amazon_tasks.unseen_task
@amazon_tasks.count_task
@amazon_tasks.single_hop
@amazon_tasks.query_node
@amazon_tasks.query_edge
def also_view_buy_num(subgraph:DiGraph, class_bin:int=None):
    return degree_count(subgraph, [AmazonMeta.ALSO_VIEW, AmazonMeta.ALSO_BUY], class_bin)

@amazon_tasks.structure_aware
@amazon_tasks.unseen_task
@amazon_tasks.count_task
@amazon_tasks.single_hop
@amazon_tasks.query_node
@amazon_tasks.query_edge
def product_num_of_category(subgraph:DiGraph, class_bin:int=None):
    return degree_count(subgraph, ['inv_' + AmazonMeta.BELONG_TO], class_bin)

# ---------------------- Multi hop ----------------------

# --------- node_num_within_hops ---------

@amazon_tasks.structure_aware
@amazon_tasks.seen_task
@amazon_tasks.count_task
@amazon_tasks.multi_hop
@amazon_tasks.query_node
@amazon_tasks.query_num
def product_num_within_hops(subgraph:DiGraph, class_bin:int=None):
    source_type = random.choice([AmazonMeta.PRODUCT, AmazonMeta.BRAND, AmazonMeta.CATEGORY])
    return node_num_within_hops(subgraph, source_type, AmazonMeta.PRODUCT, class_bin)

@amazon_tasks.structure_aware
@amazon_tasks.unseen_task
@amazon_tasks.count_task
@amazon_tasks.multi_hop
@amazon_tasks.query_node
@amazon_tasks.query_num
def brand_or_category_num_within_hops(subgraph:DiGraph, class_bin:int=None):
    source_type = random.choice([AmazonMeta.PRODUCT, AmazonMeta.BRAND, AmazonMeta.CATEGORY])
    target_type = random.choice([AmazonMeta.BRAND, AmazonMeta.CATEGORY])
    return node_num_within_hops(subgraph, source_type, target_type, class_bin)

# --------- count_path_between_nodes ---------

@amazon_tasks.structure_aware
@amazon_tasks.seen_task
@amazon_tasks.count_task
@amazon_tasks.multi_hop
@amazon_tasks.query_node
@amazon_tasks.query_num
def path_num_between_product_product(subgraph:DiGraph, class_bin:int=None):
    return count_path_between_nodes(subgraph, AmazonMeta.PRODUCT, AmazonMeta.PRODUCT, class_bin)

@amazon_tasks.structure_aware
@amazon_tasks.unseen_task
@amazon_tasks.count_task
@amazon_tasks.multi_hop
@amazon_tasks.query_node
@amazon_tasks.query_num
def path_num_between_non_product_pair(subgraph:DiGraph, class_bin:int=None):
    source_target_pairs = list(itertools.product([AmazonMeta.BRAND, AmazonMeta.CATEGORY, AmazonMeta.PRODUCT], [AmazonMeta.BRAND, AmazonMeta.CATEGORY, AmazonMeta.PRODUCT]))
    source_type, target_type = random.choice(source_target_pairs)
    while (source_type, target_type) == (AmazonMeta.PRODUCT, AmazonMeta.PRODUCT):
        source_type, target_type = random.choice(source_target_pairs)
    return count_path_between_nodes(subgraph, source_type, target_type, class_bin)

# ------------------------------- Bool tasks ------------------------------

# ---------------------- Single hop ----------------------

# --------- linked_by_edge ---------

@amazon_tasks.structure_aware
@amazon_tasks.seen_task
@amazon_tasks.bool_task
@amazon_tasks.single_hop
@amazon_tasks.query_node
@amazon_tasks.query_edge
def has_also_view_edge(subgraph:DiGraph, class_bin:int=None):
    return linked_by_edge(subgraph, AmazonMeta.PRODUCT, AmazonMeta.PRODUCT, AmazonMeta.ALSO_VIEW, class_bin)

@amazon_tasks.structure_aware
@amazon_tasks.unseen_task
@amazon_tasks.bool_task
@amazon_tasks.single_hop
@amazon_tasks.query_node
@amazon_tasks.query_edge
def has_non_also_view_edge(subgraph:DiGraph, class_bin:int=None):
    target_type, attribute_type = random.choice([(AmazonMeta.CATEGORY, AmazonMeta.BELONG_TO), (AmazonMeta.BRAND, AmazonMeta.PRODUCT_OF), (AmazonMeta.PRODUCT, AmazonMeta.ALSO_BUY)])
    return linked_by_edge(subgraph, AmazonMeta.PRODUCT, target_type, attribute_type, class_bin)

# ---------------------- Multi hop ----------------------

# --------- has_path_between_nodes ---------

@amazon_tasks.structure_aware
@amazon_tasks.seen_task
@amazon_tasks.bool_task
@amazon_tasks.multi_hop
@amazon_tasks.query_node
def has_path_between_product_product(subgraph:DiGraph, class_bin:int=None):
    return has_path_between_nodes(subgraph, AmazonMeta.PRODUCT, AmazonMeta.PRODUCT, class_bin)

@amazon_tasks.structure_aware
@amazon_tasks.unseen_task
@amazon_tasks.bool_task
@amazon_tasks.multi_hop
@amazon_tasks.query_node
def has_path_between_non_product_pair(subgraph:DiGraph, class_bin:int=None):
    source_target_pairs = list(itertools.product([AmazonMeta.BRAND, AmazonMeta.CATEGORY, AmazonMeta.PRODUCT], [AmazonMeta.BRAND, AmazonMeta.CATEGORY, AmazonMeta.PRODUCT]))
    source_type, target_type = random.choice(source_target_pairs)
    while (source_type, target_type) == (AmazonMeta.PRODUCT, AmazonMeta.PRODUCT):
        source_type, target_type = random.choice(source_target_pairs)
    return has_path_between_nodes(subgraph, source_type, target_type, class_bin)

# ------------------------------- Path tasks ------------------------------

# --------- find_path_of_length ---------

@amazon_tasks.structure_aware
@amazon_tasks.seen_task
@amazon_tasks.path_task
@amazon_tasks.multi_hop
@amazon_tasks.query_node
@amazon_tasks.query_num
def find_path_of_length_between_product_product(subgraph:DiGraph, class_bin:int=None):
    return find_path_of_length(subgraph, AmazonMeta.PRODUCT, AmazonMeta.PRODUCT, class_bin)

@amazon_tasks.structure_aware
@amazon_tasks.unseen_task
@amazon_tasks.path_task
@amazon_tasks.multi_hop
@amazon_tasks.query_node
@amazon_tasks.query_num
def find_path_of_length_between_non_product_pair(subgraph:DiGraph, class_bin:int=None):
    source_target_pairs = list(itertools.product([AmazonMeta.BRAND, AmazonMeta.CATEGORY, AmazonMeta.PRODUCT], [AmazonMeta.BRAND, AmazonMeta.CATEGORY, AmazonMeta.PRODUCT]))
    source_type, target_type = random.choice(source_target_pairs)
    while (source_type, target_type) == (AmazonMeta.PRODUCT, AmazonMeta.PRODUCT):
        source_type, target_type = random.choice(source_target_pairs)
    return find_path_of_length(subgraph, source_type, target_type, class_bin)

# --------- find_shortest_path ---------

@amazon_tasks.structure_aware
@amazon_tasks.seen_task
@amazon_tasks.path_task
@amazon_tasks.multi_hop
@amazon_tasks.query_node
def find_shortest_path_between_product_product(subgraph:DiGraph, class_bin:int=None):
    return find_shortest_path(subgraph, AmazonMeta.PRODUCT, AmazonMeta.PRODUCT, class_bin)

@amazon_tasks.structure_aware
@amazon_tasks.unseen_task
@amazon_tasks.path_task
@amazon_tasks.multi_hop
@amazon_tasks.query_node
def find_shortest_path_between_non_product_pair(subgraph:DiGraph, class_bin:int=None):
    source_target_pairs = list(itertools.product([AmazonMeta.BRAND, AmazonMeta.CATEGORY, AmazonMeta.PRODUCT], [AmazonMeta.BRAND, AmazonMeta.CATEGORY, AmazonMeta.PRODUCT]))
    source_type, target_type = random.choice(source_target_pairs)
    while (source_type, target_type) == (AmazonMeta.PRODUCT, AmazonMeta.PRODUCT):
        source_type, target_type = random.choice(source_target_pairs)
    return find_shortest_path(subgraph, source_type, target_type, class_bin)

# ------------------------------- Graph tasks ------------------------------

# --------- find_ego_graph ---------

@amazon_tasks.structure_aware
@amazon_tasks.seen_task
@amazon_tasks.graph_task
@amazon_tasks.single_hop
@amazon_tasks.query_node
@amazon_tasks.query_num
def find_1_hop_ego_graph_for_product(subgraph:DiGraph, class_bin:int=None):
    return find_ego_graph(subgraph, AmazonMeta.PRODUCT, 1, class_bin)

@amazon_tasks.structure_aware
@amazon_tasks.unseen_task
@amazon_tasks.graph_task
@amazon_tasks.single_hop
@amazon_tasks.query_node
@amazon_tasks.query_num
def find_1_hop_ego_graph_for_brand_or_category(subgraph:DiGraph, class_bin:int=None):
    center_type = random.choice([AmazonMeta.BRAND, AmazonMeta.CATEGORY])
    return find_ego_graph(subgraph, center_type, 1, class_bin)

@amazon_tasks.structure_aware
@amazon_tasks.seen_task
@amazon_tasks.graph_task
@amazon_tasks.multi_hop
@amazon_tasks.query_node
@amazon_tasks.query_num
def find_2_hop_ego_graph_for_product(subgraph:DiGraph, class_bin:int=None):
    return find_ego_graph(subgraph, AmazonMeta.PRODUCT, 2, class_bin)

@amazon_tasks.structure_aware
@amazon_tasks.unseen_task
@amazon_tasks.graph_task
@amazon_tasks.multi_hop
@amazon_tasks.query_node
@amazon_tasks.query_num
def find_2_hop_ego_graph_for_brand_or_category(subgraph:DiGraph, class_bin:int=None):
    center_type = random.choice([AmazonMeta.BRAND, AmazonMeta.CATEGORY])
    return find_ego_graph(subgraph, center_type, 2, class_bin)

# ---------------------------------------- Inductive reasoning instructions ----------------------------------------

# def build_price_prediction(self, G:DiGraph, attribute_map:dict, sample_size:int):
#     # Find candidate nodes
#     product_nodes = [node for node in G.nodes if node.startswith(self.PRODUCT) and node in attribute_map and self.PRICE in attribute_map[node]]
#     masked_product_nodes = random.sample(product_nodes, sample_size)

#     # Extract answers and remove answers from graph
#     answers = []
#     for masked_product_node in tqdm(masked_product_nodes):
#         answers.append((masked_product_node, attribute_map[masked_product_node].pop(self.PRICE)))
    
#     # Select a target node and build the question
#     for node, answer in answers:
#         question = f"Predict the price of product {node}."

#         # Build a subgraph centered at the target node
#         subgraph = G.sample_subgraph(node, 2, self.edge_sample_size)
#         graph_info = graph2data(subgraph, self.prefix2type)

#         data_item = form_data_item(graph_info, question, price, 'price_prediction')
#         return data_item


# def price_prediction(self, G:DiGraph, target_product_node:str, price:str, edge_sample_ratio:Dict[str, float]=None):
#     # Select a target node and build the question
#     question = f"Predict the price of product {target_product_node}."

#     # Build a subgraph centered at the target node
#     subgraph = sample_subgraph(target_product_node, G, 2, edge_sample_ratio, max_sample_num)
#     graph_info = graph2data(subgraph, AmazonGenerator.prefix2type)

#     data_item = form_data_item(graph_info, question, price, 'price_prediction')
#     return data_item

@amazon_tasks.inductive_reasoning
@amazon_tasks.seen_task
@amazon_tasks.bool_task
def coview_prediction(G:DiGraph, sample_size:int, edge_sample_size:Dict[str, Any]=None):
    return build_edge_prediction(G, AmazonMeta.ALSO_VIEW, sample_size, edge_sample_size)
    
@amazon_tasks.inductive_reasoning
@amazon_tasks.seen_task
@amazon_tasks.bool_task
def co_purchased_prediction(G:DiGraph, sample_size:int, edge_sample_size:Dict[str, Any]=None):
    return build_edge_prediction(G, AmazonMeta.ALSO_BUY, sample_size, edge_sample_size)
    
@amazon_tasks.inductive_reasoning
@amazon_tasks.unseen_task
@amazon_tasks.bool_task
def brand_prediction(G:DiGraph, sample_size:int, edge_sample_size:Dict[str, Any]=None):
    return build_edge_prediction(G, AmazonMeta.PRODUCT_OF, sample_size, edge_sample_size)
