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
from data_utils import read_tsv
from graph import DiGraph
import random
import itertools

class MapleMeta(DatasetMetaData):
    
    # Node types
    PAPER = "paper"
    AUTHOR = "author"
    VENUE = "venue"

    # Edge types
    WRITTEN_BY = "written_by"
    PUBLISHED_ON = "published_on"
    CITE = "cite"
    
    EDGE_ATTRBUTES = [WRITTEN_BY, PUBLISHED_ON, CITE]
    
    ATTRIBUTE_DESCRIPTION = {
        WRITTEN_BY: 'is written by {tails}', 
        PUBLISHED_ON: 'is published on {tails}', 
        CITE: 'cites {tails}', 
    }
    
    edge_sample_size = {
        f"inv_{PUBLISHED_ON}": 0.033,
        f"inv_{WRITTEN_BY}": .99,
        f"inv_{CITE}": 0.5,
        WRITTEN_BY: .99,
        CITE: 0.5,
    }
    
    attr_sample_ratio = {
        PAPER: 0.05,
        AUTHOR: 0.1
    }
    
    def __init__(self) -> None:
        super().__init__([self.PAPER, self.AUTHOR, self.VENUE])
        
    def read_data(self, category: str, args) -> Tuple[List[Tuple[str, str, dict]], Dict]:
        edges = []
        attribute_map = {} # Actually this can be maintained by nx
        prefix_norm = lambda s, c: self.type2prefix[c] + s[s.index('_') + 1:]
        start_time = time.time()
        get_fname = lambda n: osp.join(args.raw_data_dir, category, n)
        for pid, abstract in read_tsv(get_fname("papers.txt")):
            pid = prefix_norm(pid, self.PAPER)
            attribute_map[pid] = {"text": abstract}
            # edges.append((pid, cid, {"type": self.BELONG_TO}))
        for aid, author in read_tsv(get_fname("authors.txt")):
            aid = prefix_norm(aid, self.AUTHOR)
            attribute_map[aid] = {"text": author}
        for vid, venue in read_tsv(get_fname("venues.txt")):
            vid = prefix_norm(vid, self.VENUE)
            attribute_map[vid] = {"text": venue}
        for pid, aid in read_tsv(get_fname("paper-author.txt")):
            pid, aid = prefix_norm(pid, self.PAPER), prefix_norm(aid, self.AUTHOR)
            edges.append((pid, aid, {"type": self.WRITTEN_BY}))
        for pid, vid in read_tsv(get_fname("paper-venue.txt")):
            pid, vid = prefix_norm(pid, self.PAPER), prefix_norm(vid, self.VENUE)
            edges.append((pid, vid, {"type": self.PUBLISHED_ON}))
        for pid, cited_pid in read_tsv(get_fname("paper-paper.txt")):
            pid, cited_pid = prefix_norm(pid, self.PAPER), prefix_norm(cited_pid, self.PAPER)
            edges.append((pid, cited_pid, {"type": self.CITE}))
        print(f"{category} done (Elasped time: {time.time() - start_time:.2f} sec)", flush=True)
        
        return edges, attribute_map
    
maple_tasks = QuestionGenerator()

# ---------------------------------------- Structure aware instructions ----------------------------------------

# ------------------------------ Node tasks ------------------------------

# ---------------------- Single hop ----------------------

# --------- find_neighbor ---------

@maple_tasks.structure_aware
@maple_tasks.seen_task
@maple_tasks.node_task
@maple_tasks.single_hop
@maple_tasks.query_node
@maple_tasks.query_edge
def cited_paper_of_paper(subgraph:DiGraph, class_bin:int=None):
    return find_neighbor(subgraph, MapleMeta.CITE, class_bin)

@maple_tasks.structure_aware
@maple_tasks.seen_task
@maple_tasks.node_task
@maple_tasks.single_hop
@maple_tasks.query_node
@maple_tasks.query_edge
def paper_of_author(subgraph:DiGraph, class_bin:int=None):
    return find_neighbor(subgraph, 'inv_' + MapleMeta.WRITTEN_BY, class_bin)

@maple_tasks.structure_aware
@maple_tasks.unseen_task
@maple_tasks.node_task
@maple_tasks.single_hop
@maple_tasks.query_node
@maple_tasks.query_edge
def venue_or_authors_of_paper(subgraph:DiGraph, class_bin:int=None):
    attribute_type = random.choice([MapleMeta.PUBLISHED_ON, MapleMeta.WRITTEN_BY])
    return find_neighbor(subgraph, attribute_type, class_bin)

@maple_tasks.structure_aware
@maple_tasks.unseen_task
@maple_tasks.node_task
@maple_tasks.single_hop
@maple_tasks.query_node
@maple_tasks.query_edge
def paper_of_venue(subgraph:DiGraph, class_bin:int=None):
    return find_neighbor(subgraph, 'inv_' + MapleMeta.PUBLISHED_ON, class_bin)


# ---------------------- Multi hop ----------------------

# --------- find_nodes_with_shared_attributes ---------

@maple_tasks.structure_aware
@maple_tasks.seen_task
@maple_tasks.node_task
@maple_tasks.multi_hop
@maple_tasks.query_node
@maple_tasks.query_edge
def paper_with_shared_cited_paper(subgraph:DiGraph, class_bin:int=None):
    return find_nodes_with_shared_attributes(subgraph, [MapleMeta.CITE], class_bin)

@maple_tasks.structure_aware
@maple_tasks.unseen_task
@maple_tasks.node_task
@maple_tasks.multi_hop
@maple_tasks.query_node
@maple_tasks.query_edge
def paper_with_shared_author_or_venue(subgraph:DiGraph, class_bin:int=None):
    attribute_type = random.choice([MapleMeta.WRITTEN_BY, MapleMeta.PUBLISHED_ON])
    return find_nodes_with_shared_attributes(subgraph, [attribute_type], class_bin)

@maple_tasks.structure_aware
@maple_tasks.unseen_task
@maple_tasks.node_task
@maple_tasks.multi_hop
@maple_tasks.query_node
@maple_tasks.query_edge
def paper_with_shared_author_and_venue(subgraph:DiGraph, class_bin:int=None):
    return find_nodes_with_shared_attributes(subgraph, [MapleMeta.WRITTEN_BY, MapleMeta.PUBLISHED_ON], class_bin)

# --------- nodes_within_hops ---------

@maple_tasks.structure_aware
@maple_tasks.seen_task
@maple_tasks.node_task
@maple_tasks.multi_hop
@maple_tasks.query_node
@maple_tasks.query_num
def venues_within_hops_to_paper(subgraph:DiGraph, class_bin:int=None):
    return nodes_within_hops(subgraph, MapleMeta.PAPER, MapleMeta.VENUE, class_bin)

@maple_tasks.structure_aware
@maple_tasks.unseen_task
@maple_tasks.node_task
@maple_tasks.multi_hop
@maple_tasks.query_node
@maple_tasks.query_num
def venues_within_hops_to_author(subgraph:DiGraph, class_bin:int=None):
    return nodes_within_hops(subgraph, MapleMeta.AUTHOR, MapleMeta.VENUE, class_bin)

@maple_tasks.structure_aware
@maple_tasks.unseen_task
@maple_tasks.node_task
@maple_tasks.multi_hop
@maple_tasks.query_node
@maple_tasks.query_num
def authors_within_hops_to_venue(subgraph:DiGraph, class_bin:int=None):
    return nodes_within_hops(subgraph, MapleMeta.VENUE, MapleMeta.AUTHOR, class_bin)

# ------------------------------- Pair tasks ------------------------------

# ---------------------- Single hop ----------------------

# --------- find_pairs ---------

@maple_tasks.structure_aware
@maple_tasks.seen_task
@maple_tasks.pair_task
@maple_tasks.single_hop
@maple_tasks.query_node
@maple_tasks.query_edge
def cite_pairs_with_paper(subgraph:DiGraph, class_bin:int=None):
    return find_pairs(subgraph, MapleMeta.PAPER, [MapleMeta.CITE], class_bin)

@maple_tasks.structure_aware
@maple_tasks.unseen_task
@maple_tasks.pair_task
@maple_tasks.single_hop
@maple_tasks.query_node
@maple_tasks.query_edge
def pairs_with_venue_or_author(subgraph:DiGraph, class_bin:int=None):
    node_type, attribute_type = random.choice([(MapleMeta.VENUE, MapleMeta.PUBLISHED_ON), (MapleMeta.AUTHOR, MapleMeta.WRITTEN_BY)])
    return find_pairs(subgraph, node_type, [attribute_type], class_bin)

@maple_tasks.structure_aware
@maple_tasks.unseen_task
@maple_tasks.pair_task
@maple_tasks.single_hop
@maple_tasks.query_node
@maple_tasks.query_edge
def three_pair_types_with_paper(subgraph:DiGraph, class_bin:int=None):
    return find_pairs(subgraph, MapleMeta.PAPER, [MapleMeta.WRITTEN_BY, MapleMeta.CITE, MapleMeta.PUBLISHED_ON], class_bin)

# ---------------------- Multi hop ----------------------

# --------- find_pairs_with_n_shared_attributes ---------

@maple_tasks.structure_aware
@maple_tasks.seen_task
@maple_tasks.pair_task
@maple_tasks.multi_hop
@maple_tasks.query_num
@maple_tasks.query_edge
def pairs_with_shared_cite(subgraph:DiGraph, class_bin:int=None):
    return find_pairs_with_n_shared_attributes(subgraph, [MapleMeta.CITE], class_bin)

@maple_tasks.structure_aware
@maple_tasks.unseen_task
@maple_tasks.pair_task
@maple_tasks.multi_hop
@maple_tasks.query_num
@maple_tasks.query_edge
def pairs_with_shared_non_cite(subgraph:DiGraph, class_bin:int=None):
    attribute_type = random.choice([MapleMeta.WRITTEN_BY, MapleMeta.PUBLISHED_ON])
    return find_pairs_with_n_shared_attributes(subgraph, [attribute_type], class_bin)

@maple_tasks.structure_aware
@maple_tasks.unseen_task
@maple_tasks.pair_task
@maple_tasks.multi_hop
@maple_tasks.query_num
@maple_tasks.query_edge
def pairs_with_2_shared_attributes(subgraph:DiGraph, class_bin:int=None):
    attribute_types = random.choice(list(itertools.combinations([MapleMeta.CITE, MapleMeta.PUBLISHED_ON, MapleMeta.WRITTEN_BY], 2)))
    return find_pairs_with_n_shared_attributes(subgraph, list(attribute_types), class_bin)

# ------------------------------- Count tasks ------------------------------

# ---------------------- Single hop ----------------------

# --------- degree_count ---------

@maple_tasks.structure_aware
@maple_tasks.seen_task
@maple_tasks.count_task
@maple_tasks.single_hop
@maple_tasks.query_node
@maple_tasks.query_edge
def cite_num(subgraph:DiGraph, class_bin:int=None):
    return degree_count(subgraph, [MapleMeta.CITE], class_bin)

@maple_tasks.structure_aware
@maple_tasks.seen_task
@maple_tasks.count_task
@maple_tasks.single_hop
@maple_tasks.query_node
@maple_tasks.query_edge
def paper_num_of_venue(subgraph:DiGraph, class_bin:int=None):
    return degree_count(subgraph, ['inv_' + MapleMeta.PUBLISHED_ON], class_bin)

@maple_tasks.structure_aware
@maple_tasks.unseen_task
@maple_tasks.count_task
@maple_tasks.single_hop
@maple_tasks.query_node
@maple_tasks.query_edge
def cite_written_by_num(subgraph:DiGraph, class_bin:int=None):
    return degree_count(subgraph, [MapleMeta.CITE, MapleMeta.WRITTEN_BY], class_bin)

@maple_tasks.structure_aware
@maple_tasks.unseen_task
@maple_tasks.count_task
@maple_tasks.single_hop
@maple_tasks.query_node
@maple_tasks.query_edge
def paper_num_of_author(subgraph:DiGraph, class_bin:int=None):
    return degree_count(subgraph, ['inv_' + MapleMeta.WRITTEN_BY], class_bin)

# ---------------------- Multi hop ----------------------

# --------- node_num_within_hops ---------

@maple_tasks.structure_aware
@maple_tasks.seen_task
@maple_tasks.count_task
@maple_tasks.multi_hop
@maple_tasks.query_node
@maple_tasks.query_num
def paper_num_within_hops(subgraph:DiGraph, class_bin:int=None):
    source_type = random.choice([MapleMeta.PAPER, MapleMeta.AUTHOR, MapleMeta.VENUE])
    return node_num_within_hops(subgraph, source_type, MapleMeta.PAPER, class_bin)

@maple_tasks.structure_aware
@maple_tasks.unseen_task
@maple_tasks.count_task
@maple_tasks.multi_hop
@maple_tasks.query_node
@maple_tasks.query_num
def venue_or_author_num_within_hops(subgraph:DiGraph, class_bin:int=None):
    source_type = random.choice([MapleMeta.PAPER, MapleMeta.AUTHOR, MapleMeta.VENUE])
    target_type = random.choice([MapleMeta.AUTHOR, MapleMeta.VENUE])
    return node_num_within_hops(subgraph, source_type, target_type, class_bin)

# --------- count_path_between_nodes ---------

@maple_tasks.structure_aware
@maple_tasks.seen_task
@maple_tasks.count_task
@maple_tasks.multi_hop
@maple_tasks.query_node
@maple_tasks.query_num
def path_num_between_paper_paper(subgraph:DiGraph, class_bin:int=None):
    return count_path_between_nodes(subgraph, MapleMeta.PAPER, MapleMeta.PAPER, class_bin)

@maple_tasks.structure_aware
@maple_tasks.unseen_task
@maple_tasks.count_task
@maple_tasks.multi_hop
@maple_tasks.query_node
@maple_tasks.query_num
def path_num_between_non_paper_pair(subgraph:DiGraph, class_bin:int=None):
    source_target_pairs = list(itertools.product([MapleMeta.PAPER, MapleMeta.AUTHOR, MapleMeta.VENUE], [MapleMeta.PAPER, MapleMeta.AUTHOR, MapleMeta.VENUE]))
    source_type, target_type = random.choice(source_target_pairs)
    while (source_type, target_type) == (MapleMeta.PAPER, MapleMeta.PAPER):
        source_type, target_type = random.choice(source_target_pairs)
    return count_path_between_nodes(subgraph, source_type, target_type, class_bin)

# ------------------------------- Bool tasks ------------------------------

# ---------------------- Single hop ----------------------

# --------- linked_by_edge ---------

@maple_tasks.structure_aware
@maple_tasks.seen_task
@maple_tasks.bool_task
@maple_tasks.single_hop
@maple_tasks.query_node
@maple_tasks.query_edge
def has_cite_edge(subgraph:DiGraph, class_bin:int=None):
    return linked_by_edge(subgraph, MapleMeta.PAPER, MapleMeta.PAPER, MapleMeta.CITE, class_bin)

@maple_tasks.structure_aware
@maple_tasks.unseen_task
@maple_tasks.bool_task
@maple_tasks.single_hop
@maple_tasks.query_node
@maple_tasks.query_edge
def has_non_cite_edge(subgraph:DiGraph, class_bin:int=None):
    target_type, attribute_type = random.choice([(MapleMeta.AUTHOR, MapleMeta.WRITTEN_BY), (MapleMeta.VENUE, MapleMeta.PUBLISHED_ON)])
    return linked_by_edge(subgraph, MapleMeta.PAPER, target_type, attribute_type, class_bin)

# ---------------------- Multi hop ----------------------

# --------- has_path_between_nodes ---------

@maple_tasks.structure_aware
@maple_tasks.seen_task
@maple_tasks.bool_task
@maple_tasks.multi_hop
@maple_tasks.query_node
def has_path_between_paper_paper(subgraph:DiGraph, class_bin:int=None):
    return has_path_between_nodes(subgraph, MapleMeta.PAPER, MapleMeta.PAPER, class_bin)

@maple_tasks.structure_aware
@maple_tasks.unseen_task
@maple_tasks.bool_task
@maple_tasks.multi_hop
@maple_tasks.query_node
def has_path_between_non_paper_pair(subgraph:DiGraph, class_bin:int=None):
    source_target_pairs = list(itertools.product([MapleMeta.PAPER, MapleMeta.AUTHOR, MapleMeta.VENUE], [MapleMeta.PAPER, MapleMeta.AUTHOR, MapleMeta.VENUE]))
    source_type, target_type = random.choice(source_target_pairs)
    while (source_type, target_type) == (MapleMeta.PAPER, MapleMeta.PAPER):
        source_type, target_type = random.choice(source_target_pairs)
    return has_path_between_nodes(subgraph, source_type, target_type, class_bin)

# ------------------------------- Path tasks ------------------------------

# --------- find_path_of_length ---------

@maple_tasks.structure_aware
@maple_tasks.seen_task
@maple_tasks.path_task
@maple_tasks.multi_hop
@maple_tasks.query_node
@maple_tasks.query_num
def find_path_of_length_between_paper_paper(subgraph:DiGraph, class_bin:int=None):
    return find_path_of_length(subgraph, MapleMeta.PAPER, MapleMeta.PAPER, class_bin)

@maple_tasks.structure_aware
@maple_tasks.unseen_task
@maple_tasks.path_task
@maple_tasks.multi_hop
@maple_tasks.query_node
@maple_tasks.query_num
def find_path_of_length_between_non_paper_pair(subgraph:DiGraph, class_bin:int=None):
    source_target_pairs = list(itertools.product([MapleMeta.PAPER, MapleMeta.AUTHOR, MapleMeta.VENUE], [MapleMeta.PAPER, MapleMeta.AUTHOR, MapleMeta.VENUE]))
    source_type, target_type = random.choice(source_target_pairs)
    while (source_type, target_type) == (MapleMeta.PAPER, MapleMeta.PAPER):
        source_type, target_type = random.choice(source_target_pairs)
    return find_path_of_length(subgraph, source_type, target_type, class_bin)

# --------- find_shortest_path ---------

@maple_tasks.structure_aware
@maple_tasks.seen_task
@maple_tasks.path_task
@maple_tasks.multi_hop
@maple_tasks.query_node
def find_shortest_path_between_paper_paper(subgraph:DiGraph, class_bin:int=None):
    return find_shortest_path(subgraph, MapleMeta.PAPER, MapleMeta.PAPER, class_bin)

@maple_tasks.structure_aware
@maple_tasks.unseen_task
@maple_tasks.path_task
@maple_tasks.multi_hop
@maple_tasks.query_node
def find_shortest_path_between_non_paper_pair(subgraph:DiGraph, class_bin:int=None):
    source_target_pairs = list(itertools.product([MapleMeta.PAPER, MapleMeta.AUTHOR, MapleMeta.VENUE], [MapleMeta.PAPER, MapleMeta.AUTHOR, MapleMeta.VENUE]))
    source_type, target_type = random.choice(source_target_pairs)
    while (source_type, target_type) == (MapleMeta.PAPER, MapleMeta.PAPER):
        source_type, target_type = random.choice(source_target_pairs)
    return find_shortest_path(subgraph, source_type, target_type, class_bin)

# ------------------------------- Graph tasks ------------------------------

# --------- find_ego_graph ---------

@maple_tasks.structure_aware
@maple_tasks.seen_task
@maple_tasks.graph_task
@maple_tasks.single_hop
@maple_tasks.query_node
@maple_tasks.query_num
def find_1_hop_ego_graph_for_paper(subgraph:DiGraph, class_bin:int=None):
    return find_ego_graph(subgraph, MapleMeta.PAPER, 1, class_bin)

@maple_tasks.structure_aware
@maple_tasks.unseen_task
@maple_tasks.graph_task
@maple_tasks.single_hop
@maple_tasks.query_node
@maple_tasks.query_num
def find_1_hop_ego_graph_for_author_or_venue(subgraph:DiGraph, class_bin:int=None):
    center_type = random.choice([MapleMeta.AUTHOR, MapleMeta.VENUE])
    return find_ego_graph(subgraph, center_type, 1, class_bin)

@maple_tasks.structure_aware
@maple_tasks.seen_task
@maple_tasks.graph_task
@maple_tasks.multi_hop
@maple_tasks.query_node
@maple_tasks.query_num
def find_2_hop_ego_graph_for_paper(subgraph:DiGraph, class_bin:int=None):
    return find_ego_graph(subgraph, MapleMeta.PAPER, 2, class_bin)

@maple_tasks.structure_aware
@maple_tasks.unseen_task
@maple_tasks.graph_task
@maple_tasks.multi_hop
@maple_tasks.query_node
@maple_tasks.query_num
def find_2_hop_ego_graph_for_author_or_venue(subgraph:DiGraph, class_bin:int=None):
    center_type = random.choice([MapleMeta.AUTHOR, MapleMeta.VENUE])
    return find_ego_graph(subgraph, center_type, 2, class_bin)

# ---------------------------------------- Inductive reasoning instructions ----------------------------------------

    # def paper_clf(self, G:DiGraph, attribute_map, sample_size:int):
    #     # TODO: Is this too easy, remove all examples' category?
        
    #     # Find candidate nodes
    #     cate_edges = G.get_typed_edges(self.BELONG_TO)

    #     # Sample target edges and remove answers from graph
    #     target_edges = random.sample(cate_edges, sample_size)
    #     G.remove_edges_from(target_edges)
        
    #     # Build the subgraph, question and find answer for each target edge
    #     dataset = []
    #     for paper_node, answer in target_edges:
    #         question = f"Predict the category of paper {paper_node}."
    #         subgraph = G.sample_subgraph(
    #             paper_node, 2, self.edge_sample_size
    #         )
    #         dataset.append(pack_data(subgraph, question, answer))
        
    #     # Add answers back to graph
    #     G.add_edges_from(target_edges, type=self.BELONG_TO)
        
    #     return dataset

@maple_tasks.inductive_reasoning
@maple_tasks.seen_task
@maple_tasks.bool_task
def citation_prediction(G:DiGraph, sample_size:int, edge_sample_size:Dict[str, Any]=None):
    return build_edge_prediction(G, MapleMeta.CITE, sample_size, edge_sample_size)

@maple_tasks.inductive_reasoning
@maple_tasks.seen_task
@maple_tasks.bool_task
def paper_author_prediction(G:DiGraph, sample_size:int, edge_sample_size:Dict[str, Any]=None):
    return build_edge_prediction(G, MapleMeta.WRITTEN_BY, sample_size, edge_sample_size)

@maple_tasks.inductive_reasoning
@maple_tasks.unseen_task
@maple_tasks.bool_task
def paper_venue_prediction(G:DiGraph, sample_size:int, edge_sample_size:Dict[str, Any]=None):
    return build_edge_prediction(G, MapleMeta.PUBLISHED_ON, sample_size, edge_sample_size)
