import os.path as osp
import json
import gzip
import pandas as pd
from collections import defaultdict

from Graph import DiGraph

def read_tsv(fname):
    df = pd.read_csv(fname, sep='\t')
    for row in df.itertuples(index=False):
        yield row

def get_graph_json(d):
    info = defaultdict(lambda: defaultdict(list))
    for center, neighbor, relation in d['Edges']:
        info[center][relation].append(neighbor)
    return info

def remove_text_attribute(sample:dict):
    prompt:str = sample['prompt']
    start_idx, end_idx = prompt.index('[Graph]\n')+8, prompt.index('[Text Attributes]')
    new_lines = []
    for line in prompt[start_idx:end_idx].split('\n'):
        if '<text>' in line:
            name_start = line.index('<text>') - 5
            new_line = line[:name_start]
            new_line = new_line.strip(' ,')
            if ' ' not in new_line:
                continue
            new_lines.append(new_line + '\n')
    new_prompt = prompt[:start_idx] + ''.join(new_lines) + prompt[end_idx:]
    sample['prompt'] = new_prompt
    
def pack_data(graph:DiGraph, question, answer):
    return {
        "Args": question,
        "Answer": answer,
        "Edges": list(graph.edges.data("type"))
    }

def get_path(*args):
    return osp.join(*args)

def read_gzip(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.loads(l)

def remove_special_char(node_name:str):
    return ' '.join(node_name.replace('#', ' ').split())
