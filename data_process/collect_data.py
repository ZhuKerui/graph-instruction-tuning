import argparse
import os
from data_utils import *
import random
import yaml
from tqdm import tqdm
from time import time
from typing import List, Dict, Any, Tuple
from question_generator import QuestionGenerator, DatasetMetaData

training_ratio = 0.7

def encode_dataset(raw_dataset, dataset_name):
    dataset = []
    for i, d in enumerate(raw_dataset):
        dataset.append({
            "dataset": dataset_name,
            "id": f"{dataset_name}-{i}",
            "args": d['Args'],
            "context": get_graph_json(d),
            "answer": d['Answer']
        })
    return dataset

def build_all_datasets(tasks:QuestionGenerator, args, task_args:Dict[str, Dict[str, Any]]=None):
    print("\nBuild all instructions and datasets")
    for func_name, func in (tasks.structure_aware_tasks + tasks.inductive_reasoning_tasks):
        # if func_name != "has_non_cite_edge":
        #     continue
        random.seed(42)
        print(func_name)
        kwargs = {}
        if task_args and func_name in task_args:
            kwargs = task_args[func_name]
        start_time = time()
        raw_dataset = func(
            args.sample_size,
            **kwargs
        )
        
        encoded_dataset = encode_dataset(raw_dataset, func_name)
    
        # Train test split
        train_dataset, test_dataset = [], []
        dataset_size = len(encoded_dataset)
        train_inds = set(random.sample(range(dataset_size), int(dataset_size * training_ratio)))
            
        # Save dataset into file
        for i, d in enumerate(encoded_dataset):
            dataset = train_dataset if i in train_inds else test_dataset
            dataset.append(json.dumps(d))
            
        with open(args.train_fname, 'a') as f_out:
            f_out.write('\n'.join(train_dataset) + '\n')
        
        with open(args.test_fname, 'a') as f_out:
            f_out.write('\n'.join(test_dataset) + '\n')
        
        print(f"{func_name} done (Elasped time: {time() - start_time:.2f} sec)")

def encode_instruction_example(instruction:str, input:str, output:str, prompt_template:str, completion_template:str, eos_token=None):
    assert input
    
    prompt = prompt_template.format(instruction=instruction.strip(), input=input.strip())
    completion = completion_template.format(output=output.strip())

    data = {
        "prompt": prompt,
        "completion": completion + eos_token if eos_token else completion,
    }
    
    return data

def dict_to_common_format(info:dict, format:str, meta:DatasetMetaData):
    updated_info = {}
    for center, edge_info in info.items():
        updated_edge_info = {}
        for relation, neighbors in edge_info.items():
            if relation in meta.EDGE_ATTRBUTES:
                updated_edge_info[relation] = neighbors
            else:
                updated_edge_info[relation] = neighbors[0]
        if updated_edge_info:
            updated_info[center] = updated_edge_info
    
    if format == 'json':
        return json.dumps(updated_info)
    elif format == 'yaml':
        return yaml.dump(updated_info)
    elif format == 'natural':
        descriptions = []
        for center, edge_info in updated_info.items():
            temp_descriptions = []
            for relation, neighbors in edge_info.items():
                description_template = meta.ATTRIBUTE_DESCRIPTION.get(relation, 'has a ' + relation + ' attribute of {tails}')
                tails = ', '.join(neighbors) if isinstance(neighbors, list) else str(neighbors)
                temp_descriptions.append(description_template.format(tails=tails))
            descriptions.append(f"{center} {', '.join(temp_descriptions)}.")
        return ' '.join(descriptions)
    elif format == 'dot':
        descriptions = []
        for center, edge_info in updated_info.items():
            for relation, neighbors in edge_info.items():
                if isinstance(neighbors, str):
                    descriptions.append(f"{center} [label=\"{neighbors}\"]")
                else:
                    descriptions.append(f"{center} -> {{{', '.join(neighbors)}}} [label=\"{relation}\"]")
        descriptions = "; ".join(descriptions)
        return f"digraph G {{ {descriptions} }}"
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", type=str, default="amazon", choices=["amazon", "maple"])
    parser.add_argument("-r", "--raw_dname", type=str, default="")
    parser.add_argument("-s", "--sample_size", type=int, default=800)
    parser.add_argument("-c", "--categories", type=str, nargs='+')
    args = parser.parse_args()
    
    args.domain_dir = osp.join(args.data_source, 'data')
    os.makedirs(args.domain_dir, exist_ok=True)
    args.train_fname = osp.join(args.domain_dir, "train.jsonl")
    args.test_fname = osp.join(args.domain_dir, "test.jsonl")
    args.raw_data_dir = osp.join(args.data_source, 'data', args.raw_dname)
    args.categories = sorted(args.categories)
    
    random.seed(42)
    
    if args.data_source == 'amazon':
        from amazon.generator import amazon_tasks as tasks, AmazonMeta
        meta = AmazonMeta()
       
    elif args.data_source == 'maple':
        from maple.generator import maple_tasks as tasks, MapleMeta
        meta = MapleMeta()
        
    else:
        raise NotImplementedError
    
    with open(osp.join(args.data_source, 'templates.json')) as f_in:
        templates = json.load(f_in)
        task_args = {task_name: task_templates['task_args'] for task_name, task_templates in templates['task_templates'].items() if 'task_args' in task_templates}
    
    # Build base dataset if not exist
    if not os.path.exists(args.test_fname) or not os.path.exists(args.train_fname):
        meta.load_global_graph(args)

        tasks.metadata = meta

        build_all_datasets(
            tasks,
            args,
            task_args
        )
    
    task_dict = {
        'node_task': tasks.node_tasks,
        'pair_task': tasks.pair_tasks,
        'count_task': tasks.count_tasks,
        'bool_task': tasks.bool_tasks,
        'path_task': tasks.path_tasks,
        'graph_task': tasks.graph_tasks,
        'inductive_task': tasks.inductive_reasoning_tasks
    }
    
    with open('answer_formats.json') as f_in:
        answer_formats = json.load(f_in)
    
    def get_output(task, answer):
        if task == 'node_task':
            return ' '.join(answer)
        if task in ['pair_task', 'path_task', 'graph_task']:
            return ' '.join(('(%s)' % ', '.join(seq) for seq in answer))
        return str(answer)

    format_descriptions = {
        'json': 'The graph is represented in the json format.',
        'yaml': 'The graph is represented in the yaml format.',
        'natural': 'The graph is described in natural language.',
        'dot': 'The graph is described in DOT (graph description language).'
    }

    for dataset_type in ['train', 'test']:
        random.seed(42)
        datasets = defaultdict(list)
        with open(osp.join(args.domain_dir, f"{dataset_type}.jsonl")) as f_in:
            samples:List[dict] = [json.loads(line) for line in f_in]
            
            # randomly choose templates
            encoding_templates:List[Tuple[str, str, float]] = random.choices(templates['encoding_templates'], weights=[w for _, _, w in templates['encoding_templates']], k=len(samples))

            for sample, encoding_template in tqdm(zip(samples, encoding_templates), total=len(samples)):
                if dataset_type == 'train' and sample['dataset'] not in tasks.seen_tasks:
                    continue
                
                # Build question, instruction and output
                question_template:str = random.choice(templates['task_templates'][sample['dataset']]['question_template'])
                for task, task_list in task_dict.items():
                    if sample['dataset'] in task_list:
                        break
                question = question_template.format(**sample.pop('args')) + ' ' + answer_formats[task]
                instruction = f"Given the graph and text information, answer the question - {question}"
                output = get_output(task, sample.pop('answer'))
                
                prompt_template, completion_template, _ = encoding_template
                
                # Build inputs
                context:dict = sample.pop('context')
                context_wo_text = {head: {edge: tails for edge, tails in v.items() if edge in meta.EDGE_ATTRBUTES} for head, v in context.items()}
                sample_infos = [(format_name, format_description, dict_to_common_format(context, format_name, meta), dict_to_common_format(context_wo_text, format_name, meta)) for format_name, format_description in format_descriptions.items()]
                if any([(not input) or (not input_wo_text) for _, _, input, input_wo_text in sample_infos]):
                    continue
                for format_name, format_description, input, input_wo_text in sample_infos:
                    # datasets[f'{dataset_type}_{format_name}'].append(json.dumps({**sample, **encode_instruction_example(f'{instruction} {format_description}', input, output, prompt_template, completion_template)}))
                    datasets[f'{dataset_type}_{format_name}'].append(json.dumps({**sample, **encode_instruction_example(f'{instruction} {format_description}', input_wo_text, output, prompt_template, completion_template)}))
            
        for dataset_fname, dataset in datasets.items():
            with open(osp.join(args.domain_dir, f"{dataset_fname}.jsonl"), 'w') as f_out:
                f_out.write('\n'.join(dataset))

    with open(osp.join(args.domain_dir, "args.json"), 'w') as f:
        json.dump(vars(args), f)
