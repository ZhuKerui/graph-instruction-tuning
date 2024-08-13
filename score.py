import argparse
import json
from collections import defaultdict
import numpy as np
import re

def F1(ground_truth, model_output):
    ground_truth = set(ground_truth)
    model_output = set(model_output)
    correct_predictions = len(ground_truth & model_output)
    precision = correct_predictions / len(model_output) if model_output else 0
    recall = correct_predictions / len(ground_truth) if ground_truth else 0
    if precision + recall > 0:
        F1 = 2 * (precision * recall) / (precision + recall)
    else:
        F1 = 0
    return F1

def square_error(ground_truth, model_output):
    ground_truth = float(ground_truth)
    model_output = float(model_output)
    return (ground_truth - model_output) ** 2

def split_path_answer(answer:str):
    return {tuple([node.strip() for node in a.strip('( ').split(',')]) for a in answer.split(')')}

def split_pair_answer(answer:str):
    return {frozenset([node.strip() for node in a.strip('( ').split(',')]) for a in answer.split(')')}

def score_sample(d, tasks):
    delimiters = "[ ,;\n\t]"  # spaces, commas, semicolons, newlines, and tabs
    dataset_name = d["dataset"]
    gt = d["completion"]
    gen = d["output"].strip()
    if gen: # TODO: Remove this condition after length problem is solved
        if dataset_name in tasks.count_tasks:
            if gt.isnumeric() and gen.isnumeric():
                score = square_error(gt, gen)
            else:
                return -1, gt, gen
        elif dataset_name in tasks.bool_tasks:
            gt = gt.lower()
            gen = re.split(delimiters, gen)[0].lower()
            return int(gt == gen), gt, gen
        elif dataset_name in tasks.path_tasks:
            gt = sorted(split_path_answer(gt))
            # gen = sorted(list({' '.join(re.split(delimiters, seq)) for seq in gen.split('\n')}))
            gen = sorted(split_path_answer(gen))
            score = F1(gt, gen)
        elif dataset_name in (tasks.pair_tasks + tasks.graph_tasks):
            gt = sorted(split_pair_answer(gt))
            gen = sorted(split_pair_answer(gen))
            score = F1(gt, gen)
        else:
            gt = sorted(list(set(gt.split())))
            gen = sorted(list(set(re.split(delimiters, gen))))
            score = F1(gt, gen)
        return score, gt, gen
    return -1, gt, gen

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True, help=".jsonl file, with each line containing `id`, `prompt` or `messages`, and `output`.")
    parser.add_argument("--data_source", type=str, default="amazon", choices=["amazon", "maple"])
    args = parser.parse_args()
    
    if args.data_source == 'amazon':
        from amazon.generator import amazon_tasks as tasks
    else:
        from maple.generator import maple_tasks as tasks
    
    em_records = defaultdict(list)
    metric_records = defaultdict(list)

    with open(args.data_file, 'r') as f:
        for line in f.readlines():
            d = json.loads(line)
            dataset_name = d["dataset"]
            score, gt, gen = score_sample(d, tasks)
            metric_records[dataset_name].append(score)
            em_records[dataset_name].append(gt == gen)
        
    for dataset_name in em_records:
        em_record, metric_record = em_records[dataset_name], metric_records[dataset_name]
        print(f"{dataset_name} ({len(em_record)} instances): EM {np.mean(em_record) * 100:.2f} %,", end=' ')
        #if dataset_name in (tasks.count_tasks + tasks.bool_tasks):
        if dataset_name in tasks.count_tasks:
            print(f"MSE {np.mean(metric_record)}")
        else:
            print(f"F1 {np.mean([s if s >= 0 else 0 for s in metric_record]) * 100:.2f} %")
    
