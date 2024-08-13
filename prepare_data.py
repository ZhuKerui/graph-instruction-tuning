import argparse
import os
from transformers import AutoTokenizer, AutoConfig
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', type=str)
parser.add_argument('--output-dir', type=str)
parser.add_argument('--files', type=str, nargs='+')
parser.add_argument('--tokenizer-model', type=str)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
config = AutoConfig.from_pretrained(args.tokenizer_model)
datasets = []
for fname in args.files:
    with open(os.path.join(args.input_dir, fname)) as f_in:
        datasets.append([json.loads(line) for line in f_in])
new_datasets = [[] for _ in range(len(datasets))]
for samples in tqdm(zip(*datasets), total=len(datasets[0])):
    max_length = max(*[len(tokenizer.encode(sample['prompt'])) for sample in samples])
    if max_length < config.max_position_embeddings - 500:
        for i in range(len(new_datasets)):
            new_datasets[i].append(samples[i])

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)
    
for new_dataset, fname in zip(new_datasets, args.files):
    with open(os.path.join(args.output_dir, fname), 'w') as f_out:
        f_out.write('\n'.join([json.dumps(sample) for sample in new_dataset]))