import argparse
import json
import os
import vllm
import torch
from tqdm import tqdm
from vllm.lora.request import LoRARequest

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model",
        type=str
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Huggingface model name or path.")
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        help="Huggingface tokenizer name or path."
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--input_files", 
        type=str, 
        nargs="+",
        help="Input .jsonl files, with each line containing `id` and `prompt` or `messages`.")
    parser.add_argument(
        "--output_file",
        type=str,
        default="output/model_outputs.jsonl",
        help="Output .jsonl file, with each line containing `id`, `prompt` or `messages`, and `output`.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch size for prediction.")
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.7,
        help="Max memory usage for vLLM caching.")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="maximum number of new tokens to generate.")
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="whether to use sampling ; use greedy decoding otherwise.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature for sampling.")
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="top_p for sampling.")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    # check if output directory exists
    if args.output_file is not None:
        output_dir = os.path.dirname(args.output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # load the data
    instances = []
    for input_file in args.input_files:
        with open(input_file, "r") as f:
            instances += [json.loads(x) for x in f.readlines()]
    #instances = instances[:20]
    #instances = [inst for i, inst in enumerate(instances) if i + 1 in {3, 4, 10, 12, 13, 14}]
    
    prompts = []
    for instance in instances:
        prompts.append(instance["prompt"])

    if "gpt" not in args.model_name_or_path:
        model = vllm.LLM(
            model=args.base_model,
            # tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path,
            # tokenizer_mode="slow" if args.use_slow_tokenizer else "auto",
            # tensor_parallel_size=torch.cuda.device_count(),
            # gpu_memory_utilization=args.gpu_memory_utilization,
            enable_lora=True,
            max_lora_rank=64
        )
        
        sampling_params = vllm.SamplingParams(
            temperature=args.temperature if args.do_sample else 0,
            max_tokens=args.max_new_tokens
        )
        
        if '_merged' in args.model_name_or_path:
            outputs = model.generate(prompts, sampling_params, lora_request=LoRARequest("sql_adapter", 1, args.model_name_or_path.split('_merged')[0]))
        else:
            outputs = model.generate(prompts, sampling_params)
        req_inst = dict()
        for it in outputs:
            req_id = it.request_id
            if req_id not in req_inst:
                req_inst[req_id] = len(req_inst)
            instances[req_inst[req_id]]["output"] = it.outputs[0].text
    
    else:
        import openai
        import time
        client = openai.OpenAI()
        i = 0
        pbar = tqdm(total=len(prompts))
        while i < len(prompts):
            try:
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompts[i]}],
                    model=args.model_name_or_path,
                    max_tokens=args.max_new_tokens
                )
                instances[i]["output"] = chat_completion.choices[0].message.content
                i += 1
                pbar.update(1)
            except openai.BadRequestError:
                instances[i]["output"] = ''
                i += 1
                pbar.update(1)
            except openai.RateLimitError:
                print("RateLimitError: Sleep 10 sec")
                time.sleep(10)
            
    
    with open(args.output_file, 'w') as f:
        for instance in instances:
            f.write(json.dumps(instance) + '\n')
    
    print("Done.")
