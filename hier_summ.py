from transformers import AutoTokenizer
from datasets import load_dataset
from argparse import ArgumentParser
from utils import *
from tqdm import tqdm
from configs.zs_config import *
from vllm import LLM, SamplingParams
import torch
import json

from transformers.utils import logging
logging.set_verbosity_error()

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, default="sfd")
parser.add_argument("--output_file", type=str, default="")
parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
parser.add_argument("--tensor_parallel_size", type=int, default=1)
args = parser.parse_args()
dataset_name = args.dataset
dataset_config = HIER_DATASET_CONFIG[dataset_name]

model_path = args.model_path
tokenizer = AutoTokenizer.from_pretrained(model_path)
dataset = load_dataset(dataset_config["dataset"])['test']
print_first = False
sampling_params = SamplingParams(temperature=0, top_p=0.9, max_tokens=dataset_config["max_new_tokens"])
llm = LLM(model=model_path, max_model_len=15000, tensor_parallel_size=args.tensor_parallel_size) # Hard restriction to save space
final_summaries = {}

MAX_SUMMARY_LENGTH = dataset_config["max_summary_length"]

def summarize(text, merge=True, final=False, context="", max_len=MAX_SUMMARY_LENGTH):
    if not merge:
        input_text = dataset_config["chunk_prompt"].format(text, max_len)
    else:
        if context == "":
            input_text = dataset_config["merge_prompt"].format(text, max_len)
        else:
            input_text = dataset_config["merge_context_prompt"].format(context, text, max_len)
    response = vllm_generate(llm, sampling_params, input_text)
    return response

def summarize_batch(texts, max_len=MAX_SUMMARY_LENGTH):
    input_texts = [dataset_config["chunk_prompt"].format(texts[i], max_len) for i in range(len(texts))]
    responses = vllm_generate(llm, sampling_params, input_texts, mode="batch")
    return responses

def recursive_summarize(summaries, chunks, max_context_length, max_summary_length):
    def merge_summaries(summary_list):
        return "\n".join([f"Summary {i+1}: {summary_list[i]}" for i in range(len(summary_list))])

    level = 0
    summaries[level] = summarize_batch(chunks, max_len=max_summary_length)
    
    def count_tokens(text):
        return len(tokenizer.encode(text))

    while len(summaries[level]) > 1:
        next_level = level + 1
        summaries[next_level] = []
        context = ""
        merged_summaries = []
        final = True
        
        for summary in summaries[level]:
            merged_summaries.append(summary)
            merged_text = merge_summaries(merged_summaries)
            if context == "":
                prompt = dataset_config["merge_prompt"].format(merged_text, max_summary_length)
            else:
                prompt = dataset_config["merge_context_prompt"].format(context, merged_text, max_summary_length)
            
            if count_tokens(prompt) > max_context_length:
                final = False
                merged_summaries.pop()  # Remove the last summary that caused overflow
                merged_text = merge_summaries(merged_summaries)
                new_summary = summarize(merged_text, merge=True, final=False, context=context, max_len=max_summary_length)
                summaries[next_level].append(new_summary)
                context = new_summary
                merged_summaries = [summary]  # Start new merge with the overflow summary
        
        if len(merged_summaries) >= 1:
            merged_text = merge_summaries(merged_summaries)
            new_summary = summarize(merged_text, merge=True, final=final, context=context, max_len=max_summary_length)
            summaries[next_level].append(new_summary)
            context = new_summary
        
        level = next_level
    return summaries

first_level_summaries = {}
first_level_summaries_path = f'{dataset_name}_first_level_summaries.json'

for i in tqdm(range(len(dataset))):
    input_text = parse_non_ascii(dataset[i]['input'])
    output_text = parse_non_ascii(dataset[i]['output'])
    chunks = chunk_texts(input_text, tokenizer, dataset_config["chunk_size"])
    try:
        input_summaries = json.load(open(first_level_summaries_path, 'r'))
        input_summaries[0] = input_summaries["0"]
        del input_summaries["0"]
    except:
        input_summaries = {}
    final_summary_dict = recursive_summarize(input_summaries, chunks, dataset_config["max_context_length"], dataset_config["max_summary_length"] * WORD_TOKEN_RATIO)
    final_summaries[i] = final_summary_dict[max(final_summary_dict.keys())][0]
    first_level_summaries[i] = final_summary_dict[0][0]

if args.output_file == "":
    json.dump(final_summaries, open(dataset_config["output_file"], "w"), indent=4)
else:
    json.dump(final_summaries, open(args.output_file, "w"), indent=4)
