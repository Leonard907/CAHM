from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from argparse import ArgumentParser
from utils import *
from psg_select import *
from tqdm import tqdm
from configs.zs_config import *
from vllm import LLM, SamplingParams
import json

from transformers.utils import logging
logging.set_verbosity_error()

HIER_RET_METHOD_MAP = {
    "sent": sent_select,
    "summ": summ_select,
}

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, default="sfd")
parser.add_argument("--ret_method", type=str, default="summ")
parser.add_argument("--output_file", type=str, default="")
parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
parser.add_argument("--tensor_parallel_size", type=int, default=1)
parser.add_argument("--expand_factor", type=int, default=1)

args = parser.parse_args()
dataset_name = args.dataset
dataset_config = HIER_DATASET_CONFIG[dataset_name]

model_path = args.model_path
tokenizer = AutoTokenizer.from_pretrained(model_path)
dataset = load_dataset(dataset_config["dataset"])['test']
sampling_params = SamplingParams(temperature=0, top_p=0.9, max_tokens=dataset_config["max_new_tokens"])
llm = LLM(model=model_path, max_model_len=30000, tensor_parallel_size=args.tensor_parallel_size) # Hard restriction to save space
ret_func = HIER_RET_METHOD_MAP[args.ret_method]
expand_factor = args.expand_factor
final_summaries = {}

MAX_SUMMARY_LENGTH = dataset_config["max_summary_length"]

def summarize(text, chunks, merge=True, context="", max_len=MAX_SUMMARY_LENGTH, final=False):
    estimate_topk = max(1, round(max_len / HIER_RET_UNIT) * expand_factor)
    if not merge:
        input_text = dataset_config["chunk_prompt"].format(text)
    else:
        if context == "":
            input_text = dataset_config["merge_prompt"].format(text)
        else:
            input_text = dataset_config["merge_context_prompt"].format(context, text)
    response = vllm_generate(llm, sampling_params, input_text)
    if final:
        return response, []
    # Do retrieval
    selected_chunks = ret_func(response, chunks, topk=estimate_topk)
    return response, selected_chunks

def summarize_batch(texts, chunks, max_len=MAX_SUMMARY_LENGTH):
    estimate_topk = max(1, round(max_len / HIER_RET_UNIT) * expand_factor)
    input_texts = [dataset_config["chunk_prompt"].format(texts[i]) for i in range(len(texts))]
    abs_summaries = vllm_generate(llm, sampling_params, input_texts, mode="batch")
    selected_chunks = [ret_func(abs_summaries[i], chunks[i], topk=estimate_topk) for i in range(len(texts))]
    return abs_summaries, selected_chunks

def recursive_summarize(summaries, prev_chunks, chunks, max_summary_length):
    def merge_summaries(summary_list):
        return "\n".join([f"Summary {i+1}: {' '.join(sl)}" for i, sl in enumerate(summary_list)])

    def count_tokens(text):
        return len(tokenizer.encode(text))

    level = 0
    chunk_passages = [chunk_texts(chunk, tokenizer, HIER_RET_UNIT) for chunk in chunks]
    # Do top level
    first_level_abs_summaries, first_level_selected_chunks = summarize_batch(chunks, 
        chunk_passages,
        max_len=max_summary_length
    )
    summaries[level] = first_level_selected_chunks

    while len(summaries[level]) > 1:
        next_level = level + 1
        summaries[next_level] = []
        context = ""
        context_chunks = []
        merged_chunks = []
        final_merge = True
        
        for chunk_list in summaries[level]:
            merged_chunks.append(chunk_list)
            merged_text = merge_summaries(merged_chunks)  # Use all merged_chunks instead of just the current chunk_list
            if context == "":
                prompt = dataset_config["merge_prompt"].format(merged_text)
            else:
                prompt = dataset_config["merge_context_prompt"].format(context, merged_text)
            
            if count_tokens(prompt) > dataset_config["max_context_length"] * expand_factor:
                final_merge = False
                merged_chunks.pop()
                merged_text = merge_summaries(merged_chunks)  # Use all remaining merged_chunks
                flat_chunk_list = [item for sublist in merged_chunks for item in sublist] + context_chunks
                summary, selected_chunks = summarize(merged_text, flat_chunk_list, merge=True, context=context, max_len=max_summary_length)
                summaries[next_level].append(selected_chunks)
                context = summary
                context_chunks = selected_chunks
                merged_chunks = [chunk_list]
        
        if merged_chunks:
            merged_text = merge_summaries(merged_chunks)  # Use all remaining merged_chunks
            flat_chunk_list = [item for sublist in merged_chunks for item in sublist] + context_chunks
            summary, selected_chunks = summarize(merged_text, flat_chunk_list, merge=True, final=final_merge, context=context, max_len=max_summary_length)
            if final_merge:
                summaries[next_level].append(summary)
            else:
                summaries[next_level].append(selected_chunks)
        
        level = next_level

    return summaries

for i in tqdm(range(len(dataset))):
    input_text = parse_non_ascii(dataset[i]['input'])
    output_text = parse_non_ascii(dataset[i]['output'])
    chunks = chunk_texts(input_text, tokenizer, dataset_config["chunk_size"])
    final_summary_dict = recursive_summarize({}, {}, chunks, dataset_config["max_summary_length"])
    final_summaries[i] = final_summary_dict[max(final_summary_dict.keys())][0]

if args.output_file == "":
    json.dump(final_summaries, open(dataset_config["output_file"], "w"), indent=4)
else:
    json.dump(final_summaries, open(args.output_file, "w"), indent=4)