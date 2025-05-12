from transformers import AutoTokenizer
from datasets import load_dataset
from argparse import ArgumentParser
from utils import *
from tqdm import tqdm
from configs.zs_config import *
from openai import OpenAI
import torch
import json
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from transformers.utils import logging
from vllm import LLM, SamplingParams
logging.set_verbosity_error()

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, default="sfd")
parser.add_argument("--output_file", type=str, default="")
parser.add_argument("--tensor_parallel_size", type=int, default=1)
parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
parser.add_argument("--mode", type=str, default="ret")
args = parser.parse_args()
dataset_name = args.dataset
mode = args.mode
dataset_config = HIER_DATASET_CONFIG[dataset_name]

model_path = args.model_path
tokenizer = AutoTokenizer.from_pretrained(model_path)
dataset = load_dataset(dataset_config["dataset"])['test']
print_first = False
sampling_params = SamplingParams(temperature=0, top_p=0.9, max_tokens=dataset_config["max_new_tokens"])
llm = LLM(model=model_path, max_model_len=10000, tensor_parallel_size=args.tensor_parallel_size)

final_summaries = {}

MAX_SUMMARY_LENGTH = dataset_config["max_summary_length"]
MAX_CONTEXT_LENGTH = dataset_config["max_context_length"]
ATTR_CHUNK_SIZE = dataset_config["attr_chunk_size"]

def estimate_levels(num_chunks, max_context_len, summary_limit):
    # Pre-specify number of chunks at each level
    branch_factor = max_context_len // summary_limit
    num_chunks_levels = [1]
    top_layer_exact_fit = False
    while True:
        previous_num_chunks = num_chunks_levels[-1]
        next_num_chunks = previous_num_chunks * branch_factor + 1
        if next_num_chunks >= num_chunks:
            if len(num_chunks_levels) == 1:
                num_chunks_levels.append(num_chunks)
            else:
                num_chunks_levels[-1] = num_chunks
            top_layer_exact_fit = next_num_chunks == num_chunks
            break
        else:
            num_chunks_levels.append(next_num_chunks)
    # Calculate summary limits
    summary_limits = [summary_limit] # Last step specified
    division_factor = branch_factor + 1
    for _ in range(1, len(num_chunks_levels) - 1):
        next_summary_limit = max_context_len // division_factor # Merge steps constant size
        summary_limits.append(next_summary_limit)

    division_factor = division_factor if top_layer_exact_fit else division_factor + 1 # Whether underestimate first step
    summary_limits.append(max_context_len // division_factor)
    num_chunks_levels.reverse()
    summary_limits.reverse()
    summary_limits = [int(limit * WORD_TOKEN_RATIO) for limit in summary_limits]
    return num_chunks_levels, summary_limits

def summarize(text, attr_texts, attr_passages, final=False, merge=True, context="", max_len=MAX_SUMMARY_LENGTH):
    text_to_use = None
    if mode == "abs":
        text_to_use = text
    elif mode == "ret":
        text_to_use = attr_texts
    if not merge:
        _, labels = split_by_labels(text_to_use)
        input_text = dataset_config["chunk_attr_prompt"].format(text_to_use)
    else:
        if context == "":
            if mode == "mix":
                input_text = dataset_config["merge_aggr_attr_prompt"].format(text, " ".join(attr_passages))
            else:
                input_text = dataset_config["merge_attr_prompt"].format(text_to_use)
        else:
            if mode == "mix":
                input_text = dataset_config["merge_context_aggr_attr_prompt"].format(context, text, " ".join(attr_passages))
            else:
                input_text = dataset_config["merge_context_attr_prompt"].format(context, text_to_use)
    response = vllm_generate(llm, sampling_params, input_text)
    response_passages = get_top_attr(attr_passages, response, MAX_SUMMARY_LENGTH // ATTR_CHUNK_SIZE)
    return remove_citations(response), response_passages

def summarize_batch(texts, attr_texts, max_len=MAX_SUMMARY_LENGTH):
    all_labels = [", ".join(split_by_labels(text)[1]) for text in texts]
    input_texts = [dataset_config["chunk_attr_prompt"].format(texts[i], all_labels[i], max_len) for i in range(len(texts))]
    responses = vllm_generate(llm, sampling_params, input_texts, mode="batch")
    response_passages = [get_top_attr(attr_texts[i], responses[i], MAX_SUMMARY_LENGTH // ATTR_CHUNK_SIZE) for i in range(len(texts))]
                
    return [remove_citations(response) for response in responses], response_passages

def recursive_summarize(summaries, attr_passages, chunks, attr_texts, max_context_length, max_summary_length):
    def merge_summaries(summary_list=None, passage_list=None):
        if summary_list is not None:
            merged_text = " ".join(summary_list)
            return cite_text(tokenizer.encode(merged_text), ATTR_CHUNK_SIZE, tokenizer)
        else:
            merged_text = " ".join([" ".join(passage) for passage in passage_list])
            return cite_text(tokenizer.encode(merged_text), ATTR_CHUNK_SIZE, tokenizer)
    
    def count_tokens(text):
        return len(tokenizer.encode(text))

    level = 0
    pure_summaries, selected_attr_texts = summarize_batch(chunks, attr_texts, max_len=max_summary_length)
    summaries[level] = pure_summaries
    attr_passages[level] = selected_attr_texts

    while len(summaries[level]) > 1:
        next_level = level + 1
        summaries[next_level] = []
        attr_passages[next_level] = []
        context = ""
        merged_summaries = []
        merged_attr_passages = []
        final_merge = True
        
        for summary, attr_passage in zip(summaries[level], attr_passages[level]):
            merged_summaries.append(summary)
            merged_attr_passages.append(attr_passage)

            merged_abs, passages_abs = merge_summaries(summary_list=merged_summaries)
            merged_attr_texts, passages_attr = merge_summaries(passage_list=merged_attr_passages)
            merge_to_use = merged_abs if mode == "abs" else merged_attr_texts
            
            if mode == "mix":
                if context == "":
                    prompt = dataset_config["merge_aggr_attr_prompt"].format(merged_abs, " ".join(passages_attr))
                else:
                    prompt = dataset_config["merge_context_aggr_attr_prompt"].format(context, merged_abs, " ".join(passages_attr))
            else:
                if context == "":
                    prompt = dataset_config["merge_attr_prompt"].format(merge_to_use)
                else:
                    prompt = dataset_config["merge_context_attr_prompt"].format(context, merge_to_use)
            
            if count_tokens(prompt) > max_context_length:
                final_merge = False
                merged_summaries.pop()  # Remove the last summary that caused overflow
                merged_attr_passages.pop()
                merged_abs, passages_abs = merge_summaries(summary_list=merged_summaries)
                merged_attr_texts, passages_attr = merge_summaries(passage_list=merged_attr_passages)
                passage_to_use = passages_abs if mode == "abs" else passages_attr
                new_summary, new_attr_passages = summarize(merged_abs, merged_attr_texts, attr_passages=passage_to_use, merge=True, context=context, max_len=max_summary_length)
                summaries[next_level].append(new_summary)
                attr_passages[next_level].append(new_attr_passages)
                context = new_summary
                merged_summaries = [summary]  # Start new merge with the overflow summary
                merged_attr_passages = [attr_passage]
        
        if len(merged_summaries) >= 1:
            merged_abs, passages_abs = merge_summaries(summary_list=merged_summaries)
            merged_attr_texts, passages_attr = merge_summaries(passage_list=merged_attr_passages)
            passage_to_use = passages_abs if mode == "abs" else passages_attr
            new_summary, new_attr_passages = summarize(merged_abs, merged_attr_texts, attr_passages=passage_to_use, final=final_merge, merge=True, context=context, max_len=max_summary_length)
            summaries[next_level].append(new_summary)
            attr_passages[next_level].append(new_attr_passages)
            context = new_summary
        
        level = next_level
    return summaries

for i in tqdm(range(len(dataset))):
    chunks, attr_texts = chunk_texts_with_attribution(parse_non_ascii(dataset[i]['input']),
                                      tokenizer, 
                                      dataset_config["chunk_size"],
                                      ATTR_CHUNK_SIZE)
    result = recursive_summarize(
        {},
        {},
        chunks,
        attr_texts,
        MAX_CONTEXT_LENGTH,
        MAX_SUMMARY_LENGTH * WORD_TOKEN_RATIO
    )
    final_summaries[i] = result[max(result.keys())][0]

if args.output_file == "":
    json.dump(final_summaries, open(dataset_config["output_file"], "w"), indent=4)
else:
    json.dump(final_summaries, open(args.output_file, "w"), indent=4)
