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

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, default="sfd")
parser.add_argument("--ret_size", type=int, default=100)
parser.add_argument("--output_file", type=str, default="")
parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
parser.add_argument("--tensor_parallel_size", type=int, default=1)
args = parser.parse_args()
dataset_name = args.dataset
dataset_config = HIER_DATASET_CONFIG[dataset_name]

model_path = args.model_path
tokenizer = AutoTokenizer.from_pretrained(model_path)
dataset = load_dataset(dataset_config["dataset"])['test']
sampling_params = SamplingParams(temperature=0, top_p=0.9, max_tokens=dataset_config["max_new_tokens"])
llm = LLM(model=model_path, max_model_len=15000, tensor_parallel_size=args.tensor_parallel_size) # Hard restriction to save space
final_summaries = {}

MAX_SUMMARY_LENGTH = dataset_config["max_summary_length"]

def summarize(text, chunks, chunk_context, merge=True, context="", max_len=MAX_SUMMARY_LENGTH, final=False):
    estimate_topk = max(1, round(max_len / WORD_TOKEN_RATIO / HIER_RET_UNIT))
    if not merge:
        input_text = dataset_config["chunk_prompt"].format(text)
    else:
        if context == "":
            input_text = dataset_config["merge_aggr_prompt"].format(text, chunk_context)
        else:
            input_text = dataset_config["merge_context_aggr_prompt"].format(context, text, chunk_context)
    response = vllm_generate(llm, sampling_params, input_text)
    # Do retrieval
    flat_chunks = [chunk for sublist in chunks for chunk in sublist]
    selected_chunks = summ_select(response, flat_chunks, topk=estimate_topk)
    return response, selected_chunks

def summarize_batch(texts, chunks, max_len=MAX_SUMMARY_LENGTH):
    estimate_topk = max(1, round(max_len / WORD_TOKEN_RATIO / HIER_RET_UNIT))
    input_texts = [dataset_config["chunk_prompt"].format(texts[i]) for i in range(len(texts))]
    abs_summaries = vllm_generate(llm, sampling_params, input_texts, mode="batch")
    selected_chunks = [summ_select(abs_summaries[i], chunks[i], topk=estimate_topk) for i in range(len(texts))]
    return abs_summaries, selected_chunks

def recursive_summarize(summaries, abs_summaries, chunks, max_context_len, max_len):
    def merge_summaries(summary_list):
        return "\n".join([f"Summary {i+1}: {summary_list[i]}" for i in range(len(summary_list))])

    def merge_chunks(summary_list):
        return "\n".join([f"Context {i+1}: {' '.join(sl)}" for i, sl in enumerate(summary_list)])

    def count_tokens(text):
        return len(tokenizer.encode(text))

    level = 0
    # Do top level
    abs_summ, selected_chunks = summarize_batch(
        chunks, [chunk_texts(chunk, tokenizer, HIER_RET_UNIT) for chunk in chunks], max_len=max_len
    )
    summaries[level] = selected_chunks
    abs_summaries[level] = abs_summ

    # Do all merge context levels
    while len(summaries[level]) > 1:
        next_level = level + 1
        summaries[next_level] = []
        abs_summaries[next_level] = []
        context = ""
        merged_abs = []
        merged_chunks = []
        final = True

        for abs_summ, selected_chunk in zip(abs_summaries[level], summaries[level]):
            merged_abs.append(abs_summ)
            merged_chunks.append(selected_chunk)
            merged_text = merge_summaries(merged_abs)
            merged_context = merge_chunks(merged_chunks)
            if context == "":
                prompt = dataset_config["merge_aggr_prompt"].format(merged_text, merged_context)
            else:
                prompt = dataset_config["merge_context_aggr_prompt"].format(context, merged_text, merged_context)
            
            if count_tokens(prompt) > max_context_len and (len(merged_abs) + bool(context)) >= 3: # Lower bound for merging
                final = False
                merged_abs.pop()
                merged_chunks.pop()
                merged_text = merge_summaries(merged_abs)
                merged_context = merge_chunks(merged_chunks)
                new_summary, new_selected_chunks = summarize(
                    merged_text, merged_chunks, merged_context, merge=True, context=context, max_len=max_len
                )
                summaries[next_level].append(new_selected_chunks)
                abs_summaries[next_level].append(new_summary)
                context = new_summary
                merged_abs = [abs_summ]
                merged_chunks = [selected_chunk]

        if len(merged_abs) >= 1:
            merged_text = merge_summaries(merged_abs)
            merged_context = merge_chunks(merged_chunks)
            new_summary, new_selected_chunks = summarize(
                merged_text, merged_chunks, merged_context, merge=True, final=final, context=context, max_len=max_len
            )
            summaries[next_level].append(new_selected_chunks)
            abs_summaries[next_level].append(new_summary)
            context = new_summary

        level = next_level

    return abs_summaries

for i in tqdm(range(len(dataset))):
    input_text = parse_non_ascii(dataset[i]['input'])
    output_text = parse_non_ascii(dataset[i]['output'])
    chunks = chunk_texts(input_text, tokenizer, dataset_config["chunk_size"])
    final_summary_dict = recursive_summarize({}, {}, chunks, dataset_config["max_context_length"] * 2, dataset_config["max_summary_length"] * WORD_TOKEN_RATIO)
    final_summaries[i] = final_summary_dict[max(final_summary_dict.keys())][0]

if args.output_file == "":
    json.dump(final_summaries, open(dataset_config["output_file"], "w"), indent=4)
else:
    json.dump(final_summaries, open(args.output_file, "w"), indent=4)
