from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from argparse import ArgumentParser
from utils import *
from tqdm import tqdm
from configs.zs_config import HIER_DATASET_CONFIG, WORD_TOKEN_RATIO
from memsum.summarizer import MemSum
from nltk.tokenize import sent_tokenize
from vllm import LLM, SamplingParams
import torch
import json

from transformers.utils import logging
logging.set_verbosity_error()

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, default="sfd")
parser.add_argument("--model_ckpt", type=str, default="memsum_model.ckpt")
parser.add_argument("--vocab_path", type=str, default="memsum/vocabulary_200dim.pkl")
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
llm = LLM(model=model_path, max_model_len=10000, gpu_memory_utilization=0.6, tensor_parallel_size=args.tensor_parallel_size) # Hard restriction to save space
ext_model = MemSum(args.model_ckpt, args.vocab_path, gpu=0, max_doc_len=dataset_config["max_context_length"])
final_summaries = {}

MAX_SUMMARY_LENGTH = dataset_config["max_summary_length"]
MAX_CONTEXT_LENGTH = dataset_config["max_context_length"]
MAX_EXT_SENTENCES = dataset_config["max_ext_sentences"]
THRESHOLD_LENGTH = 2000

def summarize(text, chunk_context, merge=True, final=False, context="", max_len=MAX_SUMMARY_LENGTH):
    if not merge:
        input_text = dataset_config["chunk_prompt"].format(text)
    else:
        if context == "":
            input_text = dataset_config["merge_aggr_prompt"].format(text, chunk_context)
        else:
            input_text = dataset_config["merge_context_aggr_prompt"].format(context, text, chunk_context)
    response = vllm_generate(llm, sampling_params, input_text)
    return response

def summarize_batch(texts, max_len=MAX_SUMMARY_LENGTH * WORD_TOKEN_RATIO):
    input_texts = [dataset_config["chunk_prompt"].format(texts[i]) for i in range(len(texts))]
    responses = vllm_generate(llm, sampling_params, input_texts, mode="batch")
    return responses

def summarize_ext(sents):
    max_ext_sentences = min(MAX_EXT_SENTENCES, len(sents))
    ext_sents = ext_model.extract(
        [sents],
        p_stop_thres=0.6,
        max_extracted_sentences_per_document=max_ext_sentences
    )[0]
    
    truncated_sents = []
    current_length = 0
    for sent in ext_sents:
        sent_tokens = len(tokenizer.encode(sent))
        if current_length + sent_tokens <= THRESHOLD_LENGTH:
            truncated_sents.append(sent)
            current_length += sent_tokens
        else:
            break
            
    return truncated_sents

def recursive_summarize(abs_summaries, summaries, sents_chunks, chunks):
    def merge_summaries(summary_list=None, abs_summary_list=None):
        if summary_list is not None:
            return "\n".join([f"Summary {i+1}: {' '.join(summary_list[i])}" for i in range(len(summary_list))])
        else:
            return "\n".join([f"Summary {i+1}: {abs_summary_list[i]}" for i in range(len(abs_summary_list))])
    
    def count_tokens(text):
        return len(tokenizer.encode(text))

    level = 0
    # Perform extractive summarization at the first level
    abs_summaries[level] = summarize_batch(chunks)
    summaries[level] = [summarize_ext(sents_chunks[i]) for i in range(len(sents_chunks))]

    while len(summaries[level]) > 1:
        next_level = level + 1
        summaries[next_level] = []
        abs_summaries[next_level] = []
        context = ""
        merged_summaries = []
        merged_abs_summaries = []
        final = True

        for summary, abs_summary in zip(summaries[level], abs_summaries[level]):
            merged_summaries.append(summary)
            merged_abs_summaries.append(abs_summary)
            flat_chunk_list = [item for sublist in merged_summaries for item in sublist]
            merged_text = merge_summaries(summary_list=merged_summaries)
            merged_abs_text = merge_summaries(abs_summary_list=merged_abs_summaries)
            
            if context == "":
                prompt = dataset_config["merge_aggr_prompt"].format(merged_text, merged_abs_text)
            else:
                prompt = dataset_config["merge_context_aggr_prompt"].format(context, merged_text, merged_abs_text)
            
            if count_tokens(prompt) > MAX_CONTEXT_LENGTH:
                # Summarize the current batch
                final = False
                merged_summaries.pop()  # Remove the last summary that caused overflow
                merged_abs_summaries.pop()
                merged_text = merge_summaries(summary_list=merged_summaries)
                merged_abs_text = merge_summaries(abs_summary_list=merged_abs_summaries)
                flat_chunk_list = [item for sublist in merged_summaries for item in sublist]
                chunks_to_summarize = list(set(flat_chunk_list))
                new_summary = summarize_ext(chunks_to_summarize)
                new_abs_summary = summarize(merged_abs_text, merged_text, merge=True, final=False, context=context)
                summaries[next_level].append(new_summary)
                abs_summaries[next_level].append(new_abs_summary)
                context = new_abs_summary
                merged_summaries = [summary]  # Start a new merge with the overflow summary
                merged_abs_summaries = [abs_summary]
        
        # Handle the remaining summaries after the loop
        if len(merged_summaries) >= 1:
            merged_text = merge_summaries(summary_list=merged_summaries)
            merged_abs_text = merge_summaries(abs_summary_list=merged_abs_summaries)
            chunks_to_summarize = list(set(flat_chunk_list))
            new_summary = summarize_ext(chunks_to_summarize)
            new_abs_summary = summarize(merged_abs_text, merged_text, merge=True, final=final, context=context)
            summaries[next_level].append(new_summary)
            abs_summaries[next_level].append(new_abs_summary)
            context = new_abs_summary
        
        level = next_level
    
    return abs_summaries

for i in tqdm(range(len(dataset))):
    input_text = parse_non_ascii(dataset[i]['input'])
    output_text = parse_non_ascii(dataset[i]['output'])
    chunks = chunk_texts(input_text, tokenizer, dataset_config["chunk_size"])
    sents_chunks = [sent_tokenize(chunk) for chunk in chunks]
    final_summary_dict = recursive_summarize({}, {}, sents_chunks, chunks)
    final_summaries[i] = final_summary_dict[max(final_summary_dict.keys())][0]

if args.output_file == "":
    json.dump(final_summaries, open(dataset_config["output_file"], "w"), indent=4)
else:
    json.dump(final_summaries, open(args.output_file, "w"), indent=4)
