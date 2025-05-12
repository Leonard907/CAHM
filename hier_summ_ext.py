from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from argparse import ArgumentParser
from utils import *
from tqdm import tqdm
from configs.zs_config import HIER_DATASET_CONFIG
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
llm = LLM(model=model_path, max_model_len=15000, gpu_memory_utilization=0.6, tensor_parallel_size=args.tensor_parallel_size) # Hard restriction to save space
ext_model = MemSum(args.model_ckpt, args.vocab_path, gpu=0, max_doc_len=dataset_config["max_context_length"])
final_summaries = {}

MAX_SUMMARY_LENGTH = dataset_config["max_summary_length"]
MAX_CONTEXT_LENGTH = dataset_config["max_context_length"]
MAX_EXT_SENTENCES = dataset_config["max_ext_sentences"]

def summarize_ext(sents):
    ext_sents = ext_model.extract(
        [sents],
        p_stop_thres=0.6,
        max_extracted_sentences_per_document=MAX_EXT_SENTENCES
    )[0]
    return ext_sents

def recursive_summarize(summaries, chunks):
    def merge_summaries(summary_list):
        return "\n".join([f"Summary {i+1}: {' '.join(summary_list[i])}" for i in range(len(summary_list))])
    
    def count_tokens(text):
        return len(tokenizer.encode(text))

    level = 0
    # Perform extractive summarization at the first level
    summaries[level] = [summarize_ext(chunks[i]) for i in range(len(chunks))]

    while len(summaries[level]) > 1:
        next_level = level + 1
        summaries[next_level] = []
        context = ""
        merged_summaries = []
        final = True

        for summary in summaries[level]:
            merged_summaries.append(summary)
            flat_chunk_list = [item for sublist in merged_summaries for item in sublist]
            merged_text = merge_summaries(merged_summaries)
            
            if context == "":
                prompt = dataset_config["merge_prompt"].format(merged_text)
            else:
                prompt = dataset_config["merge_context_prompt"].format(context, merged_text)
            
            if count_tokens(prompt) > MAX_CONTEXT_LENGTH and (len(merged_summaries) + bool(context)) > 3:
                # Summarize the current batch
                final = False
                merged_summaries.pop()  # Remove the last summary that caused overflow
                merged_text = merge_summaries(merged_summaries)
                chunks_to_summarize = list(set(flat_chunk_list))
                new_summary = summarize_ext(chunks_to_summarize)
                summaries[next_level].append(new_summary)
                context = new_summary
                merged_summaries = [summary]  # Start a new merge with the overflow summary
        
        # Handle the remaining summaries after the loop
        if len(merged_summaries) >= 1:
            merged_text = merge_summaries(merged_summaries)
            chunks_to_summarize = list(set(flat_chunk_list))
            if final:
                prompt = dataset_config["merge_prompt"].format(merged_text)
                new_summary = vllm_generate(llm, sampling_params, prompt)
            else:
                new_summary = summarize_ext(chunks_to_summarize)
            summaries[next_level].append(new_summary)
            context = new_summary
        
        level = next_level
    
    return summaries

for i in tqdm(range(len(dataset))):
    input_text = parse_non_ascii(dataset[i]['input'])
    output_text = parse_non_ascii(dataset[i]['output'])
    chunks = chunk_texts(input_text, tokenizer, dataset_config["chunk_size"])
    sents_chunks = [sent_tokenize(chunk) for chunk in chunks]
    final_summary_dict = recursive_summarize({}, sents_chunks)
    final_summaries[i] = final_summary_dict[max(final_summary_dict.keys())][0]

if args.output_file == "":
    json.dump(final_summaries, open(dataset_config["output_file"], "w"), indent=4)
else:
    json.dump(final_summaries, open(args.output_file, "w"), indent=4)
