from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from datasets import load_dataset
from argparse import ArgumentParser
from utils import *
from tqdm import tqdm
from configs.zs_config import *
import torch
import json

from transformers.utils import logging
logging.set_verbosity_error()

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, default="sfd")
parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
parser.add_argument("--output_file", type=str, default=None)
parser.add_argument("--tensor_parallel_size", type=int, default=1)
args = parser.parse_args()
dataset_name = args.dataset
dataset_config = ZEROSHOT_DATASET_CONFIG[dataset_name]

model_path = args.model_path
tokenizer = AutoTokenizer.from_pretrained(model_path)
dataset = load_dataset(dataset_config["dataset"])['test']
sampling_params = SamplingParams(temperature=0, top_p=0.9, max_tokens=dataset_config["max_new_tokens"])
llm = LLM(model=model_path, tensor_parallel_size=args.tensor_parallel_size)
summaries = {}

for i in tqdm(range(len(dataset))):
    input_text = parse_non_ascii(dataset[i]['input'])
    output_text = parse_non_ascii(dataset[i]['output'])

    truncated_text = truncate_text(tokenizer, input_text, 128000)
    input_prompt = dataset_config["zs_prompt"].format(truncated_text)
    response = vllm_generate(llm, sampling_params, input_prompt)
    summaries[i] = response

if args.output_file is not None:
    json.dump(summaries, open(args.output_file, "w"), indent=4)
else:
    json.dump(summaries, open(dataset_config["output_file"], "w"), indent=4)