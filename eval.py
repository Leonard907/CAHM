from evaluate import load
from datasets import load_dataset
from configs.zs_config import ZEROSHOT_DATASET_CONFIG
from argparse import ArgumentParser
from tqdm import tqdm
from utils import *
from transformers import AutoTokenizer
import json
import torch

try:
    from summac.model_summac import SummaCConv
except ImportError:
    print("summac not installed")

try:
    from alignscore import AlignScore
except ImportError:
    print("alignscore not installed")

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, default="sfd")
parser.add_argument("--metric", type=str, default="rouge")
parser.add_argument("--filename", type=str, default="output.json")
parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")

MAX_LENGTH_MAP = {
    "meta-llama/Llama-3.1-8B-Instruct": 128000,
}

def eval_bertscore(dataset, summaries):
    bertscore = load('bertscore')
    total_score = 0
    predictions = []
    references = []
    for i in tqdm(range(len(dataset))):
        output_text = parse_non_ascii(dataset[i]['output'])
        summary = summaries[str(i)]
        predictions.append(summary)
        references.append(output_text)
    scores = bertscore.compute(predictions=predictions, references=references, model_type="microsoft/deberta-xlarge-mnli", batch_size=4)
    return sum(scores["precision"]) / len(scores["precision"])

def eval_rouge(dataset, summaries):
    rouge = load('rouge')
    predictions = []
    references = []
    for i in range(len(dataset)):
        output_text = parse_non_ascii(dataset[i]['output'])
        predictions.append(summaries[str(i)])
        references.append(output_text)
    scores = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    
    # Compute geometric mean of ROUGE-1, ROUGE-2, and ROUGE-L F1 scores
    print(scores)
    rouge1 = scores['rouge1']
    rouge2 = scores['rouge2']
    rougeL = scores['rougeL']
    geometric_mean = (rouge1 * rouge2 * rougeL) ** (1/3)
    
    return geometric_mean, scores

def eval_summac(dataset, summaries, max_len, tokenizer, no_ref_truncate):
    model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cuda", start_file="default", agg="mean")
    total_score = 0
    for i in tqdm(range(len(dataset))):
        input_text = parse_non_ascii(dataset[i]['input'])
        if not no_ref_truncate:
            input_text = truncate_text(tokenizer, input_text, max_len)
        summary = summaries[str(i)]
        total_score += model_conv.score([input_text], [summary])['scores'][0]
    return total_score / len(dataset)

def eval_alignscore(dataset, summaries, max_len, tokenizer, no_ref_truncate):
    scorer = AlignScore(model='roberta-large', batch_size=32, device='cuda:0', ckpt_path='../AlignScore-large.ckpt', evaluation_mode='nli_sp', verbose=False)
    total_score = 0
    for i in tqdm(range(len(dataset))):
        input_text = parse_non_ascii(dataset[i]['input'])
        if not no_ref_truncate:
            input_text = truncate_text(tokenizer, input_text, max_len)
        summary = summaries[str(i)]
        total_score += scorer.score(contexts=[input_text], claims=[summary])[0]
    return total_score / len(dataset)

if __name__ == "__main__":
    args = parser.parse_args()
    file_name = args.filename
    dataset = load_dataset(ZEROSHOT_DATASET_CONFIG[args.dataset]["dataset"])['test']
    summaries = json.load(open(file_name))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    ref_truncate = False if "hier" in args.filename else True

    with torch.no_grad():
        if args.metric == "rouge":
            scores = eval_rouge(dataset, summaries)
        elif args.metric == "summac":
            scores = eval_summac(dataset, summaries, MAX_LENGTH_MAP[args.model_path], tokenizer, ref_truncate)
        elif args.metric == "alignscore":
            scores = eval_alignscore(dataset, summaries, MAX_LENGTH_MAP[args.model_path], tokenizer, ref_truncate)
        elif args.metric == "bertscore":
            scores = eval_bertscore(dataset, summaries)

    if args.metric == "rouge":
        geometric_mean, scores = scores
        print('Metric:', args.metric)
        print(scores)
        print('Geometric Mean:', geometric_mean)
    else:
        print('Metric:', args.metric)
        print(scores)
