# CAHM
This repo contains the code for the paper [Context-aware hierarchical merging for long document summarization](https://arxiv.org/abs/2502.00977)

## Requirements
```sh
bash install.sh
wget https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt
```

## Baselines
```sh
python summ_zero_shot.py --dataset multi_lexsum --model_path <path_to_hf_model> # Zero-Shot
python hier_summ.py --dataset multi_lexsum --model_path <path_to_hf_model> # HMerge
python hier_summ_attr.py --dataset multi_lexsum --model_path <path_to_hf_model> --mode abs # Cite-HMerge
```

## Summarization
```sh
python hier_summ_attr.py --dataset multi_lexsum --model_path <path_to_hf_model> --mode ret # Cite-R
python hier_summ_attr.py --dataset multi_lexsum --model_path <path_to_hf_model> --mode mix # Cite-S
python hier_summ_ext.py --dataset multi_lexsum --model_path <path_to_hf_model> # Extract-R
python hier_summ_extmix.py --dataset multi_lexsum --model_path <path_to_hf_model> # Extract-S
python hier_summ_ret.py --dataset multi_lexsum --model_path <path_to_hf_model> # Retrieve-R
python hier_summ_retmix.py --dataset multi_lexsum --model_path <path_to_hf_model> # Retrieve-S
```
For the extractive models, please see the paper and [this repo](https://github.com/nianlonggu/MemSum) for more details.

## Evaluation
```sh
python eval.py --dataset multi_lexsum --metric [rouge|bertscore|summac|alignscore] --filename <path_to_output_file>
```
For PRISMA, please see [this repo](https://github.com/Lou1sM/modular_multimodal_summarization)

## Datasets
HF datasets are given in `configs/zs_config.py`. We have made the supersummary dataset private due to copyright considerations.

## Citation
```
@inproceedings{ou-lapata-2025-context,
    title = "Context-Aware Hierarchical Merging for Long Document Summarization",
    author = "Ou, Litu  and
      Lapata, Mirella",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.289/",
    pages = "5534--5561",
    ISBN = "979-8-89176-256-5",
    abstract = "Hierarchical Merging is a technique commonly used to summarize very long texts ({\ensuremath{>}}100K tokens) by breaking down the input into smaller sections, summarizing those sections individually, and then merging or combining those summaries into a final coherent summary. Although it helps address the limitations of large language models (LLMs) with fixed input length constraints, the recursive merging process can amplify LLM hallucinations, increasing the risk of factual inaccuracies. In this paper, we seek to mitigate hallucinations by enriching hierarchical merging with context from the source document. Specifically, we propose different approaches to contextual augmentation ranging from *replacing* intermediate summaries with relevant input context, to *refining* them while using the context as supporting evidence, and *aligning* them implicitly (via citations) to the input. Experimental results on datasets representing legal and narrative domains show that contextual augmentation consistently outperforms zero-shot and hierarchical merging baselines for the Llama 3.1 model family. Our analysis further reveals that refinement methods tend to perform best when paired with extractive summarization for identifying relevant input."
}
```
