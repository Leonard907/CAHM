# CAHM
This repo contains the code for the paper [Context-aware hierarchical merging for long document summarization](https://arxiv.org/abs/2502.00977)

## Requirements
```sh
bash install.sh
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
@misc{ou2025contextawarehierarchicalmerginglong,
      title={Context-Aware Hierarchical Merging for Long Document Summarization}, 
      author={Litu Ou and Mirella Lapata},
      year={2025},
      eprint={2502.00977},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.00977}, 
}
```
