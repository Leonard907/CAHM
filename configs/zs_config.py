ZEROSHOT_DATASET_CONFIG = {
    "multi_lexsum": {
        "dataset": "shipWr3ck/mls_cleaned_v2",
        "max_new_tokens": 2048,
        "max_summary_length": 1000,
        "output_file": "outputs_zeroshot/multi_lexsum_zs_summaries_llama3_8b.json",
        "qa_folder": "qa_files/multi_lexsum_model",
        "zs_prompt": open("prompts/init/init_hier.txt").read(),
        "zs_plan_prompt": "You are an abstractive summarizer that writes a summary for a document consisting of multiple key points. The summary should contain 10-20 sentences and about 150 words.\n\nDocument: {}\nSummary:"
    },
    "supersummary": {
        "dataset": "shipWr3ck/supersummary",
        "max_new_tokens": 2048,
        "max_summary_length": 1000,
        "output_file": "outputs_zeroshot/supersummary_zs_summaries_llama3_8b.json",
        "qa_folder": "qa_files/supersummary_model",
        "zs_prompt": open("prompts/init/init_hier.txt").read(),
        "zs_plan_prompt": "You are an abstractive summarizer that writes a summary for a document consisting of multiple key points. The summary should contain 10-20 sentences and about 150 words.\n\nDocument: {}\nSummary:"
    }
}

HIER_COMMON_CONFIG = {
    "max_context_length": 8000,
    "max_context_mix_length": 8000,
    "chunk_size": 8000,
    "attr_chunk_size": 100,
    "chunk_prompt": open("prompts/init/init_hier.txt").read(),
    "chunk_attr_prompt": open("prompts/init/init_hier_attr.txt").read(),
    "merge_prompt": open("prompts/merge/merge_hier.txt").read(),
    "merge_attr_prompt": open("prompts/merge/merge_hier_attr.txt").read(),
    "merge_context_prompt": open("prompts/merge_context/merge_context_hier.txt").read(),
    "merge_context_attr_prompt": open("prompts/merge_context/merge_context_hier_attr.txt").read(),
    "merge_aggr_prompt": open("prompts/merge_aggr/merge_aggr.txt").read(),
    "merge_context_aggr_prompt": open("prompts/merge_context/merge_context_aggr.txt").read(),
    "merge_aggr_attr_prompt": open("prompts/merge_aggr/merge_aggr_attr.txt").read(),
    "merge_context_aggr_attr_prompt": open("prompts/merge_context/merge_context_aggr_attr.txt").read(),
}

HIER_DATASET_CONFIG = {
    "multi_lexsum": {
        "dataset": "shipWr3ck/mls_cleaned_v2",
        "output_file": "outputs_hier/multi_lexsum_hier_summaries_llama3.1_8b.json",
        "max_new_tokens": 2048,
        "max_summary_length": 1000,
        "max_ext_sentences": 20,
        **HIER_COMMON_CONFIG
    },
    "supersummary": {
        "dataset": "shipWr3ck/supersummary",
        "output_file": "outputs_hier/supersummary_hier_summaries_llama3_8b.json",
        "max_new_tokens": 2048,
        "max_summary_length": 1000,
        "max_ext_sentences": 20,
        **HIER_COMMON_CONFIG
    }
}

HIER_RET_UNIT = 100
WORD_TOKEN_RATIO = 0.65