from rank_bm25 import BM25Okapi
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np

# Intend use: select topk passages given whole summary
def summ_select(query, docs, topk=5):
    bm25 = BM25Okapi([word_tokenize(doc) for doc in docs])
    doc_scores = bm25.get_scores(word_tokenize(query))
    topk_idx = np.argsort(doc_scores)[::-1][:topk]
    return [docs[i] for i in sorted(topk_idx)]

# Intend use: select topk passages based on scores against each summary sentence
def sent_select(query, docs, topk=5):
    bm25 = BM25Okapi([word_tokenize(doc) for doc in docs])
    sents = sent_tokenize(query)
    sent_scores = np.array([bm25.get_scores(word_tokenize(sent)) for sent in sents])
    topk_idx = np.argmax(sent_scores, axis=1)
    unique_idx, _ = np.unique(topk_idx, return_counts=True)
    
    # If unique indices are exactly equal to topk, return these passages
    if len(unique_idx) == topk:
        return [docs[i] for i in sorted(unique_idx)]
    
    passage_scores = np.sum(sent_scores, axis=0)

    if len(unique_idx) > topk:
        sorted_indices = np.argsort(passage_scores[unique_idx])[::-1]
        topk_passages_idx = unique_idx[sorted_indices[:topk]]
        return [docs[i] for i in sorted(topk_passages_idx)]
    if len(unique_idx) < topk:
        remaining_topk = topk - len(unique_idx)
        # Exclude already selected unique_idx from consideration
        remaining_scores = np.copy(passage_scores)
        remaining_scores[unique_idx] = -np.inf
        additional_indices = np.argsort(remaining_scores)[::-1][:remaining_topk]
        topk_passages_idx = np.concatenate((unique_idx, additional_indices))
        return [docs[i] for i in sorted(topk_passages_idx)]
    