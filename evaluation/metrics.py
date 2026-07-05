"""Ranking metrics for evaluating recommendation quality against real interaction labels."""

import math
from typing import Iterable, Sequence, Set


def precision_at_k(ranked_ids: Sequence[int], positives: Set[int], k: int) -> float:
    if not ranked_ids or k <= 0:
        return 0.0
    topk = ranked_ids[:k]
    hits = sum(1 for pid in topk if pid in positives)
    return hits / min(k, len(ranked_ids))


def recall_at_k(ranked_ids: Sequence[int], positives: Set[int], k: int) -> float:
    if not positives:
        return 0.0
    topk = set(ranked_ids[:k])
    hits = sum(1 for pid in positives if pid in topk)
    return hits / len(positives)


def hit_rate_at_k(ranked_ids: Sequence[int], positives: Set[int], k: int) -> float:
    topk = set(ranked_ids[:k])
    return 1.0 if topk & positives else 0.0


def ndcg_at_k(ranked_ids: Sequence[int], positives: Set[int], k: int) -> float:
    topk = ranked_ids[:k]
    dcg = sum(1.0 / math.log2(idx + 2) for idx, pid in enumerate(topk) if pid in positives)
    ideal_hits = min(len(positives), k)
    idcg = sum(1.0 / math.log2(idx + 2) for idx in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def pairwise_auc(pos_scores: Iterable[float], neg_scores: Iterable[float]) -> float:
    """
    Mann-Whitney U based AUC: the probability that a random positive scores
    higher than a random negative. 0.5 = model can't tell them apart (no
    better than chance); 1.0 = perfect separation.
    """
    pos_scores = list(pos_scores)
    neg_scores = list(neg_scores)
    if not pos_scores or not neg_scores:
        return float("nan")

    labeled = [(s, 1) for s in pos_scores] + [(s, 0) for s in neg_scores]
    labeled.sort(key=lambda x: x[0])

    n = len(labeled)
    rank_sum = 0.0
    i = 0
    rank = 1
    while i < n:
        j = i
        while j < n and labeled[j][0] == labeled[i][0]:
            j += 1
        tie_count = j - i
        avg_rank = rank + (tie_count - 1) / 2.0
        for idx in range(i, j):
            if labeled[idx][1] == 1:
                rank_sum += avg_rank
        rank += tie_count
        i = j

    n_pos, n_neg = len(pos_scores), len(neg_scores)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
