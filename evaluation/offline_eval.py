#!/usr/bin/env python3
"""
Offline accuracy evaluation for the Two-Tower recommendation model against
real historical interaction data.

For each user: pull their real interaction history (likes/saves/not-interested,
with timestamps) from the Spring API, fetch the Two-Tower vectors for those
specific posts plus a pool of real unseen candidates as distractors, score
everything the same way core_recommendations_service.py does (TwoTower.compute_scores
on raw vectors), rank, and measure how well the ranking recovers the posts the
user actually liked/saved vs. ones they ignored or explicitly rejected.

This measures the RAW Two-Tower model only (no metadata enhancer / RL / content
filter / diversity reranking on top) - i.e. "is the core embedding model any good
at telling what this user likes from what they don't". Use health_check.py to
check the rest of the pipeline.

Usage:
    python -m evaluation.offline_eval --user-ids 12,34,56 --k 5 10 20
    python -m evaluation.offline_eval --user-ids-file eval_users.txt
"""

import argparse
import json
import logging
import os
import sys
from statistics import mean
from typing import Dict, List, Sequence

import numpy as np
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "shared", "components"))
from TwoTower import compute_scores  # noqa: E402

from evaluation.spring_client import SpringMLClient  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    precision_at_k, recall_at_k, ndcg_at_k, hit_rate_at_k, pairwise_auc,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger("offline-eval")

DEGENERATE_SCORE_STD_THRESHOLD = 0.02


def _field(obj: Dict, *names, default=None):
    for name in names:
        if name in obj:
            return obj[name]
    return default


def _extract_labels(history: Dict) -> (set, set):
    """Returns (positive_post_ids, negative_post_ids) from an
    MLInteractionHistoryResponse-shaped dict."""
    positives, negatives = set(), set()

    for i in history.get("interactions", []):
        pid = _field(i, "postId", "post_id")
        if pid is None:
            continue
        liked = bool(_field(i, "likeState", "like_state", default=False))
        saved = bool(_field(i, "saveState", "save_state", default=False))
        if liked or saved:
            positives.add(int(pid))
        else:
            negatives.add(int(pid))

    for n in history.get("notInterested", []):
        pid = _field(n, "postId", "post_id")
        if pid is not None:
            negatives.add(int(pid))

    negatives -= positives
    return positives, negatives


def evaluate_user(client: SpringMLClient, user_id, k_values: Sequence[int] = (5, 10, 20),
                   candidate_pool_size: int = 200, min_positives: int = 2,
                   history_limit: int = 500, content_type: str = "POSTS") -> Dict:
    history = client.get_interaction_history(user_id, limit=history_limit)
    positives, negatives = _extract_labels(history)

    if len(positives) < min_positives:
        return {"user_id": user_id, "skipped": True,
                "reason": f"fewer than {min_positives} liked/saved posts in history ({len(positives)})"}

    user_vector = client.get_user_vector(user_id)
    if user_vector is None:
        return {"user_id": user_id, "skipped": True, "reason": "no user vector available"}

    labeled_ids = list(positives | negatives)
    # Counterfactual (leakage-free): excludes this user's own interactions from
    # each labeled post's behavioral genre profile, since a post's stored profile
    # can otherwise partly reflect this same user's past like/save of it.
    labeled_vectors = client.get_post_vectors_counterfactual(labeled_ids, exclude_user_id=user_id)
    candidate_vectors = client.get_candidates(user_id, content_type=content_type, limit=candidate_pool_size)

    pool: Dict[int, np.ndarray] = dict(candidate_vectors)
    pool.update(labeled_vectors)

    usable_positives = positives & pool.keys()
    usable_negatives = negatives & pool.keys()

    if len(usable_positives) < min_positives:
        missing = len(positives) - len(usable_positives)
        return {"user_id": user_id, "skipped": True,
                "reason": f"post vectors unavailable for {missing}/{len(positives)} liked/saved posts"}

    pool_ids = list(pool.keys())
    pool_matrix = np.stack([pool[pid] for pid in pool_ids])
    scores = compute_scores(user_vector.reshape(1, -1), pool_matrix, content_type.lower())[0]

    ranked = sorted(zip(pool_ids, scores), key=lambda x: x[1], reverse=True)
    ranked_ids = [pid for pid, _ in ranked]

    result = {
        "user_id": user_id,
        "skipped": False,
        "num_positives": len(usable_positives),
        "num_negatives": len(usable_negatives),
        "pool_size": len(pool_ids),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
    }
    for k in k_values:
        result[f"precision@{k}"] = precision_at_k(ranked_ids, usable_positives, k)
        result[f"recall@{k}"] = recall_at_k(ranked_ids, usable_positives, k)
        result[f"ndcg@{k}"] = ndcg_at_k(ranked_ids, usable_positives, k)
        result[f"hit_rate@{k}"] = hit_rate_at_k(ranked_ids, usable_positives, k)

    if usable_negatives:
        pos_scores = [s for pid, s in zip(pool_ids, scores) if pid in usable_positives]
        neg_scores = [s for pid, s in zip(pool_ids, scores) if pid in usable_negatives]
        result["pairwise_auc"] = pairwise_auc(pos_scores, neg_scores)

    return result


def run_offline_eval(user_ids: List, client: SpringMLClient, k_values: Sequence[int] = (5, 10, 20),
                      **kwargs) -> List[Dict]:
    results = []
    for uid in user_ids:
        try:
            r = evaluate_user(client, uid, k_values=k_values, **kwargs)
        except Exception as e:
            logger.exception(f"evaluate_user({uid}) raised")
            r = {"user_id": uid, "skipped": True, "reason": f"error: {e}"}
        results.append(r)
        if r.get("skipped"):
            logger.info(f"user {uid}: skipped ({r['reason']})")
        else:
            logger.info(f"user {uid}: {r['num_positives']} positives, {r['num_negatives']} negatives, "
                        f"pool={r['pool_size']}, precision@{k_values[0]}={r[f'precision@{k_values[0]}']:.3f}")
    return results


def summarize(results: List[Dict], k_values: Sequence[int]) -> Dict:
    evaluated = [r for r in results if not r.get("skipped")]
    skipped = [r for r in results if r.get("skipped")]
    summary = {
        "num_users_requested": len(results),
        "num_evaluated": len(evaluated),
        "num_skipped": len(skipped),
        "skip_reasons": [r["reason"] for r in skipped],
    }
    if evaluated:
        for k in k_values:
            summary[f"mean_precision@{k}"] = mean(r[f"precision@{k}"] for r in evaluated)
            summary[f"mean_recall@{k}"] = mean(r[f"recall@{k}"] for r in evaluated)
            summary[f"mean_ndcg@{k}"] = mean(r[f"ndcg@{k}"] for r in evaluated)
            summary[f"hit_rate@{k}"] = mean(r[f"hit_rate@{k}"] for r in evaluated)
        auc_vals = [r["pairwise_auc"] for r in evaluated if "pairwise_auc" in r and not np.isnan(r["pairwise_auc"])]
        if auc_vals:
            summary["mean_pairwise_auc"] = mean(auc_vals)
        summary["mean_score_std"] = mean(r["score_std"] for r in evaluated)
        if summary["mean_score_std"] < DEGENERATE_SCORE_STD_THRESHOLD:
            summary["warning"] = (
                f"Mean score std ({summary['mean_score_std']:.4f}) is near zero - the model is producing "
                "almost identical scores for every candidate. This usually means it's untrained/randomly "
                "initialized rather than that it has converged to a confident ranking."
            )
    return summary


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--user-ids", type=str, help="Comma-separated list of real user IDs to evaluate")
    p.add_argument("--user-ids-file", type=str, help="File with one user ID per line")
    p.add_argument("--k", type=int, nargs="+", default=[5, 10, 20], help="k values for precision/recall/NDCG/hit-rate")
    p.add_argument("--candidate-pool-size", type=int, default=200)
    p.add_argument("--min-positives", type=int, default=2, help="Skip users with fewer liked/saved posts than this")
    p.add_argument("--history-limit", type=int, default=500)
    p.add_argument("--content-type", type=str, default="POSTS", choices=["POSTS", "TRAILERS"])
    p.add_argument("--api-url", type=str, default=None, help="Override SPRING_API_URL")
    p.add_argument("--json-out", type=str, default=None, help="Write full per-user results to this JSON file")
    return p.parse_args()


def main():
    load_dotenv()
    args = _parse_args()

    user_ids: List[str] = []
    if args.user_ids:
        user_ids.extend(u.strip() for u in args.user_ids.split(",") if u.strip())
    if args.user_ids_file:
        with open(args.user_ids_file) as f:
            user_ids.extend(line.strip() for line in f if line.strip())
    if not user_ids:
        print("No user IDs provided. Use --user-ids or --user-ids-file.", file=sys.stderr)
        sys.exit(1)

    client = SpringMLClient(base_url=args.api_url)
    ok, detail = client.check_connectivity()
    if not ok:
        print(f"WARNING: Spring API connectivity check failed ({detail}) - continuing anyway", file=sys.stderr)

    results = run_offline_eval(
        user_ids, client, k_values=args.k,
        candidate_pool_size=args.candidate_pool_size,
        min_positives=args.min_positives,
        history_limit=args.history_limit,
        content_type=args.content_type,
    )
    summary = summarize(results, args.k)

    print("\n=== OFFLINE EVALUATION SUMMARY ===")
    print(json.dumps(summary, indent=2, default=float))

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"summary": summary, "per_user": results}, f, indent=2, default=float)
        print(f"\nFull per-user results written to {args.json_out}")


if __name__ == "__main__":
    main()
