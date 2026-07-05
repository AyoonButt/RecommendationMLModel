#!/usr/bin/env python3
"""
Component health-check / smoke test for the recommendation pipeline, run against
real data (real users, real candidates, real Spring API). This is NOT an accuracy
metric - it's a "is this pipeline actually doing something sane" diagnostic that
exercises each stage with real inputs and flags degenerate/broken behavior.

Usage:
    python -m evaluation.health_check --user-ids 12,34,56
    python -m evaluation.health_check --user-ids 12,34,56 --live-url http://localhost:5000
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import requests
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "shared", "components"))
from TwoTower import compute_scores  # noqa: E402
from MockRedis import MockRedis  # noqa: E402
from MetadataEnhancer import MetadataEnhancer  # noqa: E402
from ContentFilter import ContentFilter  # noqa: E402

from evaluation.spring_client import SpringMLClient  # noqa: E402

DEGENERATE_SCORE_STD_THRESHOLD = 0.02

# Model checkpoint files as loaded by services/core-recommendations/core_recommendations_service.py
SERVICE_DIR = os.path.join(os.path.dirname(__file__), "..", "services", "core-recommendations")


@dataclass
class CheckResult:
    name: str
    status: str  # "PASS" | "WARN" | "FAIL"
    detail: str
    data: Dict[str, Any] = field(default_factory=dict)


def check_model_checkpoint() -> CheckResult:
    model_dir = os.environ.get("MODEL_DIR", "./model_checkpoints")
    user_model = os.environ.get("USER_MODEL", "user_tower_latest.h5")
    post_model = os.environ.get("POST_MODEL", "post_tower_latest.h5")

    candidate_dirs = [
        os.path.join(SERVICE_DIR, model_dir.lstrip("./")),
        os.path.join(os.path.dirname(__file__), "..", model_dir.lstrip("./")),
    ]
    for d in candidate_dirs:
        user_path = os.path.join(d, user_model)
        post_path = os.path.join(d, post_model)
        if os.path.exists(user_path) and os.path.exists(post_path):
            return CheckResult(
                "model_checkpoint", "PASS",
                f"Found trained checkpoints at {os.path.abspath(d)}",
                {"user_model": user_path, "post_model": post_path},
            )

    checked = ", ".join(os.path.abspath(d) for d in candidate_dirs)
    return CheckResult(
        "model_checkpoint", "FAIL",
        f"No trained model checkpoint found (checked: {checked}). The service falls back to a "
        "RANDOMLY-INITIALIZED, never-trained Two-Tower model - every score it produces is noise, "
        "not a learned preference. This alone explains near-chance ranking accuracy.",
    )


def check_spring_connectivity(client: SpringMLClient) -> CheckResult:
    ok, detail = client.check_connectivity()
    if ok:
        return CheckResult("spring_api_connectivity", "PASS", f"{client.base_url} reachable ({detail})")
    return CheckResult("spring_api_connectivity", "FAIL", f"{client.base_url} unreachable ({detail})")


def check_spring_auth(client: SpringMLClient, probe_user_id) -> CheckResult:
    ok, detail = client.check_auth(probe_user_id)
    if ok:
        return CheckResult("spring_api_auth", "PASS", f"SERVICE_AUTH_TOKEN accepted ({detail})")
    return CheckResult("spring_api_auth", "FAIL", f"SERVICE_AUTH_TOKEN rejected or endpoint error ({detail})")


def check_redis() -> CheckResult:
    is_local_dev = os.environ.get("LOCAL_DEV", "True").lower() == "true"
    if is_local_dev:
        r = MockRedis(decode_responses=True)
        note = "LOCAL_DEV=true - using MockRedis, this does not exercise the real Redis/Valkey connection"
    else:
        import redis
        try:
            r = redis.Redis(
                host=os.environ.get("REDIS_HOST", "localhost"),
                port=int(os.environ.get("REDIS_PORT", 6379)),
                password=os.environ.get("REDIS_PASSWORD") or None,
                ssl=os.environ.get("REDIS_SSL", "False").lower() == "true",
                socket_timeout=10, decode_responses=True,
            )
            note = f"connected to {os.environ.get('REDIS_HOST')}:{os.environ.get('REDIS_PORT')}"
        except Exception as e:
            return CheckResult("redis_connectivity", "FAIL", f"Could not construct Redis client: {e}")

    try:
        key = "ml-eval:healthcheck"
        r.setex(key, 30, "ok")
        val = r.get(key)
        r.delete(key)
        if val in ("ok", b"ok"):
            return CheckResult("redis_connectivity", "PASS", f"Round-trip set/get/delete succeeded ({note})")
        return CheckResult("redis_connectivity", "FAIL", f"Round-trip returned unexpected value: {val!r} ({note})")
    except Exception as e:
        return CheckResult("redis_connectivity", "FAIL", f"Round-trip failed: {e} ({note})")


def check_two_tower_scoring(client: SpringMLClient, user_ids: List, content_type: str = "POSTS") -> CheckResult:
    per_user_top = {}
    all_scores = []
    problems = []

    for uid in user_ids:
        user_vector = client.get_user_vector(uid)
        candidates = client.get_candidates(uid, content_type=content_type, limit=100)
        if user_vector is None or not candidates:
            problems.append(f"user {uid}: could not fetch vector/candidates for scoring")
            continue

        post_ids = list(candidates.keys())
        matrix = np.stack([candidates[pid] for pid in post_ids])
        scores = compute_scores(user_vector.reshape(1, -1), matrix, content_type.lower())[0]

        if np.isnan(scores).any() or np.isinf(scores).any():
            problems.append(f"user {uid}: NaN/Inf in scores")
        if scores.min() < -1e-6 or scores.max() > 1 + 1e-6:
            problems.append(f"user {uid}: scores outside [0,1] range ({scores.min():.4f}, {scores.max():.4f})")

        all_scores.extend(scores.tolist())
        top = sorted(zip(post_ids, scores), key=lambda x: x[1], reverse=True)[:5]
        per_user_top[uid] = [pid for pid, _ in top]

    if not all_scores:
        return CheckResult("two_tower_scoring", "FAIL", "Could not score any users - no data returned", {})

    score_std = float(np.std(all_scores))
    score_mean = float(np.mean(all_scores))
    users_with_data = list(per_user_top.keys())
    identical_top5_across_users = (
        len(users_with_data) >= 2 and
        len({tuple(v) for v in per_user_top.values()}) == 1
    )

    data = {"score_mean": score_mean, "score_std": score_std, "per_user_top5": per_user_top}

    if problems:
        return CheckResult("two_tower_scoring", "FAIL", "; ".join(problems), data)
    if score_std < DEGENERATE_SCORE_STD_THRESHOLD:
        return CheckResult(
            "two_tower_scoring", "WARN",
            f"Score std ({score_std:.4f}) is near zero across {len(all_scores)} scores - the model is "
            "barely distinguishing between candidates (consistent with an untrained model).",
            data,
        )
    if identical_top5_across_users:
        return CheckResult(
            "two_tower_scoring", "WARN",
            f"Top-5 ranking is IDENTICAL across all {len(users_with_data)} sampled users - scores may not "
            "actually depend on the user vector.",
            data,
        )
    return CheckResult(
        "two_tower_scoring", "PASS",
        f"Scored {len(all_scores)} candidates across {len(users_with_data)} users, "
        f"mean={score_mean:.4f} std={score_std:.4f}, all in [0,1], per-user rankings differ.",
        data,
    )


def check_metadata_enhancer(client: SpringMLClient, user_id, content_type: str = "posts") -> CheckResult:
    try:
        enhancer = MetadataEnhancer(api_base_url=client.base_url, redis_client=MockRedis(decode_responses=True))
    except Exception as e:
        return CheckResult("metadata_enhancer", "FAIL", f"Could not construct MetadataEnhancer: {e}")

    candidates = client.get_candidates(user_id, content_type=content_type.upper(), limit=50)
    if not candidates:
        return CheckResult("metadata_enhancer", "WARN", f"No candidates available for user {user_id} to test with")

    post_ids = list(candidates.keys())
    base_scores = np.random.RandomState(0).uniform(0.3, 0.7, size=len(post_ids))

    try:
        enhanced = enhancer.enhance_scores(user_id=str(user_id), post_ids=post_ids, base_scores=base_scores,
                                            candidates=None, content_type=content_type)
    except Exception as e:
        return CheckResult("metadata_enhancer", "FAIL", f"enhance_scores raised: {e}")

    if len(enhanced) != len(post_ids):
        return CheckResult("metadata_enhancer", "FAIL",
                            f"enhance_scores returned {len(enhanced)} scores for {len(post_ids)} inputs")

    changed = int(np.sum(~np.isclose(enhanced, base_scores)))
    if changed == 0:
        return CheckResult("metadata_enhancer", "WARN",
                            f"enhance_scores left all {len(post_ids)} scores unchanged - eligibility/behavioral/"
                            "avoided-signal logic may not be doing anything for this user/content.")
    return CheckResult("metadata_enhancer", "PASS",
                        f"enhance_scores modified {changed}/{len(post_ids)} scores (eligibility filtering + "
                        "behavioral boosts are having an effect)")


def check_content_filter(client: SpringMLClient, user_id, content_type: str = "POSTS") -> CheckResult:
    toxicity_url = os.environ.get("TOXICITY_SERVICE_URL", client.base_url)
    try:
        cfilter = ContentFilter(api_base_url=client.base_url, toxicity_service_url=toxicity_url)
    except Exception as e:
        return CheckResult("content_filter", "FAIL", f"Could not construct ContentFilter: {e}")

    candidates = client.get_candidates(user_id, content_type=content_type, limit=50)
    if not candidates:
        return CheckResult("content_filter", "WARN", f"No candidates available for user {user_id} to test with")

    post_ids = list(candidates.keys())
    scores = np.random.RandomState(0).uniform(0.3, 0.7, size=len(post_ids))

    try:
        filtered_ids, filtered_scores, meta = cfilter.filter_recommendations(
            user_id=int(user_id) if str(user_id).isdigit() else 0, post_ids=post_ids, scores=scores,
        )
    except Exception as e:
        return CheckResult("content_filter", "FAIL", f"filter_recommendations raised: {e}")

    if "error" in meta:
        return CheckResult("content_filter", "FAIL", f"filter_recommendations reported an internal error: {meta['error']}")

    blocked = meta.get("blocked_posts", 0)
    if blocked == len(post_ids):
        return CheckResult("content_filter", "WARN",
                            f"ALL {len(post_ids)} candidates were blocked - check toxicity service connectivity "
                            "and fallback thresholds.", meta)
    return CheckResult("content_filter", "PASS",
                        f"{len(post_ids)} -> {len(filtered_ids)} posts "
                        f"(blocked={blocked}, downranked={meta.get('downranked_posts', 0)}, "
                        f"warned={meta.get('warned_posts', 0)})", meta)


def check_live_service(live_url: str, user_ids: List) -> CheckResult:
    try:
        resp = requests.get(f"{live_url}/health", timeout=5)
    except Exception as e:
        return CheckResult("live_service_e2e", "WARN", f"Service not reachable at {live_url} ({e}) - skipping")
    if resp.status_code != 200:
        return CheckResult("live_service_e2e", "WARN", f"{live_url}/health returned {resp.status_code} - skipping")

    seen_across_users = []
    problems = []
    for uid in user_ids:
        try:
            r = requests.post(f"{live_url}/recommendations", json={"userId": str(uid), "limit": 10}, timeout=30)
        except Exception as e:
            problems.append(f"user {uid}: request failed ({e})")
            continue
        if r.status_code != 200:
            problems.append(f"user {uid}: HTTP {r.status_code}")
            continue
        body = r.json()
        post_ids = body.get("postIds", [])
        scores = body.get("scores", [])
        if not post_ids:
            problems.append(f"user {uid}: 0 posts returned ({body.get('message', 'no message')})")
            continue
        if len(set(post_ids)) != len(post_ids):
            problems.append(f"user {uid}: duplicate post IDs in response")
        if scores and (min(scores) < 0 or max(scores) > 1.01):
            problems.append(f"user {uid}: scores out of [0,1] range")
        seen_across_users.append(tuple(post_ids))

    if problems:
        return CheckResult("live_service_e2e", "FAIL", "; ".join(problems))
    if len(seen_across_users) >= 2 and len(set(seen_across_users)) == 1:
        return CheckResult("live_service_e2e", "WARN",
                            "Identical recommendation list returned for every sampled user")
    return CheckResult("live_service_e2e", "PASS",
                        f"/recommendations returned valid, non-degenerate responses for {len(seen_across_users)} users")


def run_health_check(user_ids: List, api_url: Optional[str] = None, live_url: Optional[str] = None,
                      content_type: str = "POSTS") -> List[CheckResult]:
    client = SpringMLClient(base_url=api_url)
    probe_user = user_ids[0]

    results = [
        check_model_checkpoint(),
        check_spring_connectivity(client),
        check_spring_auth(client, probe_user),
        check_redis(),
        check_two_tower_scoring(client, user_ids, content_type=content_type),
        check_metadata_enhancer(client, probe_user, content_type=content_type.lower()),
        check_content_filter(client, probe_user, content_type=content_type),
    ]
    if live_url:
        results.append(check_live_service(live_url, user_ids))
    return results


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--user-ids", type=str, required=True, help="Comma-separated list of real user IDs")
    p.add_argument("--api-url", type=str, default=None, help="Override SPRING_API_URL")
    p.add_argument("--live-url", type=str, default=None,
                   help="Base URL of a running core-recommendations Flask service to end-to-end test "
                        "(e.g. http://localhost:5000). Skipped if not provided or unreachable.")
    p.add_argument("--content-type", type=str, default="POSTS", choices=["POSTS", "TRAILERS"])
    return p.parse_args()


def main():
    load_dotenv()
    args = _parse_args()
    user_ids = [u.strip() for u in args.user_ids.split(",") if u.strip()]

    results = run_health_check(user_ids, api_url=args.api_url, live_url=args.live_url,
                                content_type=args.content_type)

    print("\n=== COMPONENT HEALTH CHECK ===")
    status_order = {"FAIL": 0, "WARN": 1, "PASS": 2}
    for r in sorted(results, key=lambda r: status_order[r.status]):
        marker = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]"}[r.status]
        print(f"{marker} {r.name}: {r.detail}")

    fails = sum(1 for r in results if r.status == "FAIL")
    warns = sum(1 for r in results if r.status == "WARN")
    print(f"\n{len(results)} checks: {len(results) - fails - warns} passed, {warns} warnings, {fails} failed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
