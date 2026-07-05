"""
Thin client for the SERVICE-role internal ML endpoints on the Spring API
(same auth pattern the core-recommendations service already uses).
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

logger = logging.getLogger("ml-eval-spring-client")


class SpringMLClient:

    def __init__(self, base_url: str = None, auth_token: str = None, timeout: int = 15):
        self.base_url = (base_url or os.environ.get("SPRING_API_URL", "https://api-bingewise.com")).rstrip("/")
        self.auth_token = auth_token or os.environ.get("SERVICE_AUTH_TOKEN", "")
        if not self.auth_token:
            raise RuntimeError("SERVICE_AUTH_TOKEN is not set - cannot authenticate to the Spring API")
        self.timeout = timeout
        self.headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "X-Service-Role": "SERVICE",
        }

    def check_connectivity(self) -> Tuple[bool, str]:
        try:
            resp = requests.get(f"{self.base_url}/actuator/health", timeout=self.timeout)
            return resp.status_code == 200, f"HTTP {resp.status_code}"
        except Exception as e:
            return False, str(e)

    def check_auth(self, probe_user_id) -> Tuple[bool, str]:
        """Confirm SERVICE_AUTH_TOKEN is accepted by hitting a known SERVICE-role endpoint."""
        url = f"{self.base_url}/api/internal/ml/users/{probe_user_id}/vector"
        try:
            resp = requests.get(url, headers=self.headers, timeout=self.timeout)
            if resp.status_code in (200, 404):
                return True, f"HTTP {resp.status_code}"
            return False, f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as e:
            return False, str(e)

    def get_user_vector(self, user_id) -> Optional[np.ndarray]:
        url = f"{self.base_url}/api/internal/ml/users/{user_id}/vector"
        try:
            resp = requests.get(url, headers=self.headers, timeout=self.timeout)
        except Exception as e:
            logger.warning(f"get_user_vector({user_id}) request failed: {e}")
            return None
        if resp.status_code == 200:
            return np.array(resp.json(), dtype=np.float32)
        logger.warning(f"get_user_vector({user_id}) failed: {resp.status_code} {resp.text[:200]}")
        return None

    def get_post_vectors(self, post_ids: List[int]) -> Dict[int, np.ndarray]:
        """Fetch vectors for specific post IDs (e.g. posts a user already interacted
        with) via POST /api/internal/ml/posts/vectors/batch. Missing/unknown post
        IDs are simply absent from the returned dict."""
        vectors: Dict[int, np.ndarray] = {}
        if not post_ids:
            return vectors
        for i in range(0, len(post_ids), 500):
            batch = post_ids[i:i + 500]
            url = f"{self.base_url}/api/internal/ml/posts/vectors/batch"
            try:
                resp = requests.post(url, headers=self.headers, json=batch, timeout=self.timeout)
            except Exception as e:
                logger.warning(f"get_post_vectors batch request failed: {e}")
                continue
            if resp.status_code != 200:
                logger.warning(f"get_post_vectors batch failed: {resp.status_code} {resp.text[:200]}")
                continue
            for pid_str, vec in resp.json().items():
                if vec is not None:
                    vectors[int(pid_str)] = np.array(vec, dtype=np.float32)
        return vectors

    def get_post_vectors_counterfactual(self, post_ids: List[int], exclude_user_id) -> Dict[int, np.ndarray]:
        """
        Like get_post_vectors, but computed on the fly with exclude_user_id's own
        interactions excluded from each post's behavioral genre profile - NOT the
        stored/served vector. Use this (instead of get_post_vectors) for a user's
        own historical positives/negatives in offline evaluation, so the eval isn't
        comparing a user's taste vector against a post profile partly built from
        that same user's own past interaction with the post (leakage that inflates
        accuracy metrics without reflecting real, pre-interaction recommendation
        quality - real serving never re-scores a user against posts they've already
        interacted with in the first place).
        """
        vectors: Dict[int, np.ndarray] = {}
        if not post_ids:
            return vectors
        for i in range(0, len(post_ids), 500):
            batch = post_ids[i:i + 500]
            url = f"{self.base_url}/api/internal/ml/posts/vectors/counterfactual"
            body = {"postIds": batch, "excludeUserId": int(exclude_user_id)}
            try:
                resp = requests.post(url, headers=self.headers, json=body, timeout=self.timeout)
            except Exception as e:
                logger.warning(f"get_post_vectors_counterfactual batch request failed: {e}")
                continue
            if resp.status_code != 200:
                logger.warning(f"get_post_vectors_counterfactual batch failed: {resp.status_code} {resp.text[:200]}")
                continue
            for pid_str, vec in resp.json().items():
                if vec is not None:
                    vectors[int(pid_str)] = np.array(vec, dtype=np.float32)
        return vectors

    def get_interaction_history(self, user_id, limit: int = 500) -> Dict:
        """GET /api/internal/users/{userId}/interactions -> MLInteractionHistoryResponse."""
        url = f"{self.base_url}/api/internal/users/{user_id}/interactions"
        try:
            resp = requests.get(url, headers=self.headers, params={"limit": limit}, timeout=self.timeout)
        except Exception as e:
            logger.warning(f"get_interaction_history({user_id}) request failed: {e}")
            return {"userId": user_id, "interactions": [], "notInterested": []}
        if resp.status_code == 200:
            return resp.json()
        logger.warning(f"get_interaction_history({user_id}) failed: {resp.status_code} {resp.text[:200]}")
        return {"userId": user_id, "interactions": [], "notInterested": []}

    def get_candidates(self, user_id, content_type: str = "POSTS", limit: int = 200) -> Dict[int, np.ndarray]:
        """Real unseen candidates for a user, used as ranking distractors. These
        never overlap with the user's real interaction history by construction."""
        url = f"{self.base_url}/api/internal/ml/users/{user_id}/candidates"
        params = {
            "limit": limit,
            "contentType": content_type.upper(),
            "includeNewHighQuality": True,
            "newContentRatio": 0.3,
            "interactionLookbackDays": 30,
        }
        try:
            resp = requests.get(url, headers=self.headers, params=params, timeout=(10, 60))
        except Exception as e:
            logger.warning(f"get_candidates({user_id}) request failed: {e}")
            return {}
        if resp.status_code != 200:
            logger.warning(f"get_candidates({user_id}) failed: {resp.status_code} {resp.text[:200]}")
            return {}
        vectors = resp.json().get("vectors", {})
        return {int(pid): np.array(vec, dtype=np.float32) for pid, vec in vectors.items()}
