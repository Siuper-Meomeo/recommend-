# src/models/collaborative_svd.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


@dataclass
class SVDCollaborative:
    n_factors: int = 80
    random_state: int = 42
    rating_min: float = 0.5
    rating_max: float = 5.0
    bias_weight: float = 0.3  # ← ADD THIS: Reduce bias influence (0.0-1.0)

    # learned params
    user2idx: Optional[Dict[int, int]] = None
    item2idx: Optional[Dict[int, int]] = None
    idx2item: Optional[np.ndarray] = None

    user_factors: Optional[np.ndarray] = None  # (n_users, k)
    item_factors: Optional[np.ndarray] = None  # (n_items, k)

    global_mean: float = 0.0
    user_bias: Optional[np.ndarray] = None
    item_bias: Optional[np.ndarray] = None

    user_seen: Optional[Dict[int, set]] = None

    def fit(self, ratings_df: pd.DataFrame) -> None:
        df = ratings_df[["userId", "movieId", "rating"]].copy()
        df["userId"] = df["userId"].astype(int)
        df["movieId"] = df["movieId"].astype(int)
        df["rating"] = df["rating"].astype(float)

        # drop NA
        df = df.dropna(subset=["userId", "movieId", "rating"])
        if df.empty:
            raise ValueError("ratings_df rỗng, không thể fit SVD.")

        users = df["userId"].unique()
        items = df["movieId"].unique()

        self.user2idx = {u: i for i, u in enumerate(users)}
        self.item2idx = {m: j for j, m in enumerate(items)}
        self.idx2item = np.array(items, dtype=np.int64)

        ui = df["userId"].map(self.user2idx).to_numpy()
        ii = df["movieId"].map(self.item2idx).to_numpy()
        rr = df["rating"].to_numpy()

        n_users = len(users)
        n_items = len(items)

        R = csr_matrix((rr, (ui, ii)), shape=(n_users, n_items))

        # global mean
        self.global_mean = float(rr.mean())

        # user bias
        user_sum = np.asarray(R.sum(axis=1)).ravel()
        user_cnt = np.asarray((R != 0).sum(axis=1)).ravel()
        user_mean = user_sum / np.maximum(user_cnt, 1)
        self.user_bias = (user_mean - self.global_mean).astype(np.float32)

        # item bias
        item_sum = np.asarray(R.sum(axis=0)).ravel()
        item_cnt = np.asarray((R != 0).sum(axis=0)).ravel()
        item_mean = item_sum / np.maximum(item_cnt, 1)
        self.item_bias = (item_mean - self.global_mean).astype(np.float32)

        # centered
        centered = rr - (self.global_mean + self.user_bias[ui] + self.item_bias[ii])
        R_centered = csr_matrix((centered, (ui, ii)), shape=(n_users, n_items))

        # choose k safely
        min_dim = min(n_users, n_items)
        if min_dim <= 2:
            k = 1
        else:
            k = int(min(self.n_factors, min_dim - 1))

        svd = TruncatedSVD(n_components=k, random_state=self.random_state)
        U = svd.fit_transform(R_centered)      # (n_users, k)
        Vt = svd.components_                   # (k, n_items)

        self.user_factors = U.astype(np.float32)
        self.item_factors = Vt.T.astype(np.float32)

        # seen items (fast)
        self.user_seen = df.groupby("userId")["movieId"].apply(lambda s: set(map(int, s.tolist()))).to_dict()

    def can_recommend(self, user_id: int) -> bool:
        return self.user2idx is not None and int(user_id) in self.user2idx

    def predict_score(self, user_id: int, movie_id: int, clip: bool = True) -> float:
        if self.user2idx is None or self.item2idx is None:
            raise RuntimeError("SVD model chưa fit().")

        uid = int(user_id)
        mid = int(movie_id)

        if uid not in self.user2idx or mid not in self.item2idx:
            pred = float(self.global_mean)
            return float(np.clip(pred, self.rating_min, self.rating_max)) if clip else pred

        u = self.user2idx[uid]
        i = self.item2idx[mid]

        dot = float(np.dot(self.user_factors[u], self.item_factors[i]))
        pred = self.global_mean + float(self.user_bias[u]) + float(self.item_bias[i]) + dot

        if clip:
            pred = float(np.clip(pred, self.rating_min, self.rating_max))
        return pred

    def recommend_for_user(self, user_id: int, top_k: int = 10, candidate_k: int = 500) -> List[int]:
        if self.user2idx is None or self.item2idx is None:
            raise RuntimeError("SVD model chưa fit().")

        uid = int(user_id)
        if uid not in self.user2idx:
            return []

        u = self.user2idx[uid]
        seen = self.user_seen.get(uid, set()) if self.user_seen else set()

        # ========== FIX: REDUCE BIAS INFLUENCE ==========
        # Original: scores = base + item_bias + dot_product
        # Fixed: scores = base + (bias_weight * item_bias) + dot_product
        
        base = self.global_mean + float(self.user_bias[u])
        dot_products = self.item_factors @ self.user_factors[u]
        
        # Apply bias_weight to reduce popularity bias
        scores = base + (self.bias_weight * self.item_bias) + dot_products
        # ================================================

        # filter seen
        if seen:
            seen_idx = [self.item2idx[m] for m in seen if m in self.item2idx]
            if seen_idx:
                scores = scores.copy()
                scores[seen_idx] = -1e9

        n_items = scores.shape[0]
        cand_k = int(min(max(candidate_k, top_k), n_items))

        # faster than full sort
        top_idx = np.argpartition(-scores, cand_k - 1)[:cand_k]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        recs = []
        for i in top_idx:
            if scores[i] <= -1e8:
                continue
            recs.append(int(self.idx2item[i]))
            if len(recs) >= top_k:
                break
        return recs