# src/models/item_knn.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ItemKNN:
    """
    Item-based Collaborative Filtering using KNN with cosine similarity.
    Often performs better than SVD for top-N recommendations.
    """
    k_neighbors: int = 50
    min_support: int = 3  # Minimum co-rated users for similarity
    
    # Learned params
    user2idx: Optional[Dict[int, int]] = None
    item2idx: Optional[Dict[int, int]] = None
    idx2item: Optional[np.ndarray] = None
    
    user_item_matrix: Optional[csr_matrix] = None
    item_similarity: Optional[csr_matrix] = None  # Keep sparse to save memory
    
    user_seen: Optional[Dict[int, set]] = None
    
    def fit(self, ratings_df: pd.DataFrame) -> None:
        df = ratings_df[["userId", "movieId", "rating"]].copy()
        df["userId"] = df["userId"].astype(int)
        df["movieId"] = df["movieId"].astype(int)
        df["rating"] = df["rating"].astype(float)
        
        df = df.dropna(subset=["userId", "movieId", "rating"])
        if df.empty:
            raise ValueError("ratings_df rỗng.")
        
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
        
        # Build user-item matrix
        self.user_item_matrix = csr_matrix((rr, (ui, ii)), shape=(n_users, n_items))
        
        # Compute item-item similarity (cosine on transpose)
        # Normalize by user (each user's ratings sum to 1)
        item_user_matrix = self.user_item_matrix.T.tocsr()
        
        # Compute cosine similarity (keep sparse to save memory)
        self.item_similarity = cosine_similarity(item_user_matrix, dense_output=False)
        
        # Filter by minimum support (work with sparse matrix)
        # Count co-rated users for each item pair
        binary_matrix = (self.user_item_matrix > 0).astype(int)
        co_rated = (binary_matrix.T @ binary_matrix)
        
        # Convert to LIL format for efficient element-wise operations
        sim_lil = self.item_similarity.tolil()
        co_rated_lil = co_rated.tolil()
        
        # Mask similarities with insufficient support (keep sparse)
        sim_lil[co_rated_lil < self.min_support] = 0
        # Remove diagonal (self-similarity)
        sim_lil.setdiag(0)
        
        # Convert back to CSR for efficient operations
        self.item_similarity = sim_lil.tocsr()
        
        # Store seen items
        self.user_seen = df.groupby("userId")["movieId"].apply(
            lambda s: set(map(int, s.tolist()))
        ).to_dict()
    
    def predict_score(self, user_id: int, movie_id: int) -> float:
        """Predict rating using weighted average of similar items."""
        if self.user2idx is None or self.item2idx is None:
            raise RuntimeError("Model chưa fit().")
        
        uid = int(user_id)
        mid = int(movie_id)
        
        if uid not in self.user2idx or mid not in self.item2idx:
            return 3.0  # Default rating
        
        u_idx = self.user2idx[uid]
        m_idx = self.item2idx[mid]
        
        # Get user's rated items
        user_ratings = self.user_item_matrix[u_idx].toarray().ravel()
        rated_items = np.where(user_ratings > 0)[0]
        
        if len(rated_items) == 0:
            return 3.0
        
        # Get similarities to target item (sparse matrix indexing)
        sims = self.item_similarity[m_idx, rated_items]
        if hasattr(sims, 'toarray'):
            sims = sims.toarray().ravel()
        
        # Weighted average
        if sims.sum() == 0:
            return 3.0
        
        pred = np.dot(sims, user_ratings[rated_items]) / sims.sum()
        return float(np.clip(pred, 0.5, 5.0))
    
    def recommend_for_user(self, user_id: int, top_k: int = 10) -> List[int]:
        """Recommend items based on similar items user has rated."""
        if self.user2idx is None or self.item2idx is None:
            raise RuntimeError("Model chưa fit().")
        
        uid = int(user_id)
        if uid not in self.user2idx:
            return []
        
        u_idx = self.user2idx[uid]
        seen = self.user_seen.get(uid, set())
        
        # Get user's rated items and their ratings
        user_ratings = self.user_item_matrix[u_idx].toarray().ravel()
        rated_items = np.where(user_ratings > 0)[0]
        
        if len(rated_items) == 0:
            return []
        
        # For each candidate item, compute score as weighted sum of similarities
        # Use sparse matrix operations for efficiency (avoid dense conversion)
        n_items = self.item_similarity.shape[0]
        
        # Get similarities for all items to user's rated items (sparse)
        sim_matrix = self.item_similarity[:, rated_items]  # (n_items, n_rated) - sparse
        
        # Compute scores using sparse matrix multiplication (memory efficient)
        user_ratings_vec = np.array(user_ratings[rated_items], dtype=np.float32)
        scores = sim_matrix.dot(user_ratings_vec)  # Sparse matrix @ vector = efficient
        
        # Convert to dense only for the final scores array (much smaller)
        if hasattr(scores, 'toarray'):
            scores = scores.toarray().ravel()
        else:
            scores = np.asarray(scores).ravel()
        
        # Set scores for seen items to -inf
        seen_indices = [self.item2idx[mid] for mid in seen if mid in self.item2idx]
        if seen_indices:
            scores[seen_indices] = -1e9
        
        # Get top-K
        top_idx = np.argsort(scores)[::-1][:top_k]
        
        recs = []
        for i in top_idx:
            if scores[i] > 0:
                recs.append(int(self.idx2item[i]))
        
        return recs[:top_k]