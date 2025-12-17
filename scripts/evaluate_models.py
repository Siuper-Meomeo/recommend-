# scripts/evaluate_models.py
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import vstack
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from src.data_loader import load_ratings
from src.preprocessing import build_movie_features
from src.models.collaborative_svd import SVDCollaborative
from src.models.item_knn import ItemKNN
from src.models.content_based import ContentBasedRecommender

# ---------------- CONFIG ----------------
K = 10                           # Top-K recommend để chấm
HOLDOUT_K = 5                    # Leave-K-out cho test (mỗi user giữ lại K item)
LIKE_THRESHOLD = 4.0             # định nghĩa "liked" cho Top-N metrics
MIN_RATINGS_PER_USER = 10        # lọc user có đủ lịch sử
N_SAMPLE_USERS = 500
RANDOM_SEED = 42

METRICS_PATH = BASE_DIR / "data" / "processed" / "metrics.json"


def train_test_split_random_liked(
    ratings: pd.DataFrame,
    k: int = 5,
    like_threshold: float = 4.0,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Random leave-K-out: mỗi user random K phim liked làm test
    (Không bias theo thời gian)
    """
    np.random.seed(random_state)
    
    df = ratings.copy()
    df["userId"] = df["userId"].astype(int)
    df["movieId"] = df["movieId"].astype(int)
    df["rating"] = df["rating"].astype(float)
    
    df = df.dropna(subset=["userId", "movieId", "rating"])
    
    liked = df[df["rating"] >= like_threshold]
    if liked.empty:
        return df, df.iloc[0:0].copy()
    
    # Random K phim liked mỗi user
    test_idx = []
    for uid, group in liked.groupby("userId"):
        if len(group) >= k:
            sampled = group.sample(n=k, random_state=random_state)
            test_idx.extend(sampled.index.tolist())
    
    test = df.loc[test_idx].copy()
    train = df.drop(index=test_idx).copy()
    
    return train, test


def build_user_profile_recs(
    uid: int,
    train_df: pd.DataFrame,
    cb: ContentBasedRecommender,
    movies_df: pd.DataFrame,
    X,
    top_k: int = 10,
    like_threshold: float = 4.0,
) -> list[int]:
    """
    Content-based user profile:
    - lấy các phim user liked trong TRAIN
    - average vector -> cosine similarity với toàn bộ phim
    - exclude phim đã seen trong TRAIN
    """
    ur = train_df[train_df["userId"] == uid]
    liked = ur[ur["rating"] >= like_threshold]
    if liked.empty:
        return []

    vecs = []
    for mid in liked["movieId"].astype(int).tolist():
        v = cb.get_vector(int(mid))
        if v is not None:
            vecs.append(v)

    if not vecs:
        return []

    profile = vecs[0] if len(vecs) == 1 else vstack(vecs).mean(axis=0)

    # np.matrix -> np.array (an toàn)
    if hasattr(profile, "A"):
        profile = np.asarray(profile)

    sims = cosine_similarity(profile, X).ravel()

    # remove seen (train)
    seen = set(ur["movieId"].astype(int).tolist())

    # map movieId -> row index in movies_df
    id_to_idx = {int(mid): i for i, mid in enumerate(movies_df["movieId"].astype(int).tolist())}
    for mid in seen:
        idx = id_to_idx.get(mid)
        if idx is not None:
            sims[idx] = -1e9

    top_idx = np.argsort(sims)[::-1][:top_k]
    return movies_df.iloc[top_idx]["movieId"].astype(int).tolist()


def precision_recall_at_k(recs: list[int], gt: set[int], k: int) -> tuple[float, float]:
    """
    gt là set các item relevant trong TEST (ở đây là K phim liked cuối).
    """
    if k <= 0:
        return 0.0, 0.0
    if not gt:
        return 0.0, 0.0

    recs_k = recs[:k]
    hits = len(set(recs_k) & gt)
    precision = hits / k
    recall = hits / len(gt)
    return float(precision), float(recall)


def main():
    np.random.seed(RANDOM_SEED)

    ratings = load_ratings()

    # filter users with enough total ratings (ổn định hơn cho CF)
    cnt = ratings.groupby("userId").size()
    eligible_users = cnt[cnt >= MIN_RATINGS_PER_USER].index.astype(int).tolist()
    if not eligible_users:
        raise RuntimeError("Không có user nào đủ MIN_RATINGS_PER_USER để evaluate.")

    # sample users
    sample_users = np.random.choice(
        eligible_users, size=min(N_SAMPLE_USERS, len(eligible_users)), replace=False
    ).astype(int).tolist()

    # ---------- split train/test ----------
    train_df, test_df = train_test_split_random_liked(
        ratings, k=HOLDOUT_K, like_threshold=LIKE_THRESHOLD, random_state=RANDOM_SEED
    )

    # Build content features once
    # Use ORIGINAL data (non-enriched) for baseline comparison
    movies_df, X, _, _ = build_movie_features(
        use_enriched_movies=False,     # Use ORIGINAL movies.csv
        use_enriched_features=False    # Only use genres/tags
    )
    cb = ContentBasedRecommender(movies_df, X)

    # ========== FIT 2 MODELS ==========
    print("\n" + "="*70)
    print("TRAINING MODELS")
    print("="*70)
    
    # 1. SVD for rating prediction (RMSE/MAE)
    print("Training SVD for rating prediction...")
    svd = SVDCollaborative(n_factors=20, bias_weight=0.0, random_state=RANDOM_SEED)
    svd.fit(train_df)
    print(f"  ✓ SVD trained with {len(svd.item2idx)} items")
    
    # 2. ItemKNN for top-N recommendations (Precision/Recall)
    print("Training ItemKNN for top-N recommendations...")
    knn = ItemKNN(k_neighbors=50, min_support=3)
    knn.fit(train_df)
    print(f"  ✓ ItemKNN trained with {len(knn.item2idx)} items")
    
    print("="*70 + "\n")

    # ---------- RMSE/MAE using SVD ----------
    print("Evaluating rating prediction (RMSE/MAE) using SVD...")
    y_true, y_pred = [], []
    for _, row in test_df.iterrows():
        uid = int(row["userId"])
        if uid not in sample_users:
            continue
        mid = int(row["movieId"])
        y_true.append(float(row["rating"]))
        y_pred.append(float(svd.predict_score(uid, mid)))

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred))) if y_true else None
    mae = float(mean_absolute_error(y_true, y_pred)) if y_true else None
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}\n")

    # ---------- Precision/Recall@K ----------
    test_by_user = dict(tuple(test_df.groupby("userId")))

    # ItemKNN aggregates (for top-N)
    sum_prec_knn = sum_rec_knn = 0.0
    n_eval_knn = 0

    # SVD aggregates (for comparison)
    sum_prec_svd = sum_rec_svd = 0.0
    n_eval_svd = 0

    # CB aggregates
    sum_prec_cb = sum_rec_cb = 0.0
    n_eval_cb = 0

    # ========== DETAILED DEBUG FOR FIRST 3 USERS ==========
    print("="*70)
    print("DETAILED DEBUG - FIRST 3 USERS")
    print("="*70)

    for idx, uid in enumerate(sample_users[:3]):
        uid = int(uid)
        if uid not in test_by_user:
            continue
        
        gt = set(test_by_user[uid]["movieId"].astype(int).tolist())
        if not gt:
            continue
        
        print(f"\n{'='*70}")
        print(f"USER {uid}")
        print(f"{'='*70}")
        print(f"Ground truth: {sorted(gt)}")
        
        # ItemKNN recommendations
        knn_recs = knn.recommend_for_user(uid, top_k=K)
        print(f"\nItemKNN top-{K}: {knn_recs}")
        print(f"  Hits: {set(knn_recs) & gt}")
        
        # SVD recommendations (for comparison)
        svd_recs = svd.recommend_for_user(uid, top_k=K)
        print(f"\nSVD top-{K}: {svd_recs}")
        print(f"  Hits: {set(svd_recs) & gt}")

    print("\n" + "="*70)
    print("END DEBUG")
    print("="*70 + "\n")

    # ========== MAIN EVALUATION LOOP ==========
    print("Evaluating top-N recommendations...")
    
    for uid in sample_users:
        uid = int(uid)
        if uid not in test_by_user:
            continue

        gt = set(test_by_user[uid]["movieId"].astype(int).tolist())
        if not gt:
            continue

        # --- ItemKNN recs (PRIMARY for top-N) ---
        knn_recs = knn.recommend_for_user(uid, top_k=K)
        if knn_recs:
            p_knn, r_knn = precision_recall_at_k(knn_recs, gt, K)
            sum_prec_knn += p_knn
            sum_rec_knn += r_knn
            n_eval_knn += 1

        # --- SVD recs (for comparison) ---
        svd_recs = svd.recommend_for_user(uid, top_k=K)
        if svd_recs:
            p_svd, r_svd = precision_recall_at_k(svd_recs, gt, K)
            sum_prec_svd += p_svd
            sum_rec_svd += r_svd
            n_eval_svd += 1

        # --- CB recs ---
        cb_recs = build_user_profile_recs(
            uid=uid,
            train_df=train_df,
            cb=cb,
            movies_df=movies_df,
            X=X,
            top_k=K,
            like_threshold=LIKE_THRESHOLD,
        )

        if cb_recs:
            p_cb, r_cb = precision_recall_at_k(cb_recs, gt, K)
            sum_prec_cb += p_cb
            sum_rec_cb += r_cb
            n_eval_cb += 1

    # ========== SAVE METRICS ==========
    metrics = {
        "K": K,
        "holdout_k": HOLDOUT_K,
        "like_threshold": LIKE_THRESHOLD,
        "min_ratings_per_user": MIN_RATINGS_PER_USER,
        "n_sample_users": len(sample_users),
        "n_eval_users_knn": n_eval_knn,
        "n_eval_users_svd": n_eval_svd,
        "n_eval_users_cb": n_eval_cb,
        "collaborative_svd": {
            "note": "Used for rating prediction only",
            "rmse": rmse,
            "mae": mae,
            f"precision@{K}": (sum_prec_svd / n_eval_svd) if n_eval_svd else 0.0,
            f"recall@{K}": (sum_rec_svd / n_eval_svd) if n_eval_svd else 0.0,
        },
        "collaborative_itemknn": {
            "note": "Used for top-N recommendations",
            f"precision@{K}": (sum_prec_knn / n_eval_knn) if n_eval_knn else 0.0,
            f"recall@{K}": (sum_rec_knn / n_eval_knn) if n_eval_knn else 0.0,
        },
        "content_based_user_profile": {
            f"precision@{K}": (sum_prec_cb / n_eval_cb) if n_eval_cb else 0.0,
            f"recall@{K}": (sum_rec_cb / n_eval_cb) if n_eval_cb else 0.0,
        },
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nSaved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()