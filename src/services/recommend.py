
# src/services/recommend.py (SIMPLIFIED VERSION)
from __future__ import annotations


from dataclasses import dataclass
from typing import Optional, List

from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import vstack, load_npz
from sklearn.metrics.pairwise import cosine_similarity

from src.data_loader import load_ratings, load_links
from src.preprocessing import build_movie_features
from src.models.content_based import ContentBasedRecommender
from src.models.item_knn import ItemKNN
from src.services.user_profile import (
    build_user_seed_movies,
    explain_cbf_from_seeds,
)
_SYSTEM_READY = False
# ========== SIMPLE THRESHOLD ==========
MIN_RATINGS_FOR_COLLAB = 10  # Duy nhất 1 threshold
MIN_RATINGS_FOR_CF = MIN_RATINGS_FOR_COLLAB  # Alias for backward compatibility

# ========== PATHS & GLOBAL CACHE ==========
BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "data" / "processed" / "models"
KNN_PATH = MODELS_DIR / "item_knn.pkl"

FEATURES_DIR = BASE_DIR / "data" / "processed" / "features"
MOVIES_PATH = FEATURES_DIR / "movies_df.parquet"
X_PATH = FEATURES_DIR / "X_features.npz"

_MOVIES_DF: Optional[pd.DataFrame] = None
_LINKS_DF: Optional[pd.DataFrame] = None
_X = None
_CB: Optional[ContentBasedRecommender] = None
_KNN: Optional[ItemKNN] = None


@dataclass
class RecResult:
    mode: str
    explanation: str
    recs_df: pd.DataFrame


def init_system() -> None:
    global _SYSTEM_READY
    global _MOVIES_DF, _LINKS_DF, _X, _CB, _KNN

    if _SYSTEM_READY:
        return  # ✅ QUAN TRỌNG: đã init thì thôi

    print("=" * 60)
    print("🔄 INITIALIZING RECOMMENDATION SYSTEM")
    print("=" * 60)

    # 1️⃣ Load movies + features (chỉ 1 lần)
    # ƯU TIÊN: load từ file đã build sẵn (scripts/build_content_features.py)
    global _MOVIES_DF, _X, _LINKS_DF, _CB
    try:
        if MOVIES_PATH.exists() and X_PATH.exists():
            print(f"  Loading content features from {FEATURES_DIR} ...")
            _MOVIES_DF = pd.read_parquet(MOVIES_PATH)
            _X = load_npz(X_PATH)
            print("  ✓ Loaded precomputed movies_df & X")
        else:
            print("  Content features not found on disk → building and saving...")
            _MOVIES_DF, _X, _, _ = build_movie_features(
                use_enriched_movies=True,
                use_enriched_features=False,
            )
            FEATURES_DIR.mkdir(parents=True, exist_ok=True)
            _MOVIES_DF.to_parquet(MOVIES_PATH, index=False)
            from scipy.sparse import save_npz  # local import to avoid unused import if not needed

            save_npz(X_PATH, _X)
            print(f"  ✓ Saved movies_df to {MOVIES_PATH}")
            print(f"  ✓ Saved X to {X_PATH}")
    except Exception as e:
        # Fallback: build in-memory nếu load/save lỗi
        print(f"  ⚠️ Error loading/saving content features ({e}) → building in-memory")
        _MOVIES_DF, _X, _, _ = build_movie_features(
            use_enriched_movies=True,
            use_enriched_features=False,
        )

    _LINKS_DF = load_links()

    # 2️⃣ Content-Based (nhẹ hơn)
    _CB = ContentBasedRecommender(_MOVIES_DF, _X)

    # 3️⃣ ItemKNN: ƯU TIÊN LOAD TỪ DISK, CHỈ TRAIN NẾU THIẾU
    global _KNN
    try:
        if KNN_PATH.exists():
            print(f"  Loading ItemKNN model from {KNN_PATH} ...")
            _KNN = joblib.load(KNN_PATH)
            print("  ✓ Loaded pre-trained ItemKNN")
        else:
            print("  ItemKNN model not found on disk → training ONCE and saving...")
            ratings = load_ratings()
            _KNN = ItemKNN(k_neighbors=30, min_support=2)
            _KNN.fit(ratings)
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(_KNN, KNN_PATH)
            print(f"  ✓ Trained & saved ItemKNN to {KNN_PATH}")
    except Exception as e:
        # Fallback: train in-memory nếu load/save bị lỗi
        print(f"  ⚠️ Error loading/saving ItemKNN ({e}) → training in-memory")
        ratings = load_ratings()
        _KNN = ItemKNN(k_neighbors=30, min_support=2)
        _KNN.fit(ratings)

    _SYSTEM_READY = True  # ✅ ĐÁNH DẤU ĐÃ INIT

    print("✅ Recommender system READY")
    print("=" * 60 + "\n")


def _ensure_init() -> None:
    if not _SYSTEM_READY:
        init_system()


def _merge_links(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    return df.merge(_LINKS_DF, on="movieId", how="left")


def _user_rating_count(user_id: int) -> int:
    r = load_ratings()
    return int((r["userId"].astype(int) == int(user_id)).sum())


# ========== 2 FUNCTIONS - THAT'S IT! ==========

def recommend_by_seed(movie_id: int, top_k: int = 10) -> RecResult:
    """
    GUEST: Content-based item-to-item
    """
    _ensure_init()
    movie_id = int(movie_id)

    ids = _CB.recommend_similar(movie_id, top_k=top_k)
    if not ids:
        return RecResult(
            mode="content",
            explanation="Không tìm thấy phim tương tự.",
            recs_df=_MOVIES_DF.iloc[0:0],
        )

    recs_df = _MOVIES_DF[_MOVIES_DF["movieId"].astype(int).isin(ids)].copy()
    recs_df = _merge_links(recs_df)

    explanation = "Phim tương tự dựa trên thể loại và nội dung."
    return RecResult(mode="content", explanation=explanation, recs_df=recs_df)


def recommend_for_you(user_id: int, top_k: int = 10, like_threshold: float = 4.0) -> RecResult:
    """
    LOGGED USER: Simple 2-bucket strategy
    
    Rule 1: < 10 ratings → Content-Based
    Rule 2: ≥ 10 ratings → ItemKNN
    
    That's it. No hybrid. No complexity.
    """
    _ensure_init()
    uid = int(user_id)
    n = _user_rating_count(uid)

    # ========== BUCKET 1: Cold-start (< 10 ratings) ==========
    if n < MIN_RATINGS_FOR_COLLAB:
        seed_ids = build_user_seed_movies(uid, like_threshold=like_threshold, max_seeds=5)
        
        if not seed_ids:
            return RecResult(
                mode="content",
                explanation=f"Bạn mới có {n} ratings. Hãy rate thêm phim để nhận gợi ý cá nhân hoá!",
                recs_df=_MOVIES_DF.iloc[0:0],
            )
        
        # Build content profile
        vecs = []
        for mid in seed_ids:
            v = _CB.get_vector(int(mid))
            if v is not None:
                vecs.append(v)

        if not vecs:
            return RecResult(
                mode="content",
                explanation="Không tạo được hồ sơ từ lịch sử của bạn.",
                recs_df=_MOVIES_DF.iloc[0:0],
            )

        profile = vecs[0] if len(vecs) == 1 else vstack(vecs).mean(axis=0)
        sims = cosine_similarity(profile, _X).ravel()

        # Remove seen
        ratings = load_ratings()
        seen = set(ratings[ratings["userId"].astype(int) == uid]["movieId"].astype(int).tolist())
        id_to_idx = {int(mid): i for i, mid in enumerate(_MOVIES_DF["movieId"].astype(int).tolist())}
        for mid in seen:
            if mid in id_to_idx:
                sims[id_to_idx[mid]] = -1e9

        top_idx = sims.argsort()[::-1][:top_k]
        ids = _MOVIES_DF.iloc[top_idx]["movieId"].astype(int).tolist()

        recs_df = _MOVIES_DF[_MOVIES_DF["movieId"].astype(int).isin(ids)].copy()
        recs_df = _merge_links(recs_df)

        explanation = explain_cbf_from_seeds(seed_ids, _MOVIES_DF)
        return RecResult(mode="content", explanation=explanation, recs_df=recs_df)

    # ========== BUCKET 2: Warm-start (≥ 10 ratings) ==========
    else:
        ids = _KNN.recommend_for_user(uid, top_k=top_k)
        
        if not ids:
            return RecResult(
                mode="collab",
                explanation="Không đủ dữ liệu để tạo gợi ý.",
                recs_df=_MOVIES_DF.iloc[0:0],
            )
        
        recs_df = _MOVIES_DF[_MOVIES_DF["movieId"].astype(int).isin(ids)].copy()
        recs_df = _merge_links(recs_df)
        
        explanation = "Gợi ý dựa trên hành vi của những người dùng có sở thích giống bạn."
        return RecResult(mode="collab", explanation=explanation, recs_df=recs_df)