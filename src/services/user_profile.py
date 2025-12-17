# src/services/user_profile.py
from __future__ import annotations
import pandas as pd
from src.data_loader import load_ratings

def build_user_seed_movies(user_id: int, like_threshold: float = 4.0, max_seeds: int = 5) -> list[int]:
    r = load_ratings()
    ur = r[r["userId"].astype(int) == int(user_id)]
    liked = ur[ur["rating"] >= like_threshold].sort_values("rating", ascending=False)
    return liked["movieId"].astype(int).tolist()[:max_seeds]

def explain_cbf_from_seeds(seed_movie_ids: list[int], movies_df: pd.DataFrame) -> str:
    seeds = movies_df[movies_df["movieId"].astype(int).isin(list(map(int, seed_movie_ids)))]
    if seeds.empty:
        return "Gợi ý dựa trên các phim bạn đã thích gần đây."

    genre_counts = {}
    for g in seeds["genres"].fillna("").tolist():
        for p in str(g).split("|"):
            p = p.strip()
            if p:
                genre_counts[p] = genre_counts.get(p, 0) + 1

    if not genre_counts:
        return "Gợi ý dựa trên nội dung/tags của các phim bạn đã thích."

    top = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:2]
    top_genres = ", ".join([t[0] for t in top])
    return f"Gợi ý vì bạn thường thích thể loại: {top_genres}."
