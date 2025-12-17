"""
scripts/train_knn_model.py

Train ItemKNN model OFFLINE and save to disk so that the web app
only needs to LOAD the model (no retraining on Render / production).

This version includes options to reduce model size:
- Filter to top-N most-rated movies (max_items)
- Increase min_support to make similarity matrix sparser
- Use joblib compression

Usage (ví dụ giảm size mạnh):
    python scripts/train_knn_model.py --max-items 8000 --k-neighbors 30 --min-support 3
"""

import sys
import argparse
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from src.data_loader import load_ratings  # noqa: E402
from src.models.item_knn import ItemKNN  # noqa: E402


MODELS_DIR = BASE_DIR / "data" / "processed" / "models"
KNN_PATH = MODELS_DIR / "item_knn.pkl"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-items", type=int, default=12000,
                        help="Giới hạn số movie phổ biến nhất để giảm size (default: 12000)")
    parser.add_argument("--k-neighbors", type=int, default=30,
                        help="Số neighbors cho ItemKNN (default: 30)")
    parser.add_argument("--min-support", type=int, default=3,
                        help="Số user tối thiểu để tạo similarity (default: 3)")
    parser.add_argument("--compress", type=int, default=3,
                        help="joblib compress level (0-9), default: 3")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("TRAINING ItemKNN MODEL (OFFLINE)")
    print("=" * 70)
    print(f"Config: max_items={args.max_items}, k_neighbors={args.k_neighbors}, min_support={args.min_support}")

    ratings = load_ratings()
    print(f"Loaded ratings: {len(ratings):,} rows")

    # Filter to top-N popular movies to shrink model size
    if args.max_items and args.max_items > 0:
        movie_counts = ratings.groupby("movieId").size().reset_index(name="cnt")
        top_movies = movie_counts.sort_values("cnt", ascending=False).head(args.max_items)["movieId"]
        ratings = ratings[ratings["movieId"].isin(top_movies)]
        print(f"Filtered to top-{args.max_items} movies → ratings now: {len(ratings):,}")

    model = ItemKNN(k_neighbors=args.k_neighbors, min_support=args.min_support)
    model.fit(ratings)
    print(f"✓ ItemKNN trained with {len(model.item2idx)} items")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, KNN_PATH, compress=args.compress)
    print(f"\n✓ Saved ItemKNN model to: {KNN_PATH} (compress={args.compress})")
    print("=" * 70)


if __name__ == "__main__":
    main()


