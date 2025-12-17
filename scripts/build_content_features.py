"""
scripts/build_content_features.py

Precompute content-based features (movies_df + TF-IDF matrix X) ONE TIME
and save to disk, so that the web app only needs to LOAD them.

Usage:
    python scripts/build_content_features.py
"""

import sys
from pathlib import Path

import pandas as pd
from scipy.sparse import save_npz

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from src.preprocessing import build_movie_features  # noqa: E402


FEATURES_DIR = BASE_DIR / "data" / "processed" / "features"
MOVIES_PATH = FEATURES_DIR / "movies_df.parquet"
X_PATH = FEATURES_DIR / "X_features.npz"


def main():
    print("=" * 70)
    print("BUILDING CONTENT-BASED FEATURES (OFFLINE)")
    print("=" * 70)

    movies_df, X, _, _ = build_movie_features(
        use_enriched_movies=True,
        use_enriched_features=False,
    )

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    # Save movies_df
    movies_df.to_parquet(MOVIES_PATH, index=False)
    print(f"✓ Saved movies_df to {MOVIES_PATH}")

    # Save sparse feature matrix
    save_npz(X_PATH, X)
    print(f"✓ Saved X (sparse features) to {X_PATH}")

    print("=" * 70)
    print("DONE. Web app can now LOAD these features instead of recomputing.")
    print("=" * 70)


if __name__ == "__main__":
    main()


