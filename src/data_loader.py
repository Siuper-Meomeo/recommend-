# src/data_loader.py
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]


def load_movies(enriched: bool = False) -> pd.DataFrame:
    """
    Load movies data
    
    Args:
        enriched: If True, load enriched data with TMDB info (overview, popularity, etc.)
                  Falls back to original if enriched file doesn't exist
    
    Returns:
        DataFrame with columns: movieId, title, genres, [overview, poster_url, ...]
    """
    if enriched:
        enriched_path = BASE_DIR / "data" / "processed" / "movies_enriched.csv"
        if enriched_path.exists():
            df = pd.read_csv(enriched_path, encoding="utf-8")
            print(f"✓ Loaded enriched movies: {len(df)} rows")
            return df
        else:
            print(f"⚠️  Enriched file not found at {enriched_path}, using original movies.csv")
    
    # Fallback to original
    path = BASE_DIR / "data" / "raw" / "movies.csv"
    df = pd.read_csv(path, encoding="utf-8")
    return df


def load_ratings() -> pd.DataFrame:
    """Load ratings data"""
    path = BASE_DIR / "data" / "raw" / "ratings.csv"
    df = pd.read_csv(path, encoding="utf-8")
    return df


def load_links() -> pd.DataFrame:
    """
    Load links data (movieId -> imdbId, tmdbId)
    
    Returns:
        DataFrame with columns: movieId, imdbId, tmdbId
    """
    path = BASE_DIR / "data" / "raw" / "links.csv"
    df = pd.read_csv(path, encoding="utf-8")
    
    # Ensure correct types
    df["movieId"] = df["movieId"].astype(int)
    df["tmdbId"] = pd.to_numeric(df["tmdbId"], errors="coerce")
    df["imdbId"] = pd.to_numeric(df["imdbId"], errors="coerce")
    
    return df


def load_tags() -> pd.DataFrame:
    """Load tags data"""
    path = BASE_DIR / "data" / "raw" / "tags.csv"
    
    if not path.exists():
        # Return empty DataFrame if tags.csv doesn't exist
        return pd.DataFrame({"userId": [], "movieId": [], "tag": [], "timestamp": []})
    
    df = pd.read_csv(path, encoding="utf-8")
    return df