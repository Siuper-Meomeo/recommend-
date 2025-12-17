"""
scripts/enrich_tmdb_to_csv.py

Làm giàu movies.csv với dữ liệu từ TMDB API:
- Overview (plot summary)
- Poster URL
- Vote average/count
- Popularity
- Release date
- Production countries/companies

Usage:
    python scripts/enrich_tmdb_to_csv.py
"""

import sys
import time
import os
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from src.data_loader import load_movies, load_links

# ========== CONFIG ==========
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "YOUR_TMDB_API_KEY_HERE")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
RATE_LIMIT_DELAY = 0.05  # 50ms delay giữa các request (20 req/s)

OUTPUT_PATH = BASE_DIR / "data" / "processed" / "movies_enriched.csv"


def get_tmdb_movie_details(tmdb_id: int) -> dict:
    """
    Fetch movie details from TMDB API
    
    Returns dict with keys:
    - overview, poster_path, vote_average, vote_count, popularity,
      release_date, runtime, budget, revenue, original_language
    """
    if pd.isna(tmdb_id) or tmdb_id <= 0:
        return {}
    
    url = f"{TMDB_BASE_URL}/movie/{int(tmdb_id)}"
    params = {"api_key": TMDB_API_KEY}
    
    try:
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            return {
                "overview": data.get("overview", ""),
                "poster_path": data.get("poster_path", ""),
                "vote_average": data.get("vote_average", 0.0),
                "vote_count": data.get("vote_count", 0),
                "popularity": data.get("popularity", 0.0),
                "release_date": data.get("release_date", ""),
                "runtime": data.get("runtime", 0),
                "budget": data.get("budget", 0),
                "revenue": data.get("revenue", 0),
                "original_language": data.get("original_language", ""),
                "status": data.get("status", ""),
            }
        elif response.status_code == 404:
            # Movie not found in TMDB
            return {}
        else:
            print(f"  Warning: TMDB API returned {response.status_code} for tmdbId={tmdb_id}")
            return {}
    
    except requests.exceptions.RequestException as e:
        print(f"  Error fetching tmdbId={tmdb_id}: {e}")
        return {}


def enrich_movies():
    """
    Main enrichment workflow
    """
    print("="*70)
    print("ENRICHING MOVIES WITH TMDB DATA")
    print("="*70)
    
    # Check API key
    if TMDB_API_KEY == "YOUR_TMDB_API_KEY_HERE":
        print("\n❌ ERROR: Bạn chưa thay TMDB_API_KEY!")
        print("\nCách lấy API key:")
        print("1. Đăng ký tại: https://www.themoviedb.org/signup")
        print("2. Vào Settings → API → Request API Key")
        print("3. Copy API Key (v3 auth) và paste vào script\n")
        return
    
    # Load data
    print("\n1. Loading data...")
    movies = load_movies()
    links = load_links()
    
    # Merge to get tmdbId
    df = movies.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
    
    # Filter movies that need enrichment
    # (có tmdbId và chưa có overview)
    if "overview" in df.columns:
        to_enrich = df[df["overview"].isna() & df["tmdbId"].notna()].copy()
        print(f"   Found {len(to_enrich)} movies to enrich (missing overview)")
    else:
        to_enrich = df[df["tmdbId"].notna()].copy()
        print(f"   Found {len(to_enrich)} movies to enrich (all with tmdbId)")
    
    if len(to_enrich) == 0:
        print("\n✓ All movies already enriched!")
        return
    
    # Ask user confirmation
    print(f"\n2. Will fetch data for {len(to_enrich)} movies from TMDB")
    print(f"   Estimated time: ~{len(to_enrich) * RATE_LIMIT_DELAY:.1f}s")
    confirm = input("\n   Continue? (y/n): ").strip().lower()
    
    if confirm != "y":
        print("Cancelled.")
        return
    
    # Enrich
    print("\n3. Fetching TMDB data...")
    enriched_data = []
    
    for _, row in tqdm(to_enrich.iterrows(), total=len(to_enrich), desc="   Progress"):
        tmdb_id = row["tmdbId"]
        details = get_tmdb_movie_details(tmdb_id)
        
        # Merge with original row
        enriched_row = row.to_dict()
        enriched_row.update(details)
        enriched_data.append(enriched_row)
        
        # Rate limiting
        time.sleep(RATE_LIMIT_DELAY)
    
    # Create enriched dataframe
    enriched_df = pd.DataFrame(enriched_data)
    
    # Merge with movies that don't need enrichment
    already_enriched = df[~df["movieId"].isin(enriched_df["movieId"])]
    final_df = pd.concat([already_enriched, enriched_df], ignore_index=True)
    
    # Sort by movieId
    final_df = final_df.sort_values("movieId").reset_index(drop=True)
    
    # Save
    print(f"\n4. Saving to {OUTPUT_PATH}...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    
    # Stats
    print("\n" + "="*70)
    print("ENRICHMENT COMPLETE!")
    print("="*70)
    print(f"Total movies: {len(final_df)}")
    print(f"Movies with overview: {final_df['overview'].notna().sum()}")
    print(f"Movies with poster: {final_df['poster_path'].notna().sum()}")
    print(f"Average vote_average: {final_df['vote_average'].mean():.2f}")
    print(f"Output: {OUTPUT_PATH}")
    print("="*70)
    
    # Sample
    print("\nSample enriched data:")
    print(final_df[["title", "overview", "vote_average", "popularity"]].head(3))


def main():
    try:
        enrich_movies()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress saved.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()