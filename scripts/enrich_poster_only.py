"""
Chỉ lấy poster_path từ TMDB (không lấy overview/cast/director)
→ UI có poster
→ Features vẫn chỉ dùng genres/tags (precision cao)
"""

import sys, time, requests, pandas as pd
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from src.data_loader import load_movies, load_links

TMDB_API_KEY = "YOUR_API_KEY_HERE"  # ← Thay API key
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "movies_with_posters.csv"

def get_poster(tmdb_id):
    if pd.isna(tmdb_id):
        return {}
    try:
        url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}"
        r = requests.get(url, params={"api_key": TMDB_API_KEY}, timeout=5)
        if r.status_code == 200:
            data = r.json()
            return {"poster_path": data.get("poster_path", "")}
        return {}
    except:
        return {}

def main():
    movies = load_movies(enriched=False)
    links = load_links()
    df = movies.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
    
    enriched = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        poster_info = get_poster(row.get("tmdbId"))
        enriched_row = row.to_dict()
        enriched_row.update(poster_info)
        enriched.append(enriched_row)
        time.sleep(0.05)
    
    final = pd.DataFrame(enriched).sort_values("movieId")
    final.to_csv(OUTPUT_PATH, index=False)
    print(f"✓ Saved to {OUTPUT_PATH}")
    print(f"Posters: {final['poster_path'].notna().sum()}/{len(final)}")

if __name__ == "__main__":
    main()