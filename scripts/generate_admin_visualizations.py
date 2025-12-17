"""
scripts/generate_admin_visualizations.py

Generate HTML visualizations cho admin dashboard.
Chạy script này trước khi deploy để pre-generate các biểu đồ.

Usage:
    python scripts/generate_admin_visualizations.py
"""

import sys
from pathlib import Path
import pandas as pd
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from src.data_loader import load_movies, load_ratings

OUTPUT_DIR = BASE_DIR / "app" / "static" / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_admin_visualizations():
    """Generate HTML visualizations cho admin dashboard"""
    print("="*70)
    print("GENERATING ADMIN VISUALIZATIONS")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    ratings = load_ratings()
    movies = load_movies()
    
    # Sample data để giảm memory (max 100k rows)
    max_rows = min(100000, len(ratings))
    if len(ratings) > max_rows:
        print(f"Sampling {max_rows:,} rows from {len(ratings):,} total ratings...")
        ratings_sample = ratings.sample(n=max_rows, random_state=42)
    else:
        ratings_sample = ratings
    
    # Merge với movies
    df = ratings_sample.merge(
        movies[["movieId", "title", "genres"]],
        on="movieId",
        how="left"
    )
    
    print(f"Working with {len(df):,} rows")
    
    # ======================
    # 1. Rating Distribution
    # ======================
    print("\n1. Generating Rating Distribution...", end=" ")
    try:
        fig_rating = px.histogram(
            df,
            x="rating",
            nbins=10,
            title="Rating Distribution"
        )
        html_rating = fig_rating.to_html(full_html=False)
        
        output_path = OUTPUT_DIR / "rating_distribution.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_rating)
        print(f"✓ Saved to {output_path}")
    except Exception as e:
        print(f"✗ ERROR: {e}")
    
    # ======================
    # 2. Genre Frequency
    # ======================
    print("2. Generating Genre Frequency...", end=" ")
    try:
        g = (
            df.dropna(subset=["genres"])
              .assign(genres=df["genres"].astype(str).str.split("|"))
              .explode("genres")
        )
        genre_counts = g["genres"].value_counts().head(15).reset_index()
        genre_counts.columns = ["genre", "count"]
        
        fig_genre = px.bar(
            genre_counts,
            x="genre",
            y="count",
            title="Genre Frequency (Top 15)"
        )
        html_genre = fig_genre.to_html(full_html=False)
        
        output_path = OUTPUT_DIR / "genre_frequency.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_genre)
        print(f"✓ Saved to {output_path}")
    except Exception as e:
        print(f"✗ ERROR: {e}")
    
    # ======================
    # 3. Top Movies
    # ======================
    print("3. Generating Top Movies...", end=" ")
    try:
        top_items = df["title"].value_counts().head(15).reset_index()
        top_items.columns = ["title", "count"]
        
        fig_top = px.bar(
            top_items,
            x="title",
            y="count",
            title="Top Movies by Rating Count"
        )
        fig_top.update_layout(xaxis_tickangle=-30)
        html_top = fig_top.to_html(full_html=False)
        
        output_path = OUTPUT_DIR / "top_movies.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_top)
        print(f"✓ Saved to {output_path}")
    except Exception as e:
        print(f"✗ ERROR: {e}")
    
    # ======================
    # 4. Genre × Rating Heatmap
    # ======================
    print("4. Generating Genre-Rating Heatmap...", end=" ")
    try:
        top10_genres = set(genre_counts["genre"].head(10))
        g2 = g[g["genres"].isin(top10_genres)].copy()
        g2["rating_round"] = g2["rating"].round().astype(int).clip(1, 5)
        
        pivot = pd.pivot_table(
            g2,
            index="genres",
            columns="rating_round",
            values="movieId",
            aggfunc="count",
            fill_value=0
        )
        
        fig_heat = px.imshow(
            pivot,
            title="Genre × Rating Heatmap",
            aspect="auto"
        )
        html_heat = fig_heat.to_html(full_html=False)
        
        output_path = OUTPUT_DIR / "genre_rating_heatmap.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_heat)
        print(f"✓ Saved to {output_path}")
    except Exception as e:
        print(f"✗ ERROR: {e}")
    
    print("\n" + "="*70)
    print("VISUALIZATION GENERATION COMPLETE!")
    print("="*70)
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nAll visualizations are ready for admin dashboard.")
    print("="*70)


if __name__ == "__main__":
    generate_admin_visualizations()

