# app/visualization.py
"""
Module chứa các hàm tạo visualization cho admin dashboard.
Tách riêng để giảm tải khi deploy - chỉ import khi cần thiết.
"""
import pandas as pd
import plotly.express as px


def create_visualizations(df: pd.DataFrame) -> dict:
    """
    Tạo các visualization từ dataframe đã merge ratings và movies.
    
    Args:
        df: DataFrame chứa ratings và movies đã merge
        
    Returns:
        dict: Dictionary chứa các HTML visualization với keys:
            - rating_distribution
            - genre_frequency
            - top_movies
            - genre_rating_heatmap
    """
    # Rating distribution
    fig_rating = px.histogram(
        df,
        x="rating",
        nbins=10,
        title="Rating Distribution"
    )

    # Genre frequency
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

    # Top movies
    top_items = df["title"].value_counts().head(15).reset_index()
    top_items.columns = ["title", "count"]

    fig_top = px.bar(
        top_items,
        x="title",
        y="count",
        title="Top Movies by Rating Count"
    )
    fig_top.update_layout(xaxis_tickangle=-30)

    # Genre × Rating heatmap
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

    return {
        "rating_distribution": fig_rating.to_html(full_html=False),
        "genre_frequency": fig_genre.to_html(full_html=False),
        "top_movies": fig_top.to_html(full_html=False),
        "genre_rating_heatmap": fig_heat.to_html(full_html=False),
    }

