"""
scripts/visualization.py

Generate comprehensive visualizations for recommendation system:
- Dataset statistics
- Model performance comparison
- User behavior analysis
- Item popularity distribution

Usage:
    python scripts/visualization.py
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from src.data_loader import load_movies, load_ratings, load_links

OUTPUT_DIR = BASE_DIR / "data" / "processed" / "visualizations"
METRICS_PATH = BASE_DIR / "data" / "processed" / "metrics.json"


def load_metrics() -> dict:
    """Load evaluation metrics"""
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def create_rating_distribution():
    """1. Rating distribution histogram"""
    ratings = load_ratings()
    
    fig = px.histogram(
        ratings,
        x="rating",
        nbins=10,
        title="📊 Phân bố Rating",
        labels={"rating": "Rating", "count": "Số lượng"},
        color_discrete_sequence=["#636EFA"]
    )
    
    fig.update_layout(
        template="plotly_dark",
        showlegend=False,
        xaxis_title="Rating",
        yaxis_title="Số lượng",
        hovermode="x unified"
    )
    
    return fig


def create_genre_frequency():
    """2. Genre frequency bar chart"""
    movies = load_movies()
    ratings = load_ratings()
    
    df = ratings.merge(movies[["movieId", "genres"]], on="movieId", how="left")
    
    # Explode genres
    genre_df = (
        df.dropna(subset=["genres"])
        .assign(genres=df["genres"].astype(str).str.split("|"))
        .explode("genres")
    )
    
    genre_counts = genre_df["genres"].value_counts().head(15).reset_index()
    genre_counts.columns = ["genre", "count"]
    
    fig = px.bar(
        genre_counts,
        x="genre",
        y="count",
        title="🎬 Tần suất thể loại (Top 15)",
        labels={"genre": "Thể loại", "count": "Số ratings"},
        color="count",
        color_continuous_scale="Viridis"
    )
    
    fig.update_layout(
        template="plotly_dark",
        xaxis_tickangle=-45,
        showlegend=False
    )
    
    return fig


def create_top_movies():
    """3. Top movies by rating count"""
    movies = load_movies()
    ratings = load_ratings()
    
    df = ratings.merge(movies[["movieId", "title"]], on="movieId", how="left")
    top_items = df["title"].value_counts().head(15).reset_index()
    top_items.columns = ["title", "count"]
    
    fig = px.bar(
        top_items,
        x="count",
        y="title",
        orientation="h",
        title="🏆 Top phim theo số lượng rating (Top 15)",
        labels={"title": "Phim", "count": "Số ratings"},
        color="count",
        color_continuous_scale="Sunset"
    )
    
    fig.update_layout(
        template="plotly_dark",
        yaxis=dict(autorange="reversed"),
        showlegend=False
    )
    
    return fig


def create_genre_rating_heatmap():
    """4. Heatmap: Genre vs Rating"""
    movies = load_movies()
    ratings = load_ratings()
    
    df = ratings.merge(movies[["movieId", "genres"]], on="movieId", how="left")
    
    # Explode and filter top 10 genres
    genre_df = (
        df.dropna(subset=["genres"])
        .assign(genres=df["genres"].astype(str).str.split("|"))
        .explode("genres")
    )
    
    top_genres = genre_df["genres"].value_counts().head(10).index.tolist()
    g2 = genre_df[genre_df["genres"].isin(top_genres)].copy()
    g2["rating_round"] = g2["rating"].round().astype(int).clip(1, 5)
    
    # Pivot table
    pivot = pd.pivot_table(
        g2,
        index="genres",
        columns="rating_round",
        values="movieId",
        aggfunc="count",
        fill_value=0
    )
    
    pivot = pivot.reindex(sorted(pivot.index), axis=0)
    
    fig = px.imshow(
        pivot,
        title="🔥 Heatmap: Thể loại × Rating",
        labels=dict(x="Rating", y="Thể loại", color="Số lượng"),
        color_continuous_scale="YlOrRd",
        aspect="auto"
    )
    
    fig.update_layout(template="plotly_dark")
    
    return fig


def create_model_comparison():
    """5. Model performance comparison"""
    metrics = load_metrics()
    
    if not metrics:
        return None
    
    # Extract precision and recall
    models = []
    precisions = []
    recalls = []
    
    if "collaborative_svd" in metrics:
        models.append("SVD")
        precisions.append(metrics["collaborative_svd"].get("precision@10", 0) * 100)
        recalls.append(metrics["collaborative_svd"].get("recall@10", 0) * 100)
    
    if "collaborative_itemknn" in metrics:
        models.append("ItemKNN")
        precisions.append(metrics["collaborative_itemknn"].get("precision@10", 0) * 100)
        recalls.append(metrics["collaborative_itemknn"].get("recall@10", 0) * 100)
    
    if "content_based_user_profile" in metrics:
        models.append("Content-Based")
        precisions.append(metrics["content_based_user_profile"].get("precision@10", 0) * 100)
        recalls.append(metrics["content_based_user_profile"].get("recall@10", 0) * 100)
    
    # Create grouped bar chart
    fig = go.Figure(data=[
        go.Bar(name="Precision@10", x=models, y=precisions, marker_color="#636EFA"),
        go.Bar(name="Recall@10", x=models, y=recalls, marker_color="#EF553B")
    ])
    
    fig.update_layout(
        title="📈 So sánh hiệu suất các mô hình",
        xaxis_title="Model",
        yaxis_title="Score (%)",
        barmode="group",
        template="plotly_dark",
        legend=dict(x=0.7, y=1)
    )
    
    return fig


def create_rating_prediction_comparison():
    """6. RMSE/MAE comparison (if available)"""
    metrics = load_metrics()
    
    if not metrics or "collaborative_svd" not in metrics:
        return None
    
    svd_metrics = metrics["collaborative_svd"]
    rmse = svd_metrics.get("rmse", 0)
    mae = svd_metrics.get("mae", 0)
    
    fig = go.Figure(data=[
        go.Bar(
            x=["RMSE", "MAE"],
            y=[rmse, mae],
            marker_color=["#00CC96", "#AB63FA"],
            text=[f"{rmse:.3f}", f"{mae:.3f}"],
            textposition="outside"
        )
    ])
    
    fig.update_layout(
        title="🎯 SVD Rating Prediction Error",
        yaxis_title="Error",
        template="plotly_dark",
        showlegend=False
    )
    
    return fig


def create_user_activity_distribution():
    """7. User activity distribution"""
    ratings = load_ratings()
    
    user_activity = ratings.groupby("userId").size().reset_index(name="count")
    
    fig = px.histogram(
        user_activity,
        x="count",
        nbins=50,
        title="👥 Phân bố hoạt động người dùng",
        labels={"count": "Số ratings/user", "density": "Tần suất"},
        color_discrete_sequence=["#FFA15A"]
    )
    
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Số ratings",
        yaxis_title="Số lượng users",
        showlegend=False
    )
    
    # Add median line
    median_activity = user_activity["count"].median()
    fig.add_vline(
        x=median_activity,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Median: {median_activity:.0f}",
        annotation_position="top right"
    )
    
    return fig


def create_item_popularity_distribution():
    """8. Item popularity (long tail)"""
    ratings = load_ratings()
    
    item_popularity = ratings.groupby("movieId").size().reset_index(name="count")
    item_popularity = item_popularity.sort_values("count", ascending=False).reset_index(drop=True)
    item_popularity["rank"] = item_popularity.index + 1
    
    fig = px.line(
        item_popularity.head(1000),
        x="rank",
        y="count",
        title="📉 Long Tail: Độ phổ biến của phim (Top 1000)",
        labels={"rank": "Xếp hạng", "count": "Số ratings"},
        color_discrete_sequence=["#19D3F3"]
    )
    
    fig.update_layout(
        template="plotly_dark",
        yaxis_type="log",
        yaxis_title="Số ratings (log scale)",
        showlegend=False
    )
    
    return fig


def create_rating_timeline():
    """9. Rating activity over time (if timestamp available)"""
    ratings = load_ratings()
    
    if "timestamp" not in ratings.columns:
        return None
    
    # Convert timestamp to datetime
    ratings["date"] = pd.to_datetime(ratings["timestamp"], unit="s")
    ratings["month"] = ratings["date"].dt.to_period("M").astype(str)
    
    timeline = ratings.groupby("month").size().reset_index(name="count")
    
    fig = px.line(
        timeline,
        x="month",
        y="count",
        title="📅 Hoạt động rating theo thời gian",
        labels={"month": "Tháng", "count": "Số ratings"},
        color_discrete_sequence=["#FECB52"]
    )
    
    fig.update_layout(
        template="plotly_dark",
        xaxis_tickangle=-45,
        showlegend=False
    )
    
    return fig


def create_comprehensive_dashboard():
    """10. Combined dashboard with all metrics"""
    metrics = load_metrics()
    
    if not metrics:
        print("⚠️  No metrics found. Run evaluate_models.py first.")
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Model Precision@10 (%)",
            "Model Recall@10 (%)",
            "SVD Error Metrics",
            "Dataset Statistics"
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "table"}]
        ]
    )
    
    # 1. Precision comparison
    models = []
    precisions = []
    recalls = []
    
    for key in ["collaborative_svd", "collaborative_itemknn", "content_based_user_profile"]:
        if key in metrics:
            model_name = key.replace("collaborative_", "").replace("_user_profile", "").upper()
            models.append(model_name)
            precisions.append(metrics[key].get("precision@10", 0) * 100)
            recalls.append(metrics[key].get("recall@10", 0) * 100)
    
    fig.add_trace(
        go.Bar(x=models, y=precisions, marker_color="#636EFA"),
        row=1, col=1
    )
    
    # 2. Recall comparison
    fig.add_trace(
        go.Bar(x=models, y=recalls, marker_color="#EF553B"),
        row=1, col=2
    )
    
    # 3. Error metrics
    if "collaborative_svd" in metrics:
        svd = metrics["collaborative_svd"]
        fig.add_trace(
            go.Bar(
                x=["RMSE", "MAE"],
                y=[svd.get("rmse", 0), svd.get("mae", 0)],
                marker_color=["#00CC96", "#AB63FA"]
            ),
            row=2, col=1
        )
    
    # 4. Dataset stats table
    ratings = load_ratings()
    movies = load_movies()
    
    stats_data = [
        ["Total Ratings", f"{len(ratings):,}"],
        ["Total Movies", f"{len(movies):,}"],
        ["Total Users", f"{ratings['userId'].nunique():,}"],
        ["Avg Ratings/User", f"{len(ratings)/ratings['userId'].nunique():.1f}"],
        ["Avg Ratings/Movie", f"{len(ratings)/len(movies):.1f}"],
        ["Sparsity", f"{(1 - len(ratings)/(ratings['userId'].nunique()*len(movies)))*100:.2f}%"]
    ]
    
    fig.add_trace(
        go.Table(
            header=dict(values=["Metric", "Value"], fill_color="#1f1f1f", font=dict(color="white")),
            cells=dict(values=list(zip(*stats_data)), fill_color="#2f2f2f", font=dict(color="white"))
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title_text="🎯 Recommendation System Dashboard",
        template="plotly_dark",
        showlegend=False,
        height=800
    )
    
    return fig


def main():
    print("="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    visualizations = [
        ("rating_distribution", create_rating_distribution, "Rating Distribution"),
        ("genre_frequency", create_genre_frequency, "Genre Frequency"),
        ("top_movies", create_top_movies, "Top Movies"),
        ("genre_rating_heatmap", create_genre_rating_heatmap, "Genre-Rating Heatmap"),
        ("model_comparison", create_model_comparison, "Model Comparison"),
        ("rating_prediction", create_rating_prediction_comparison, "Rating Prediction Error"),
        ("user_activity", create_user_activity_distribution, "User Activity"),
        ("item_popularity", create_item_popularity_distribution, "Item Popularity"),
        ("rating_timeline", create_rating_timeline, "Rating Timeline"),
        ("dashboard", create_comprehensive_dashboard, "Comprehensive Dashboard"),
    ]
    
    print("\nGenerating visualizations...")
    
    for filename, func, title in visualizations:
        try:
            print(f"  - {title}...", end=" ")
            fig = func()
            
            if fig is None:
                print("SKIPPED (no data)")
                continue
            
            output_path = OUTPUT_DIR / f"{filename}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(fig.to_plotly_json(), f, ensure_ascii=False)

            print(f"✓ Saved to {output_path}")

            
        except Exception as e:
            print(f"✗ ERROR: {e}")
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nView visualizations by opening HTML files in browser.")
    print("="*70)


if __name__ == "__main__":
    main()