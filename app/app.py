# app/app.py
import sys
import json
from pathlib import Path
from functools import wraps
import pandas as pd

from flask import Flask, render_template, request, redirect, url_for, session

# -------------------------------------------------
# Setup import path
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

# -------------------------------------------------
# Import services
# -------------------------------------------------
from src.services.recommend import (
    init_system,
    recommend_by_seed,
    recommend_for_you,
    MIN_RATINGS_FOR_COLLAB as MIN_RATINGS_FOR_CF
)
from src.data_loader import load_links, load_movies, load_ratings
from src.tmdb_client import get_poster_url

# -------------------------------------------------
# Flask app
# -------------------------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "maimeo"

# # init system at start - cache models and data to save memory
# init_system()

# admin password
ADMIN_PASSWORD = "admin123"


# -------------------------------------------------
# Utils
# -------------------------------------------------
# Cache for UI data to avoid reloading on every request
_MOVIES_UI_CACHE = None
_RATINGS_CACHE = None

def admin_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    return wrapper


def _load_movies_df_for_ui():
    """Load movies for UI with caching to save memory"""
    global _MOVIES_UI_CACHE
    if _MOVIES_UI_CACHE is None:
        movies = load_movies(enriched=True)
        links = load_links()
        _MOVIES_UI_CACHE = movies.merge(
            links[["movieId", "tmdbId", "imdbId"]],
            on="movieId",
            how="left"
        )
    return _MOVIES_UI_CACHE


def _get_ratings_cached():
    """Get ratings with caching to save memory"""
    global _RATINGS_CACHE
    if _RATINGS_CACHE is None:
        _RATINGS_CACHE = load_ratings()
    return _RATINGS_CACHE


def search_movie_by_title(query: str, limit: int = 20):
    q = str(query or "").strip().lower()
    if not q:
        return _load_movies_df_for_ui().iloc[0:0]

    movies = _load_movies_df_for_ui()
    mask = movies["title"].astype(str).str.lower().str.contains(q, na=False)
    return movies[mask].head(limit)


# -------------------------------------------------
# HOME (search)
# -------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    movies = []
    query = ""

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        df = search_movie_by_title(query, limit=20)

        movies = [
            {
                "movieId": int(row.movieId),
                "title": row.title,
                "genres": row.genres if "genres" in df.columns else "",
            }
            for _, row in df.iterrows()
        ]

    return render_template(
        "home.html",
        movies=movies,
        query=query,
        user_id=session.get("user_id"),
    )


# -------------------------------------------------
# REGISTER / LOGIN / LOGOUT
# -------------------------------------------------
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"
USER_PASSWORD = "user123"

@app.route("/register", methods=["GET", "POST"])
def register():
    error = None
    ratings = _get_ratings_cached()
    max_uid = int(ratings["userId"].max()) if len(ratings) else 0
    suggested = max_uid + 1

    if request.method == "POST":
        user_id_str = request.form.get("user_id", "").strip()
        try:
            user_id = int(user_id_str) if user_id_str else suggested
            session["user_id"] = user_id
            session.pop("admin_logged_in", None)
            return redirect(url_for("for_you"))
        except ValueError:
            error = "UserId không hợp lệ"

    return render_template("register.html", error=error, suggested_user_id=suggested)


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        # Admin login
        if username.lower() == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session["admin_logged_in"] = True
            session.pop("user_id", None)
            return redirect(url_for("admin_view"))

        # User login
        if password != USER_PASSWORD:
            error = "Sai mật khẩu."
            return render_template("login.html", error=error)

        try:
            user_id = int(username)
        except ValueError:
            error = "User đăng nhập bằng userId (số). Admin dùng username = admin."
            return render_template("login.html", error=error)

        session["user_id"] = user_id
        session.pop("admin_logged_in", None)
        return redirect(url_for("for_you"))

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.pop("user_id", None)
    session.pop("admin_logged_in", None)
    return redirect(url_for("home"))


# -------------------------------------------------
# FOR YOU
# -------------------------------------------------
@app.route("/for-you")
def for_you():
    user_id = session.get("user_id")
    if user_id is None:
        return redirect(url_for("login"))

    user_id_int = int(user_id)
    
    # ======================
    # Lấy 10 rating gần đây nhất của user (tối ưu memory)
    # ======================
    ratings = _get_ratings_cached()
    
    # Filter user ratings (chỉ tạo view, không copy)
    user_mask = ratings["userId"].astype(int) == user_id_int
    user_ratings = ratings[user_mask]
    n_ratings = len(user_ratings)  # Tính ngay từ đây, tránh query lại

    recent_ratings = []
    if n_ratings > 0:
        # Chỉ lấy các cột cần thiết trước khi sort để giảm memory
        cols_needed = ["movieId", "rating"]
        if "timestamp" in user_ratings.columns:
            cols_needed.append("timestamp")
        
        user_ratings_subset = user_ratings[cols_needed]
        
        # Sort và lấy top 10 (không copy toàn bộ)
        if "timestamp" in user_ratings_subset.columns:
            user_ratings_top10 = user_ratings_subset.nlargest(10, "timestamp")
        else:
            # Nếu không có timestamp, lấy 10 dòng cuối cùng (giả định là mới nhất)
            user_ratings_top10 = user_ratings_subset.tail(10)

        # Join với movies để lấy title, genres (chỉ merge với 10 dòng)
        movies = _load_movies_df_for_ui()
        merged = user_ratings_top10.merge(
            movies[["movieId", "title", "genres"]],
            on="movieId",
            how="left"
        )

        for _, row in merged.iterrows():
            ts = row.get("timestamp")
            ts_str = ""
            if pd.notna(ts):
                try:
                    ts_str = pd.to_datetime(int(ts), unit="s").strftime("%Y-%m-%d %H:%M")
                except Exception:
                    ts_str = str(ts)

            recent_ratings.append(
                {
                    "movieId": int(row.get("movieId")),
                    "title": row.get("title", "Unknown"),
                    "genres": row.get("genres", ""),
                    "rating": float(row.get("rating", 0.0)),
                    "timestamp": ts_str,
                }
            )

    result = recommend_for_you(user_id_int, top_k=10)

    recs = []
    for _, row in result.recs_df.iterrows():
        # Use helper to get poster
        poster = get_poster_url(
            tmdb_id=row.get("tmdbId"),
            poster_path=row.get("poster_path") or row.get("poster_url")
        )
        
        recs.append({
            "movieId": int(row.get("movieId")),
            "title": row.get("title"),
            "genres": row.get("genres", ""),
            "avg_rating": float(row.get("avg_rating", 0.0)),
            "poster_url": poster,
        })

    return render_template(
        "for_you.html",
        user_id=user_id,
        mode=result.mode,
        explanation=result.explanation,
        recs=recs,
        n_ratings=n_ratings,
        k_threshold=MIN_RATINGS_FOR_CF,
        recent_ratings=recent_ratings,
    )


# -------------------------------------------------
# MOVIE DETAIL (ENHANCED WITH FULL INFO)
# -------------------------------------------------
@app.route("/movie/<int:movie_id>")
def movie_detail(movie_id: int):
    user_id = session.get("user_id")

    # Load movies with all enriched data
    movies = _load_movies_df_for_ui()
    
    # ========== DEBUG: Check columns ==========
    print(f"\n=== MOVIE DETAIL DEBUG ===")
    print(f"Movies columns: {movies.columns.tolist()}")
    print(f"Has poster_path? {'poster_path' in movies.columns}")
    # =========================================
    
    # Filter by movieId
    movie_df = movies[movies["movieId"].astype(int) == int(movie_id)]

    if movie_df.empty:
        return "Không tìm thấy phim", 404

    row = movie_df.iloc[0]
    
    # ========== DEBUG: Check row values ==========
    print(f"Main movie: {row.get('title')}")
    print(f"  tmdbId: {row.get('tmdbId')}")
    print(f"  poster_path: {row.get('poster_path', 'N/A')}")
    # =============================================
    
    # Get poster URL
    poster = get_poster_url(
        tmdb_id=row.get("tmdbId"),
        poster_path=row.get("poster_path") or row.get("poster_url")
    )
    
    print(f"  poster URL result: {poster}")  # ← DEBUG
    
    # ... rest of code ...
    
    # Build comprehensive movie info
    movie_info = {
        # Basic info
        "movieId": int(row.get("movieId")),
        "title": row.get("title", "Unknown"),
        "genres": row.get("genres", ""),
        "poster_url": poster,
        
        # Ratings
        "avg_rating": float(row.get("avg_rating", 0.0)) if pd.notna(row.get("avg_rating")) else 0.0,
        "rating_count": int(row.get("rating_count", 0)) if pd.notna(row.get("rating_count")) else 0,
        "vote_average": float(row.get("vote_average", 0.0)) if pd.notna(row.get("vote_average")) else None,
        "vote_count": int(row.get("vote_count", 0)) if pd.notna(row.get("vote_count")) else None,
        
        # Details
        "overview": row.get("overview", ""),
        "tagline": row.get("tagline", ""),
        "release_date": row.get("release_date", ""),
        "runtime": int(row.get("runtime", 0)) if pd.notna(row.get("runtime")) else None,
        
        # People
        "director": row.get("director", ""),
        "cast_top10": row.get("cast_top10", ""),
        
        # Production
        "production_companies": row.get("production_companies", ""),
        "production_countries": row.get("production_countries", ""),
        "original_language": row.get("original_language", ""),
        
        # Financial
        "budget": int(row.get("budget", 0)) if pd.notna(row.get("budget")) else None,
        "revenue": int(row.get("revenue", 0)) if pd.notna(row.get("revenue")) else None,
        
        # Status
        "status": row.get("status", ""),
        
        # Links
        "tmdbId": int(row.get("tmdbId")) if pd.notna(row.get("tmdbId")) else None,
        "imdbId": int(row.get("imdbId")) if pd.notna(row.get("imdbId")) else None,
    }

    # Get recommendations
    result = recommend_by_seed(movie_id, top_k=12)  # Get more recommendations
    
    # ========== DEBUG: Check recommendations ==========
    print(f"\nRecommendations dataframe columns: {result.recs_df.columns.tolist()}")
    print(f"Has poster_path in recs? {'poster_path' in result.recs_df.columns}")
    if len(result.recs_df) > 0:
        first_rec = result.recs_df.iloc[0]
        print(f"First rec: {first_rec.get('title')}")
        print(f"  poster_path: {first_rec.get('poster_path', 'N/A')}")
    print(f"=== END DEBUG ===\n")
    # ==================================================
    
    recs = []
    for _, r in result.recs_df.iterrows():
        rec_poster = get_poster_url(
            tmdb_id=r.get("tmdbId"),
            poster_path=r.get("poster_path") or r.get("poster_url")
        )
        
        recs.append({
            "movieId": int(r.get("movieId")),
            "title": r.get("title"),
            "genres": r.get("genres", ""),
            "avg_rating": float(r.get("avg_rating", 0.0)),
            "poster_url": rec_poster,
        })

    return render_template(
        "movie_detail.html",
        user_id=user_id,
        movie=movie_info,
        recs=recs,
        mode="content",
    )


# -------------------------------------------------
# ADMIN
# -------------------------------------------------
@app.route("/admin")
@admin_required
def admin_view():
    # ======================
    # Load data (use cached versions)
    # ======================
    ratings = _get_ratings_cached()
    movies = _load_movies_df_for_ui()

    # ======================
    # DATASET STATISTICS
    # ======================
    n_ratings = len(ratings)
    n_movies = movies["movieId"].nunique()
    n_users = ratings["userId"].nunique()

    avg_rating = ratings["rating"].mean()
    median_rating = ratings["rating"].median()

    sparsity = 100 * (1 - n_ratings / (n_users * n_movies))

    stats = {
        "total_ratings": int(n_ratings),
        "total_movies": int(n_movies),
        "total_users": int(n_users),
        "avg_rating": float(avg_rating),
        "median_rating": float(median_rating),
        "sparsity": float(sparsity),
    }

    # ======================
    # VISUALIZATIONS (đọc từ file đã pre-generate để tránh OOM)
    # ======================
    visualizations = {}
    
    # Đường dẫn đến các file HTML đã generate
    viz_dir = BASE_DIR / "app" / "static" / "visualizations"
    
    viz_files = {
        "rating_distribution": viz_dir / "rating_distribution.html",
        "genre_frequency": viz_dir / "genre_frequency.html",
        "top_movies": viz_dir / "top_movies.html",
        "genre_rating_heatmap": viz_dir / "genre_rating_heatmap.html",
    }
    
    # Đọc các file HTML đã generate
    for key, file_path in viz_files.items():
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    visualizations[key] = f.read()
            except Exception as e:
                print(f"Warning: Could not load {key}: {e}")
                visualizations[key] = f"<div class='alert alert-warning'>Visualization not available. Run: python scripts/generate_admin_visualizations.py</div>"
        else:
            # Fallback: generate on-the-fly nếu file chưa có (cho development)
            visualizations[key] = f"<div class='alert alert-info'>Visualization not pre-generated. Run: python scripts/generate_admin_visualizations.py</div>"

    # ======================
    # LOAD MODEL METRICS (optional)
    # ======================
    metrics_path = BASE_DIR / "data" / "processed" / "metrics.json"
    metrics = {}

    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    # ======================
    # RENDER
    # ======================
    return render_template(
        "admin_view.html",
        stats=stats,
        metrics=metrics,
        visualizations=visualizations,
    )



# -------------------------------------------------
# RUN
# -------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
