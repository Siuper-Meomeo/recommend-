# src/preprocessing.py
import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

from .data_loader import load_ratings, load_movies, load_tags, load_links

YEAR_PATTERN = re.compile(r"\((\d{4})\)$")


def extract_year(title: str):
    """Extract year from title like 'Toy Story (1995)'"""
    m = YEAR_PATTERN.search(str(title))
    if m:
        return int(m.group(1))
    return np.nan


def clean_title(title: str):
    """Remove year from title"""
    return YEAR_PATTERN.sub("", str(title)).strip()


def _ensure_col(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Ensure column exists, return empty string series if not
    (For optional enriched columns)
    """
    if col not in df.columns:
        return pd.Series([""] * len(df), index=df.index)
    return df[col].fillna("").astype(str)


def build_movie_features(
    max_tfidf_features: int = 8000,
    use_enriched_movies: bool = False,
    use_enriched_features: bool = False,
):
    """
    Build movie feature matrix for content-based filtering
    
    Args:
        max_tfidf_features: Max number of TF-IDF features
        use_enriched_movies: Whether to LOAD enriched data (for UI display)
        use_enriched_features: Whether to USE enriched fields (overview/cast) for TF-IDF
    
    Returns:
        - movies_df: DataFrame for UI/recommendations (with ALL enriched fields)
        - X: Feature matrix (TF-IDF + numeric features)
        - tfidf: Fitted TfidfVectorizer(s)
        - scaler: Fitted StandardScaler
    
    Strategy:
        - use_enriched_movies=True: Load full data (has poster, overview, cast for UI)
        - use_enriched_features=False: Only use genres/tags for TF-IDF (high precision)
    """
    
    # ========== 1. LOAD DATA ==========
    ratings = load_ratings()
    movies = load_movies(enriched=use_enriched_movies)
    tags = load_tags()
    links = load_links()
    
    # ========== 2. MERGE WITH LINKS (IMPORTANT FOR POSTER!) ==========
    # Always merge with links to get tmdbId for poster lookup
    df = movies.merge(links[["movieId", "tmdbId", "imdbId"]], on="movieId", how="left")
    
    # ========== 3. RATING STATS ==========
    stats = ratings.groupby("movieId")["rating"].agg(
        avg_rating="mean",
        rating_count="count"
    ).reset_index()
    
    df = df.merge(stats, on="movieId", how="left")
    
    # ========== 4. TAGS AGGREGATION ==========
    if "tag" in tags.columns and len(tags) > 0:
        tags_agg = (
            tags.groupby("movieId")["tag"]
            .apply(lambda x: " ".join(map(str, x)))
            .reset_index()
        )
        df = df.merge(tags_agg, on="movieId", how="left")
    
    # ========== 5. ENSURE BASIC COLUMNS ==========
    df["genres"] = df.get("genres", "(no genres listed)").fillna("(no genres listed)")
    df["tag"] = df.get("tag", "").fillna("")
    
    # Extract year and clean title
    df["year"] = df["title"].apply(extract_year)
    df["clean_title"] = df["title"].apply(clean_title)
    
    # ========== 6. TEXT FEATURES (SELECTIVE BASED ON use_enriched_features) ==========
    # Basic text (always available)
    genres_text = df["genres"].astype(str).str.replace("|", " ", regex=False)
    tags_text = df["tag"].astype(str)
    
    # Enriched text (optional)
    overview = _ensure_col(df, "overview")
    keywords = _ensure_col(df, "keywords")
    cast_top10 = _ensure_col(df, "cast_top10")
    director = _ensure_col(df, "director")
    tagline = _ensure_col(df, "tagline")
    
    # ========== DECISION POINT: Use enriched features or not? ==========
    if use_enriched_features and use_enriched_movies:
        print("  → Using ENRICHED features (genres + tags + overview + cast + director)")
        
        # Strategy: Tiered TF-IDF with weights
        df["text_high_signal"] = (
            genres_text + " " +
            genres_text + " " +
            keywords + " " +
            director
        ).str.strip()
        
        df["text_medium_signal"] = (
            tags_text + " " +
            cast_top10 + " " +
            tagline
        ).str.strip()
        
        df["text_low_signal"] = overview.str.strip()
        
        # High signal TF-IDF
        tfidf_high = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.6
        )
        X_high = tfidf_high.fit_transform(df["text_high_signal"])
        
        # Medium signal TF-IDF
        tfidf_medium = TfidfVectorizer(
            stop_words="english",
            max_features=2000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.7
        )
        X_medium = tfidf_medium.fit_transform(df["text_medium_signal"])
        
        # Low signal TF-IDF
        tfidf_low = TfidfVectorizer(
            stop_words="english",
            max_features=1000,
            ngram_range=(1, 1),
            min_df=5,
            max_df=0.8
        )
        X_low = tfidf_low.fit_transform(df["text_low_signal"])
        
        # Weighted combination
        X_text = hstack([
            X_high * 2.0,
            X_medium * 1.0,
            X_low * 0.5
        ])
        
        tfidf = (tfidf_high, tfidf_medium, tfidf_low)
        
        print(f"    - High-signal TF-IDF: {X_high.shape[1]}")
        print(f"    - Medium-signal TF-IDF: {X_medium.shape[1]}")
        print(f"    - Low-signal TF-IDF: {X_low.shape[1]}")
        
    else:
        print("  → Using BASIC features (genres + tags + keywords) for high precision")
        
        # Enhanced basic: genres + tags + keywords
        # Keywords are high-signal (like genres) but more specific
        df["text"] = (
            genres_text + " " +
            genres_text + " " +
            genres_text + " " +  # 3x weight for genres
            keywords + " " +      # Add keywords (high signal!)
            keywords + " " +      # 2x weight for keywords
            tags_text
        ).str.strip()
        
        # Single TF-IDF with RELAXED settings
        tfidf = TfidfVectorizer(
            stop_words="english",
            max_features=max_tfidf_features,
            ngram_range=(1, 2),    # Unigrams + bigrams
            min_df=1,              # Keep all terms (preserve distinctive features)
            max_df=0.95,           # Only filter very common terms
            sublinear_tf=True      # Use log scaling (reduce impact of term frequency)
        )
        X_text = tfidf.fit_transform(df["text"])
        
        print(f"    - TF-IDF features: {X_text.shape[1]}")
    
    # ========== CONTINUE WITH NUMERIC FEATURES (SAME FOR BOTH) ==========
    
    # ========== 7. NUMERIC FEATURES ==========
    # Fill missing values
    df["avg_rating"] = df["avg_rating"].fillna(df["avg_rating"].mean())
    df["rating_count"] = df["rating_count"].fillna(0).astype(float)
    df["rating_count_log"] = np.log1p(df["rating_count"])
    df["year"] = df["year"].fillna(df["year"].median())
    
    # Standard numeric features
    numeric_cols = ["year", "avg_rating", "rating_count_log"]
    
    # Add enriched numeric features if available
    if "vote_average" in df.columns:
        df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0)
        numeric_cols.append("vote_average")
    
    if "popularity" in df.columns:
        df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0.0)
        df["popularity_log"] = np.log1p(df["popularity"])
        numeric_cols.append("popularity_log")
    
    if "runtime" in df.columns:
        df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce").fillna(0.0)
        numeric_cols.append("runtime")
    
    # Scale numeric features
    X_num_raw = df[numeric_cols].values
    scaler = StandardScaler()
    X_num = scaler.fit_transform(X_num_raw)
    
    # ========== 8. COMBINE FEATURES ==========
    X = hstack([X_text, X_num])
    
    # ========== 9. PREPARE MOVIES_DF FOR UI ==========
    # Essential columns
    keep_cols = [
        "movieId",
        "title",
        "clean_title",
        "genres",
        "year",
        "avg_rating",
        "rating_count",
    ]
    
    # Add optional columns if available (prioritized order)
    optional_cols = [
        # IDs
        "tmdbId",
        "imdbId",
        
        # Visual
        "poster_path",
        "poster_url",
        
        # Content
        "overview",
        "tagline",
        
        # Metadata
        "release_date",
        "runtime",
        "status",
        "original_language",
        
        # Ratings
        "vote_average",
        "vote_count",
        "popularity",
        
        # People
        "director",
        "cast_top10",
        
        # Production
        "production_companies",
        "production_countries",
        "keywords",
        
        # Financial
        "budget",
        "revenue",
    ]
    
    for col in optional_cols:
        if col in df.columns:
            keep_cols.append(col)
    
    movies_df = df[keep_cols].copy()
    
    # ========== 10. GENERATE POSTER_URL IF POSTER_PATH EXISTS ==========
    if "poster_path" in movies_df.columns and "poster_url" not in movies_df.columns:
        # Convert poster_path to full URL
        movies_df["poster_url"] = movies_df["poster_path"].apply(
            lambda p: f"https://image.tmdb.org/t/p/w500{p}" if pd.notna(p) and p else None
        )
    
    print(f"✓ Built features for {len(movies_df)} movies")
    print(f"  - Total TF-IDF features: {X_text.shape[1]}")
    print(f"  - Numeric features: {len(numeric_cols)}")
    print(f"  - Grand total features: {X.shape[1]}")
    
    if "poster_url" in movies_df.columns:
        n_posters = movies_df["poster_url"].notna().sum()
        print(f"  - Movies with posters: {n_posters}/{len(movies_df)}")
    
    return movies_df, X, tfidf, scaler