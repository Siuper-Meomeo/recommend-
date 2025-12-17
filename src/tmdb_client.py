# src/tmdb_client.py
"""
Helper functions for TMDB poster URLs
"""

import pandas as pd


def get_poster_url(tmdb_id, poster_path=None):
    """
    Get TMDB poster URL from tmdbId or poster_path
    
    Args:
        tmdb_id: TMDB movie ID (from links.csv)
        poster_path: Direct poster path (from enriched data)
    
    Returns:
        Full poster URL or None
    """
    # Priority 1: Use poster_path if available (from enriched data)
    if poster_path and pd.notna(poster_path) and str(poster_path).strip():
        path = str(poster_path).strip()
        if path.startswith("/"):
            return f"https://image.tmdb.org/t/p/w500{path}"
        elif path.startswith("http"):
            return path
    
    # Priority 2: Construct from tmdbId (fallback)
    # Note: This won't give actual poster without API call
    # Just return None if no poster_path
    if tmdb_id and pd.notna(tmdb_id):
        # Could make API call here, but expensive
        # Better to just return None and let frontend handle
        pass
    
    return None