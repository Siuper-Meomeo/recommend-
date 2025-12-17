# src/models/content_based.py
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self, movies_df, X_features):
        """
        movies_df: DataFrame có cột movieId, align với X_features theo index
        X_features: sparse matrix (TF-IDF + numeric)
        """
        self.movies_df = movies_df.reset_index(drop=True)
        self.X = csr_matrix(X_features)

        # map movieId -> row index
        self.movie_id_to_index = {
            int(mid): idx for idx, mid in enumerate(self.movies_df["movieId"].astype(int).tolist())
        }

    def recommend_similar(self, movie_id, top_k=10, min_sim=1e-9):
        """
        Recommend phim tương tự theo cosine similarity.
        - min_sim: lọc kết quả similarity quá thấp (tránh kết quả rác)
        """
        movie_id = int(movie_id)
        idx = self.movie_id_to_index.get(movie_id)
        if idx is None:
            return []

        vec = self.X[idx]
        sims = cosine_similarity(vec, self.X).ravel()

        # loại chính nó
        sims[idx] = -1.0

        # lấy top lớn hơn top_k để còn lọc min_sim
        # (tránh trường hợp filter xong còn < top_k)
        candidate_k = min(max(top_k * 5, top_k), sims.shape[0])
        top_idx = np.argpartition(-sims, kth=candidate_k - 1)[:candidate_k]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

        # lọc theo min_sim
        top_idx = [i for i in top_idx if sims[i] > min_sim][:top_k]
        if not top_idx:
            return []

        return self.movies_df.iloc[top_idx]["movieId"].astype(int).tolist()

    def get_vector(self, movie_id):
        """
        Lấy vector feature của một movie để dùng cho user-profile CBF hoặc hybrid (nếu cần).
        """
        mid = int(movie_id)
        idx = self.movie_id_to_index.get(mid)
        if idx is None:
            return None
        return self.X[idx]
