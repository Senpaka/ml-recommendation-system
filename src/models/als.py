import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix


class AlsRecommender():

    def __init__(
        self,
        factors: int = 16,
        regularization: float = 0.5,
        iterations: int = 10,
        random_state: int = 42,
        alpha: float = 1.0,
        num_threads: int = 6,
    ):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.random_state = random_state
        self.alpha = alpha
        self.num_threads = num_threads

    def fit(
        self,
        history: pd.DataFrame,
        n_users: int,
    ):
        movie_ids = history["movie_id"].unique()

        self.movie_to_als = {
            movie_id: idx
            for idx, movie_id in enumerate(movie_ids)
        }
    
        self.als_to_movie = {
            idx: movie_id
            for movie_id, idx in self.movie_to_als.items()
        }

        self.seen_movies = history.groupby("user_idx")["movie_id"].apply(set).to_dict()
    
        positive = history.loc[
            history["rating"] >= 4,
            ["user_idx", "movie_id", "rating"]
        ].copy()
    
        positive["als_movie_idx"] = positive["movie_id"].map(self.movie_to_als)
        values = (positive["rating"] - 3).values.astype(np.float32)
    
        self.user_item = csr_matrix(
            (
                values,
                (positive["user_idx"], positive["als_movie_idx"])
            ),
            shape=(n_users, len(self.movie_to_als)),
            dtype=np.float32
        )
    
        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            num_threads=self.num_threads,
            random_state=self.random_state,
            alpha=self.alpha
        )
    
        self.model.fit(self.user_item, show_progress=True)

        return self

    def recommend(
        self,
        user_idx: int,
        k: int = 500
    ):
        
        
        candidates = {}
        watched = self.seen_movies.get(user_idx, set())
        recommendations = []

        request_k = min(k + len(watched), self.user_item.shape[1])

        ids, scores = self.model.recommend(
            userid=int(user_idx),
            user_items=self.user_item[int(user_idx)],
            N=request_k,
            filter_already_liked_items=True,
        )

        for movie_idx, score in zip(ids, scores):
        
            movie_id = self.als_to_movie[int(movie_idx)]

            if movie_id in watched:
                continue

            recommendations.append({
                "user_idx": user_idx,
                "movie_id": movie_id,
                "score": float(score)
            })

            if len(recommendations) >= k:
                break

        return recommendations