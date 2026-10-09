import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


class SimilarityRecommender:

    def __init__(
        self,
        k_neighbors: int = 100,
        max_sim_seeds: int = 50,
    ):
        self.k_neighbors = k_neighbors
        self.max_sim_seeds = max_sim_seeds

    def fit(
        self,
        history,
        n_users,
    ):
        movie_ids = history["movie_id"].unique()

        self.movie_to_sim = {
            movie_id: idx
            for idx, movie_id in enumerate(movie_ids)
        }

        self.sim_to_movie = {
            idx: movie_id
            for movie_id, idx in self.movie_to_sim.items()
        }

        positive = history.loc[
            history["rating"] >= 4,
            ["user_idx", "movie_id", "rating"]
        ].copy()

        positive["sim_movie_idx"] = positive["movie_id"].map(self.movie_to_sim)
        values = (positive["rating"] - 3).values.astype(np.float32)

        user_movie = csr_matrix(
            (
                values,
                (positive["user_idx"], positive["sim_movie_idx"])
            ),
            shape=(n_users, len(self.movie_to_sim)),
            dtype=np.float32
        )

        self.movie_user = normalize(user_movie.T.tocsr(), axis=1)

        self.model = NearestNeighbors(
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )

        self.model.fit(self.movie_user)

        n_neighbors = min(self.k_neighbors + 1, self.movie_user.shape[0])

        distances, indices = self.model.kneighbors(
            self.movie_user,
            n_neighbors=n_neighbors,
        )

        self.neighbors = {}

        for movie_idx in range(self.movie_user.shape[0]):

            mask = indices[movie_idx] != movie_idx

            self.neighbors[movie_idx] = (
                indices[movie_idx][mask][:self.k_neighbors],
                (1.0 - distances[movie_idx][mask])[:self.k_neighbors]
            )

        self.liked_recent = (
            history.loc[history["rating"] >= 4]
            .sort_values(["user_idx", "timestamp"])
            .groupby("user_idx")
            .tail(self.max_sim_seeds)
            .groupby("user_idx")["movie_id"]
            .apply(list)
            .to_dict()
        )

        self.seen_movies = (
            history.groupby("user_idx")["movie_id"]
            .apply(set)
            .to_dict()
        )

        return self

    def recommend(
        self,
        user_idx: int,
        k: int = 500
    ):
        watched = self.seen_movies.get(user_idx, set())
        
        sim_scores = {}

        
        for seed_movie_id in self.liked_recent.get(user_idx, []):
            seed_idx = self.movie_to_sim.get(seed_movie_id)

            if seed_idx is None:
                continue

            neighbor_ids, neighbor_scores = self.neighbors[seed_idx]

            for neighbor_idx, score in zip(neighbor_ids, neighbor_scores):
                movie_id = self.sim_to_movie[neighbor_idx]

                if movie_id in watched:
                    continue

                sim_scores[movie_id] = max(sim_scores.get(movie_id, 0), score)

        sim_top = sorted(sim_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        return [
            {
                "user_idx": user_idx,
                "movie_id": movie_id,
                "score": float(score)
            }
            for movie_id, score in sim_top
        ]