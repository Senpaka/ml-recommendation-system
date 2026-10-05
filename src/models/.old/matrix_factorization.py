import numpy as np


class FactorizationRecommender:

    def __init__(
        self,
        n_users,
        n_movies,
        n_factor, 
        lr, 
        reg,
        user_encoder,
        movie_encoder, 
        epochs=20, 
        random_state=42,
        
    ):
        self.n_users = n_users
        self.n_movies = n_movies
        self.n_factor = n_factor
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.random_state = random_state
        self.movie_encoder = movie_encoder
        self.user_encoder = user_encoder

        self.movie_id_to_idx = {
            movie_id: idx
            for idx, movie_id in enumerate(self.movie_encoder.classes_)
        }

        self.idx_to_movie_id = {
            idx: movie_id
            for idx, movie_id in enumerate(self.movie_encoder.classes_)
        }

        self.idx_to_user_id = {
            idx: user_id
            for idx, user_id in enumerate(self.user_encoder.classes_)
        }

        self.user_id_to_idx = {
            user_id: idx
            for idx, user_id in enumerate(self.user_encoder.classes_)
        }

    def fit(
        self,
        train, 
        show=False, 
    ):

        rng = np.random.default_rng(self.random_state)

        P = rng.normal(
            0,
            0.1,
            size=(self.n_users, self.n_factor),
        )

        Q = rng.normal(
            0,
            0.1,
            size=(self.n_movies, self.n_factor),
        )

        mu = train.rating.mean()

        bu = np.zeros(self.n_users)
        bi = np.zeros(self.n_movies)

        for epoch in range(self.epochs):
            errors = []

            for row in train.itertuples(index=False):

                u = self.user_id_to_idx[row.user_id]
                m = self.movie_id_to_idx[row.movie_id]
                r = row.rating

                prediction = (
                    mu + bu[u] + bi[m] + P[u] @ Q[m]
                )

                error = r - prediction

                errors.append(error)

                old_p = P[u].copy()

                bu[u] += self.lr * (
                    error - self.reg * bu[u]
                )

                bi[m] += self.lr * (
                    error - self.reg * bi[m]
                )

                P[u] += self.lr * (
                    error * Q[m] - self.reg * P[u]
                )

                Q[m] += self.lr * (
                    error * old_p - self.reg * Q[m]
                )

            if show:
                rmse = np.sqrt(
                    np.mean(
                        np.square(errors)
                    )
                )
                print(
                    f"Epoch {epoch + 1}: "
                    f"RMSE {rmse:.4}: "
                )

        self.P = P
        self.Q = Q
        self.mu = mu
        self.bu = bu
        self.bi = bi

        self.user_watched = (
            train
            .groupby("user_id")["movie_id"]
            .apply(set)
            .to_dict()
        )

        return self
    
    def recommend(self, user_id, exclude_movies: dict[int, set[int]] | None = None, k=10):

        user_idx = self.user_encoder.transform([user_id])[0]

        scores = (
            self.mu 
            + self.bu[user_idx] 
            + self.bi 
            + self.P[user_idx] @ self.Q.T
        )

        if exclude_movies is not None:
            watched_ids = exclude_movies.get(user_id, set())
        else:
            watched_ids = self.user_watched.get(user_id, set())
 
        watched_idx = [
            self.movie_id_to_idx[movie_id]
            for movie_id in watched_ids
            if movie_id in self.movie_id_to_idx
        ]

        scores[watched_idx] = -np.inf

        recommended_idx = np.argsort(scores)[::-1][:k]

        return [
            (
                self.idx_to_movie_id[movie_idx],
                scores[movie_idx]
            )
            for movie_idx in recommended_idx
        ]
