import numpy as np


class FactorizationRecommender:

    def __init__(
        self,
        n_users,
        n_movies,
        n_factor, 
        lr, 
        reg, 
        epochs=20, 
        random_state=42

    ):
        self.n_users = n_users
        self.n_movies = n_movies
        self.n_factor = n_factor
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.random_state = random_state

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

                u = row.user_idx
                m = row.movie_idx
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

        return self
    
    def recommend(self, df, user_idx, k=10):

        watched = (
            df.loc[
                df.user_idx == user_idx,
                "movie_idx"
            ]
            .unique()
        )

        scores = (
            self.mu 
            + self.bu[user_idx] 
            + self.bi 
            + self.P[user_idx] @ self.Q.T
        )

        scores[watched] = -np.inf

        recommended = np.argsort(scores)[::-1][:k]

        return [
            (movie_idx, scores[movie_idx])
            for movie_idx in recommended
        ]
