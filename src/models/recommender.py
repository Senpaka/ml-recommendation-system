import pandas as pd

class Recommender:

    def __init__(self, m:int = 100):
        self.m = m
        self.popularity = None


    def fit(self, interactions: pd.DataFrame):

        C = interactions.rating.mean()

        popularity = (
            interactions.groupby("movie_id").rating
            .agg(
                rating_count="count",
                avg_rating="mean"
            )
            .reset_index()
        )

        popularity["score"] = (
            popularity.rating_count 
            / (popularity.rating_count + self.m) 
            * popularity.avg_rating 
            + self.m 
            / (popularity.rating_count + self.m)
            * C
        )

        self.popularity = (
            popularity
            .sort_values(by="score", ascending=False)
            .reset_index(drop=True)
        )

        return self

    def recommend(self, user_id: int, interactions: pd.DataFrame, k: int = 10):
        
        watched_movies = set(
            interactions.loc[
                interactions.user_id == user_id, "movie_id"
            ]
        )

        return (
            self.popularity[
                ~self.popularity.movie_id.isin(watched_movies)
            ]
            .head(k)["movie_id"]
            .tolist()
        )