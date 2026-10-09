class Recommender:

    def __init__(self, m:int = 100):
        self.m = m
        self.popularity = None


    def fit(self, interactions):

        C = interactions.rating.mean()

        popularity = (
            interactions
            .groupby("movie_id")
            .rating
            .agg(
                rating_count="count",
                avg_rating="mean"
            )
            .reset_index()
        )

        popularity["pop_score"] = (
            popularity.rating_count
            /
            (popularity.rating_count + self.m)
            *
            popularity.avg_rating
            +
            self.m
            /
            (popularity.rating_count + self.m)
            *
            C
        )


        self.popularity = (
            popularity
            .sort_values(
                "pop_score",
                ascending=False
            )
            .reset_index(drop=True)
        )


        self.user_watched = (
            interactions
            .groupby("user_idx")["movie_id"]
            .apply(set)
            .to_dict()
        )

        return self



    def recommend(
        self,
        user_idx,
        exclude_movies=None,
        k=10
    ):

        watched_movies = (
            exclude_movies.get(user_idx, set())
            if exclude_movies
            else self.user_watched.get(user_idx, set())
        )

        top = self.popularity[
            ~self.popularity.movie_id.isin(
                watched_movies
            )
            .head(k)
        ]

        return [
            [row.movie_id, row.score]
            for row in top.itertuples(index=False)
        ]
