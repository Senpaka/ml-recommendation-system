import pandas as pd

from catboost import CatBoostRanker


class Ranker:


    def __init__(
        self,
        iterations,
        learning_rate,
        depth,
        random_seed: int =42,
        verbose: int | bool = False,
        loss_function: str = "YetiRank",
        eval_metric: str = "NDCG:top=10",
    ):
        self.model = CatBoostRanker(
                loss_function=loss_function,
                eval_metric=eval_metric,
                iterations=iterations,
                learning_rate=learning_rate,
                depth=depth,
                random_seed=random_seed,
                verbose=verbose
            )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        group: pd.Series,
        features: list[str]
    ):

        self.model.fit(
            X,
            y,
            group_id=group,
            cat_features=features
        )

        return self


    def predict(
        self, 
        candidates: pd.DataFrame
    ):
        return self.model.predict(candidates)