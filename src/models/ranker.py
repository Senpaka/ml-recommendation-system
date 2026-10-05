import pandas as pd

from catboost import CatBoostRanker, Pool


class Ranker:

    def __init__(
        self,
        iterations: int = 2000,
        learning_rate: float = 0.05,
        depth: int = 7,
        l2_leaf_reg: float = 4.0,
        random_strength: float = 0.5,
        loss_function: str = "YetiRank",
        eval_metric: str = "NDCG:top=10",
        task_type: str = "GPU",
        random_seed: int = 42,
        verbose: int | bool = 200,
    ):
        self.model = CatBoostRanker(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            random_strength=random_strength,
            loss_function=loss_function,
            eval_metric=eval_metric,
            task_type=task_type,
            random_seed=random_seed,
            verbose=verbose,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        group_id: pd.Series,
        cat_features: list[str],
        eval_data: tuple | None = None,
        early_stopping_rounds: int | None = None,
    ):
        train_pool = Pool(
            data=X,
            label=y,
            group_id=group_id,
            cat_features=cat_features,
        )

        eval_pool = None

        if eval_data is not None:
            X_val, y_val, group_val = eval_data

            eval_pool = Pool(
                data=X_val,
                label=y_val,
                group_id=group_val,
                cat_features=cat_features,
            )

        self.model.fit(
            train_pool,
            eval_set=eval_pool,
            early_stopping_rounds=early_stopping_rounds,
        )

        return self

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)

    def get_feature_importance(self):
        return self.model.get_feature_importance()

    def get_best_iteration(self):
        return self.model.get_best_iteration()

    def get_best_score(self):
        return self.model.get_best_score()