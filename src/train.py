import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

from src.config import Config
from src.data.preprocessing import preprocess_data
from src.data.temporal_split import TemporalSplit
from src.features.features import add_labels, build_features
from src.metrics.metrics import ndcg_at_k, recall_at_k
from src.retrieval import RetrievalPipeline


def train(
    users: pd.DataFrame, movies: pd.DataFrame, interactions: pd.DataFrame
) -> tuple[CatBoostRanker, dict[str, int | float]]:

    config = Config()
    splitter = TemporalSplit()

    users_df, movies_df, interactions_df = preprocess_data(
        raw_users=users,
        raw_movies=movies,
        raw_interactions=interactions,
    )

    train_df, test_target = splitter.split_to_two(interactions_df)
    train_history, train_target = splitter.split_to_two(train_df)

    train_rank_df = _build_df(
        history=train_history,
        target=train_target,
        users=users_df,
        movies=movies_df,
        config=config,
        query_prefix="train",
    )

    X_train = train_rank_df.drop(columns=config.COLUMNS_TO_DROP)
    feature_columns = X_train.columns.tolist()

    train_pool = Pool(
        data=X_train,
        label=train_rank_df["label"],
        group_id=train_rank_df["group_id"],
        cat_features=config.CAT_FEATURES,
    )

    model_params = {
        "loss_function": "YetiRank",
        "eval_metric": "NDCG:top=10",
        "random_seed": 42,
        "thread_count": 6,
        "verbose": 100,
        "task_type": "GPU",
        **config.catboost_params,
    }

    model = CatBoostRanker(**model_params)
    model.fit(train_pool, use_best_model=False)

    test_rank_df = _build_df(
        history=train_df,
        target=test_target,
        users=users_df,
        movies=movies_df,
        config=config,
        query_prefix="test",
    )

    metrics = _evaluate_ranker(
        model=model,
        data=test_rank_df,
        target=test_target,
        feature_columns=feature_columns,
        cat_features=config.CAT_FEATURES,
        k=10,
    )

    return model, metrics


def _build_df(
    history: pd.DataFrame,
    target: pd.DataFrame,
    users: pd.DataFrame,
    movies: pd.DataFrame,
    config: Config,
    query_prefix: str,
) -> pd.DataFrame:

    retrieval = RetrievalPipeline(
        als_params=config.als_params,
        sim_params=config.sim_params,
        pop_params=config.pop_params,
        k_als=config.k_als,
        k_sim=config.k_sim,
        k_pop=config.k_pop,
        candidate_k=config.candidate_k,
    )

    retrieval.fit(history)

    target_users = target["user_id"].unique()
    unknown_users = pd.Index(target_users).difference(history["user_id"].unique())

    if len(unknown_users):
        raise ValueError(
            f"Cold-start handing is required for users:{unknown_users[:5].tolist()}"
        )

    candidates = retrieval.make_candidates(target_users)

    if candidates.empty:
        raise ValueError(f"No candidates generated for {query_prefix}")

    data = build_features(
        candidates=candidates,
        history=history,
        df_users=users,
        df_movies=movies,
        query_prefix=query_prefix,
    )

    data = add_labels(data=data, target=target)

    return data.sort_values(["group_id", "movie_id"]).reset_index(drop=True)


def _evaluate_ranker(
    model: CatBoostRanker,
    data: pd.DataFrame,
    target: pd.DataFrame,
    feature_columns: list[str],
    cat_features: list[str],
    k: int = 10,
) -> dict[str, int | float]:

    if k <= 0:
        raise ValueError("k must be positive")

    X = data.loc[:, feature_columns]

    predictions_pool = Pool(data=X, cat_features=cat_features)

    scored = data[["user_id", "movie_id"]].copy()
    scored["score"] = model.predict(predictions_pool)

    predictions = (
        scored.sort_values(["user_id", "score"], ascending=[True, False])
        .groupby("user_id")["movie_id"]
        .apply(lambda x: x.head(k).to_numpy())
        .to_dict()
    )

    relevant = (
        target.loc[target["rating"] >= 4]
        .groupby("user_id")["movie_id"]
        .apply(set)
        .to_dict()
    )

    if not relevant:
        raise ValueError("No positive interactions in the evaluation target")

    recalls = []
    ndcgs = []

    for user_id, y_true in relevant.items():
        y_pred = predictions.get(user_id, [])

        recalls.append(recall_at_k(y_true, y_pred, k=k))
        ndcgs.append(ndcg_at_k(y_true, y_pred, k=k))

    return {
        f"recall@{k}": float(np.mean(recalls)),
        f"ndcg@{k}": float(np.mean(ndcgs)),
        "evaluated_users": len(ndcgs),
    }


if __name__ == "__main__":
    from pathlib import Path

    DATA_DIR = Path(__file__).resolve().parent.parent / "datasets"

    df_movies = pd.read_csv(
        DATA_DIR / "movies.dat",
        sep="::",
        engine="python",
        encoding="Latin-1",
        names=["movie_id", "title", "genre"],
    )

    df_rating = pd.read_csv(
        DATA_DIR / "ratings.dat",
        sep="::",
        engine="python",
        encoding="Latin-1",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )

    df_users = pd.read_csv(
        DATA_DIR / "users.dat",
        sep="::",
        engine="python",
        encoding="Latin-1",
        names=["user_id", "gender", "age_group", "occupation", "zip_code"],
    )

    model, metrics = train(df_users, df_movies, df_rating)

    print(metrics)
