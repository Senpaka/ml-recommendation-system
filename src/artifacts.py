import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from catboost import CatBoostRanker

from src.config import Config
from src.retrieval import RetrievalPipeline


def save_artifacts(
    directory: str,
    ranker: CatBoostRanker,
    retrieval: RetrievalPipeline,
    users: pd.DataFrame,
    movies: pd.DataFrame,
    history: pd.DataFrame,
    feature_columns: list[str],
    config: Config,
    metrics: dict[str, int | float],
) -> None:

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

    ranker.save_model(str(path / "ranker.cbm"))
    joblib.dump(retrieval, path / "retrieval.joblib")

    users.to_parquet(path / "users.parquet", index=False)
    movies.to_parquet(path / "movies.parquet", index=False)
    history.to_parquet(path / "history.parquet", index=False)

    metadata = {
        "feature_columns": feature_columns,
        "cat_features": config.CAT_FEATURES,
        "config": asdict(config),
        "evaluation_metrics": metrics,
        "n_users": len(users),
        "n_movies": len(movies),
    }

    (path / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_artifacts(directory: str) -> dict[str, Any]:
    path = Path(directory)

    ranker = CatBoostRanker()
    ranker.load_model(str(path / "ranker.cbm"))

    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))

    return {
        "ranker": ranker,
        "retrieval": joblib.load(path / "retrieval.joblib"),
        "users": pd.read_parquet(path / "users.parquet"),
        "movies": pd.read_parquet(path / "movies.parquet"),
        "history": pd.read_parquet(path / "history.parquet"),
        "metadata": metadata,
    }
