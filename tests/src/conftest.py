import pandas as pd
import pytest


@pytest.fixture
def history():
    return pd.DataFrame(
        {
            "user_id": [20, 100, 20, 7],
            "movie_id": [500, 900, 900, 300],
            "rating": [5, 4, 2, 5],
            "timestamp": [1, 2, 3, 4],
        }
    )


@pytest.fixture
def popularity_history():
    return pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1, 1],
            "movie_id": [100, 200, 100, 300, 400],
            "rating": [5, 2, 4, 5, 3],
        }
    )


@pytest.fixture
def small_history():
    return pd.DataFrame(
        {
            "user_idx": [],
            "movie_id": [],
            "rating": [],
        }
    )


@pytest.fixture
def fast_config():
    from src.config import Config

    config = Config()
    config.als_params.update(factors=3, iterations=2, num_threads=1)
    config.sim_params.update(k_neighbors=8, max_sim_seeds=3)
    config.catboost_params.update(
        iterations=3,
        depth=2,
        task_type="CPU",
        thread_count=1,
        verbose=False,
        allow_writing_files=False,
    )
    config.k_als = 8
    config.k_sim = 8
    config.k_pop = 8
    config.candidate_k = 10
    return config


@pytest.fixture
def raw_dataset():
    users = pd.DataFrame(
        {
            "user_id": [20 + 17 * i for i in range(8)],
            "gender": ["F", "M"] * 4,
            "age_group": [25] * 8,
            "occupation": list(range(8)),
            "zip_code": ["00123"] * 8,
        }
    )
    movies = pd.DataFrame(
        {
            "movie_id": [500 + 13 * i for i in range(24)],
            "title": [f"Film {i} (2000)" for i in range(24)],
            "genre": ["Action|Comedy", "Drama"] * 12,
        }
    )
    rows = []
    for i, user_id in enumerate(users["user_id"]):
        for t in range(16):
            rows.append(
                {
                    "user_id": user_id,
                    "movie_id": 500 + 13 * ((i * 3 + t) % 24),
                    "rating": [5, 4, 2, 5][t % 4],
                    "timestamp": 1000 + t,
                }
            )
    return users, movies, pd.DataFrame(rows)


@pytest.fixture
def fitted_retrieval(raw_dataset, fast_config):
    from src.retrieval import RetrievalPipeline

    pipeline = RetrievalPipeline(
        als_params=fast_config.als_params,
        sim_params=fast_config.sim_params,
        pop_params=fast_config.pop_params,
        k_als=fast_config.k_als,
        k_sim=fast_config.k_sim,
        k_pop=fast_config.k_pop,
        candidate_k=fast_config.candidate_k,
    )
    return pipeline.fit(raw_dataset[2])
