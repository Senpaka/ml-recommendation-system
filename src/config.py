from dataclasses import dataclass, field
from typing import TypedDict


class ALSParams(TypedDict, total=False):
    factors: int
    regularization: float
    iterations: int
    random_state: int
    alpha: float
    num_threads: int


class SimilarityParams(TypedDict, total=False):
    k_neighbors: int
    max_sim_seeds: int


class PopularityParams(TypedDict, total=False):
    m: int


class CatBoostParams(TypedDict, total=False):
    iterations: int
    learning_rate: float
    depth: int
    l2_leaf_reg: float
    random_strength: float


@dataclass
class Config:
    als_params: ALSParams = field(
        default_factory=lambda: ALSParams(
            factors=16,
            regularization=0.5,
            iterations=10,
            random_state=42,
            alpha=1.0,
            num_threads=6,
        )
    )

    sim_params: SimilarityParams = field(
        default_factory=lambda: SimilarityParams(k_neighbors=100, max_sim_seeds=50)
    )

    pop_params: PopularityParams = field(
        default_factory=lambda: PopularityParams(m=100)
    )

    catboost_params: CatBoostParams = field(
        default_factory=lambda: CatBoostParams(
            iterations=1000,
            learning_rate=0.05,
            depth=7,
            l2_leaf_reg=4,
            random_strength=0.5,
        )
    )

    k_als: int = 500
    k_sim: int = 1100
    k_pop: int = 300
    candidate_k: int = 500

    COLUMNS_TO_DROP: list[str] = field(
        default_factory=lambda: [
            "label",
            "group_id",
            "user_id",
            "user_idx",
            "movie_id",
        ]
    )

    CAT_FEATURES: list[str] = field(
        default_factory=lambda: [
            "gender",
            "age_group",
            "occupation",
            "zip_code",
        ]
    )
