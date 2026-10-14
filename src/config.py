from dataclasses import dataclass, field


@dataclass
class Config:

    als_params: dict = field(default_factory=lambda: {
        "factors": 16,
        "regularization": 0.5,
        "iterations": 10,
        "random_state": 42,
        "alpha": 1.0,
        "num_threads": 6,
    })

    sim_params: dict = field(default_factory=lambda: {
        "k_neighbors": 100,
        "max_sim_seeds": 50,
    })

    pop_params: dict = field(default_factory=lambda: {
        "m": 100,
    })

    catboost_params: dict = field(default_factory=lambda: {
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 7,
        "l2_leaf_reg": 4,
        "random_strength": 0.5,
    })

    k_als: int = 500
    k_sim: int = 1100
    k_pop: int = 300
    candidate_k: int = 500

    COLUMNS_TO_DROP: list[str] = field(default_factory=lambda: [
        "label",
        "group_id",
        "user_id",
        "user_idx",
        "movie_id",
    ])

    CAT_FEATURES: list[str] = field(default_factory=lambda: [
        "gender",
        "age_group",
        "occupation",
        "zip_code",
    ])