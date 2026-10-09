from src.models.als import AlsRecommender
from src.models.recommender import Recommender
from src.models.similary import SimilarityRecommender
from src.generator import CandidateGenerator

import pandas as pd



class RetrievalPipeline:

    def __init__(
        self,
        als_params: dict,
        sim_params: dict,
        pop_params: dict,
        user_encoder,
        k_als: int = 500,
        k_sim: int = 1100,
        k_pop: int = 300,
        candidate_k: int = 500
    ):
        self.als_model = AlsRecommender(**als_params)
        self.similarity_model = SimilarityRecommender(**sim_params)
        self.popularity_model = Recommender(**pop_params)

        self.user_encoder = user_encoder

        self.k_als = k_als
        self.k_sim = k_sim
        self.k_pop = k_pop
        self.candidate_k = candidate_k

        self.generator = None

    def fit(self, history: pd.DataFrame, n_users: int):

        self.als_model.fit(history=history, n_users=n_users)
        self.similarity_model.fit(history=history, n_users=n_users)
        self.popularity_model.fit(history)

        self.generator = CandidateGenerator(
            als_model=self.als_model,
            similarity_model=self.similarity_model,
            popularity_model=self.popularity_model,
            user_encoder=self.user_encoder,
            k_als=self.k_als,
            k_sim=self.k_sim,
            k_pop=self.k_pop,
            candidate_k=self.candidate_k
        )

        return self

    def make_candidates(self, users: pd.DataFrame):

        if self.generator is None:
            raise RuntimeError("RetrievalPipeline must be fitted first.")

        return self.generator.build_candidates(users=users)
        

