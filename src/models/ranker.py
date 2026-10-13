import pandas as pd

from catboost import CatBoostRanker, Pool


class Ranker:
    """
    Модель ранжирования кандидатов на градиентном бустинге CatBoost

    Занимает стадию reranking в рекомендательном пайплайне: 
    получает признаки и вычисляет скоры релевантности кандидатов

    Инкапсулирует `catboost.CatBoostRanker` и `catboost.Pool`

    Attributes:
        model (CatBoostRanker): Внутренняя модель CatBoost, 
        создается в `__init__` и обучается в `fit`
    """

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
        """
        Создает необученную модель ранжирования.

        Все аргументы прокидываются напрямую в `catboost.CatBoostRanker`
        поэтому их семантика совпадает с CatBoost.

        Args:
            iterations (int): Максимальное число деревьев
            learning_rate (float): Скорость обучения (шаг обучения)
            depth (int): Глубина деревьев
            l2_leaf_reg (float): Коэффициент L2-регуляризации
            random_strength (float): Сила случайности при выборе сплитов
            loss_function (str): Функция потерь. По умолчанию `YetiRank`
            eval_metric (str): Метрика для валидации. По умолчанию `NDCG:top=10`
            task_type (str): `GPU` или `CPU`
            random_seed (int): Seed для воспроизводимости модели
            verbose (int, bool): Частота логирования. `False` - отключить,
                `True` - Каждую итерацию, `int` - раз в N итераций
        """
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
        """
        Обучает модель ранжирования на размеченных данных

        Обарачивает входные данные в `catboost.Pool` 
        и вызывает `CatBoostRanker.fit`. Если передан 
        `eval_data`, строит отдельный `Pool` для валидации 

        Args:
            X (pd.DataFrame): Матрица признаков
            y (pd.Series): Метки реливантности
            group_id: (pd.Series): Индетификатор группы. 
                Строки с одинаковым `group_id` относятся к одному запросу
            cat_features (list[str]): Именования категориальный колонок в `X`
            eval_data (tuple | None): Кортеж вида `(X_val, y_val, group_val)`
                Используется для валидации. Если `None` валидация не выполняется
            early_stopping_rounds (int | None): Число итераций без улучшения `eval_metric`,
                после которых обучение останавливается
                
        Returns:
            Ranker: Обученный экземпляр модели.
        """

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
        """
        Возвращает скоры релевантности 

        Args: 
            X (pd.DataFrame): Матрица признаком, с тем же набором колонок, 
                что использовался при `fit`
        
        Returns:
            np.ndarray: Одномерный массив той же длинны, что и `X`.
                Чем выше, тем релевантнее пара
        """
        return self.model.predict(X)

    def get_feature_importance(self):
        """
        Возращает важности признаков.

        Returns:
            np.ndarray: Массив содержащий важности признаков
        """
        return self.model.get_feature_importance()

    def get_best_iteration(self):
        """
        Возвращает номер лучшей итерации по `eval_data`

        Returns:
            int: Индекс итерации, на которой метрика на валидации была лучшей
        """
        return self.model.get_best_iteration()

    def get_best_score(self):
        """
        Возвращает лучшие значения метрик по `eval_data`.

        Returns:
            dict: Словарь вида `{"validation": {"NDCG:top=10": 0.12, ...}}`
        """
        return self.model.get_best_score()