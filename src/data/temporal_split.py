import pandas as pd


class TemporalSplit:
    """
    Разделяет взаимодействия по времени отдельно для каждого пользователя.

    Сортирует данные по user_id и timestamp, затем разделяет историю
    каждого пользователя по количеству строк. Более ранние взаимодействия
    попадают в обучающую часть, более поздние — в проверочные.

    Общая временная граница для всех пользователей не устанавливается.
    Взаимодействия с одинаковым timestamp могут попасть в разные части.
    """

    def split_to_three(
        self, data: pd.DataFrame, train_size: float = 0.8
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Разделяет историю каждого пользователя на train, validation и test.

        В train попадают первые int(n * train_size) взаимодействий,
        в validation — следующие int(n * (1 - train_size) / 2),
        в test — оставшиеся, где n — число взаимодействий пользователя.

        Из-за округления validation и test могут различаться по размеру.
        Для коротких историй отдельные части могут оказаться пустыми.

        Args:
            data (pd.DataFrame): Непустая таблица взаимодействий
                с колонками user_id и timestamp.
            train_size (float): Доля обучающей части для каждого пользователя.
                Должна находиться строго между 0 и 1. По умолчанию 0.8.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                Таблицы (train, validation, test) с исходными колонками
                и сброшенными индексами.

        Raises:
            ValueError: Если train_size не находится строго между 0 и 1
                или входная таблица пуста.
        """
        if not 0 < train_size < 1:
            raise ValueError("train_size must be between 0 and 1")

        train_part = []
        test_part = []
        val_part = []

        df = data.copy().sort_values(by=["user_id", "timestamp"])

        for _, group in df.groupby("user_id"):
            split_index = int(len(group) * train_size)
            val_test_split_index = split_index + int(len(group) * (1 - train_size) / 2)

            train_part.append(group.iloc[:split_index])

            val_part.append(group.iloc[split_index:val_test_split_index])

            test_part.append(group.iloc[val_test_split_index:])

        train_df = pd.concat(train_part).reset_index(drop=True)
        test_df = pd.concat(test_part).reset_index(drop=True)
        val_df = pd.concat(val_part).reset_index(drop=True)

        return train_df, val_df, test_df

    def split_to_two(
        self, data: pd.DataFrame, train_size: float = 0.8
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Разделяет историю каждого пользователя на две последовательные части.

        В первую часть попадают первые int(n * train_size) взаимодействий,
        во вторую — оставшиеся, где n — число взаимодействий пользователя.
        Части можно использовать как историю и целевой период.

        Для коротких историй первая часть может оказаться пустой.

        Args:
            data (pd.DataFrame): Непустая таблица взаимодействий
                с колонками user_id и timestamp.
            train_size (float): Доля первой части для каждого пользователя.
                Должна находиться строго между 0 и 1. По умолчанию 0.8.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Таблицы (train, test)
                с исходными колонками и сброшенными индексами.

        Raises:
            ValueError: Если train_size не находится строго между 0 и 1
                или входная таблица пуста.
        """
        if not 0 < train_size < 1:
            raise ValueError("train_size must be between 0 and 1")

        train_part = []
        test_part = []

        df = data.copy().sort_values(by=["user_id", "timestamp"])

        for _, group in df.groupby("user_id"):
            split_index = int(len(group) * train_size)

            train_part.append(group.iloc[:split_index])

            test_part.append(group.iloc[split_index:])

        train_df = pd.concat(train_part).reset_index(drop=True)
        test_df = pd.concat(test_part).reset_index(drop=True)

        return train_df, test_df
