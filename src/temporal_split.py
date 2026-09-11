

import pandas as pd

class TemporalSplit:

    def split(
        self, 
        data: pd.DataFrame,
        train_size: float = 0.8
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        if not 0 < train_size < 1:
            raise ValueError("train_size must be between 0 and 1")

        train_part = []
        test_part = []
        val_part = []

        df = data.copy().sort_values(by=["user_id", "timestamp"])

        for _, group in df.groupby("user_id"):
            
            split_index = int(len(group) * train_size)
            val_test_split_index = split_index + int(len(group) * (1 - train_size))

            train_part.append(
                group.iloc[:split_index]
            )

            val_part.append(
                group.iloc[split_index:val_test_split_index]
            )

            test_part.append(
                group.iloc[val_test_split_index:]
            )

        train_df = pd.concat(train_part).reset_index(drop=True)
        test_df = pd.concat(test_part).reset_index(drop=True)
        val_df = pd.concat(val_part).reset_index(drop=True)
    
        return train_df, val_df, test_df