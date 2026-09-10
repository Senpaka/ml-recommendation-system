

import pandas as pd

class TemporalSplit:

    def split(
        self, 
        data: pd.DataFrame,
        test_size: float = 0.8
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")

        train_part = []
        test_part = []

        df = data.copy().sort_values(by=["user_id", "timestamp"])

        for _, group in df.groupby("user_id"):
            
            split_index = int(len(group) * (1 - test_size))

            train_part.append(
                group.iloc[:split_index]
            )

            test_part.append(
                group.iloc[split_index:]
            )

        train_df = pd.concat(train_part).reset_index(drop=True)
        test_df = pd.concat(test_part).reset_index(drop=True)
    
        return train_df, test_df