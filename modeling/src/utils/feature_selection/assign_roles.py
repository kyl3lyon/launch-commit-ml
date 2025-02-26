# --- Imports ---
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.model_selection import train_test_split

# --- Functions ---
def assign_modeling_roles(
    df: pd.DataFrame,
    target_column: str,
    predictor_columns: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Assign modeling roles to variables in a dataset and perform train-test split.

    This function takes a DataFrame, assigns the target variable and predictor variables,
    and then splits the data into training and testing sets.

    Args:
        df (pd.DataFrame): The input DataFrame containing all variables.
        target_column (str): The name of the column to be used as the target variable.
        predictor_columns (Optional[List[str]]): A list of column names to be used as predictor variables.
            If None, all columns except the target column will be used as predictors.
        test_size (float): The proportion of the dataset to include in the test split. Default is 0.2.
        random_state (int): Controls the shuffling applied to the data before applying the split. Default is 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing:
            - X_train: Training data features
            - X_test: Testing data features
            - y_train: Training data target
            - y_test: Testing data target

    Raises:
        ValueError: If the target_column is not in the DataFrame or if any of the predictor_columns are not in the DataFrame.
    """
    # Validate target column
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the DataFrame.")

    # Assign predictor columns
    if predictor_columns is None:
        predictor_columns = [col for col in df.columns if col != target_column]
    else:
        # Validate predictor columns
        missing_columns = set(predictor_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"The following predictor columns are not in the DataFrame: {missing_columns}")

    # Separate features (X) and target (y)
    X = df[predictor_columns]
    y = df[target_column]

    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test
