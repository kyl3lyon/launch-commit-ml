# --- Imports ---
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional

# --- Functions ---
def label_target_variable(df: pd.DataFrame, scrub_column: str, delay_column: str, target_column: str) -> pd.DataFrame:
    """
    Labels the target variable based on scrub and delay columns.

    This function creates a new column (target_column) in the DataFrame
    that represents the target variable. The target is labeled as:
    - 2 if there was a scrub (scrub_column == 1)
    - 1 if there was a delay (delay_column == 1)
    - 0 if both scrub_column and delay_column are NaN.

    Args:
        df (pd.DataFrame): The input DataFrame.
        scrub_column (str): The name of the column containing scrub information.
        delay_column (str): The name of the column containing delay information.
        target_column (str): The name of the new column to be created for the target variable.

    Returns:
        pd.DataFrame: The input DataFrame with the new target column added.
    """
    df[target_column] = np.where(
        df[scrub_column] == 1, 2,  # Class 2 for scrub (worse condition)
        np.where(df[delay_column] == 1, 1, 0)  # Class 1 for delay, 0 for nothing
    )
    return df


def label_weather_condition(df: pd.DataFrame, 
                            code_column: str, 
                            condition_codes: List[int], 
                            new_column_name: str) -> pd.DataFrame:
    """
    Labels weather conditions in a pandas DataFrame based on WMO Weather codes.
    
    Args:
    df (pd.DataFrame): The input DataFrame containing weather data.
    code_column (str): The name of the column containing WMO Weather codes.
    condition_codes (List[int]): A list of WMO Weather codes to label.
    new_column_name (str): The name of the new column to be created with labels.
    
    Returns:
    pd.DataFrame: The input DataFrame with a new column containing labels.
    
    Example:
    >>> df = pd.DataFrame({'weather_code': [0, 45, 95, 99, 61]})
    >>> df = label_weather_condition(df, 'weather_code', [95, 96, 99], 'is_thunderstorm')
    >>> print(df)
       weather_code  is_thunderstorm
    0             0                0
    1            45                0
    2            95                1
    3            99                1
    4            61                0
    """
    # WMO Weather interpretation codes dictionary
    WMO_CODES: Dict[Union[int, str], str] = {
        0: "Clear sky",
        1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 48: "Depositing rime fog",
        51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
        56: "Light freezing drizzle", 57: "Dense freezing drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        66: "Light freezing rain", 67: "Heavy freezing rain",
        71: "Slight snow fall", 73: "Moderate snow fall", 75: "Heavy snow fall",
        77: "Snow grains",
        80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
        85: "Slight snow showers", 86: "Heavy snow showers",
        95: "Thunderstorm: Slight or moderate",
        96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
    }

    # Create a new column with 0s
    df[new_column_name] = 0
    
    # Label the specified condition codes as 1
    df.loc[df[code_column].isin(condition_codes), new_column_name] = 1
    
    return df