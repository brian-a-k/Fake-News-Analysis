from typing import *
import pandas as pd


def clean_columns(raw_cols: List[str]) -> List[str]:
    return [name.lower().replace(' ', '_') for name in raw_cols]


def fill_null_string_columns(data: pd.DataFrame, fill_dict: dict):
    return data.fillna(value=fill_dict)


# Removes all characters that are NOT: (letters numbers . ? , !)
def clean_text_column(data: pd.DataFrame, column_name: str) -> pd.DataFrame:
    data[column_name] = data[column_name].str.replace('[^0-9a-zA-Z’ .!?,]', '', regex=True)
    return data







