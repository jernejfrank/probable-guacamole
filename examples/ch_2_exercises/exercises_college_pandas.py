"""Solving exercises involving the College dataset."""

import pandas as pd
from beartype.typing import Dict
from hamilton.function_modifiers import extract_columns, pipe_input, step, value


def college_raw() -> pd.DataFrame:
    return pd.read_csv("../../datasets/College.csv")


def college_summary(college_raw: pd.DataFrame) -> pd.DataFrame:
    return college_raw.describe()


def _rename_col(df: pd.DataFrame, rename_map: Dict[str, str]) -> pd.DataFrame:
    return df.rename(columns=rename_map)


def _set_index(df: pd.DataFrame, index_col: str) -> pd.DataFrame:
    return df.set_index(index_col)


@extract_columns("Top10perc", "Apps", "Enroll", "Outstate", "Private")
@pipe_input(
    step(_rename_col, rename_map=value({"Unnamed: 0": "College"})),
    step(_set_index, index_col=value("College")),
)
def college(college_raw: pd.DataFrame) -> pd.DataFrame:
    return college_raw


def scatter_matrix_top10_apps_enroll(
    Top10perc: pd.Series, Apps: pd.Series, Enroll: pd.Series
) -> pd.DataFrame:
    return pd.concat([Top10perc, Apps, Enroll], axis=1)


def elite(Top10perc: pd.Series) -> pd.Series:
    return Top10perc > 50
