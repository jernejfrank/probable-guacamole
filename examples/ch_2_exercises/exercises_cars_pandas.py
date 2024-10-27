"""Solving the exercises involving the Cars dataset."""

import numpy as np
import pandas as pd
from hamilton.function_modifiers import (
    mutate,
    value,
)


def auto() -> pd.DataFrame:
    return pd.read_csv("../../datasets/Auto.csv")


@mutate(auto, invalid_symbol=value("?"))
def replace_invalids_with_nan(
    df: pd.DataFrame, invalid_symbol: str = "?"
) -> pd.DataFrame:
    return df.replace({invalid_symbol: pd.NA})


@mutate(auto)
def drop_NA(df: pd.DataFrame) -> pd.DataFrame:  # noqa N802
    return df.dropna()


@mutate(auto)
def reset_index(df: pd.DataFrame) -> pd.DataFrame:
    return df.reset_index()


@mutate(auto)
def convert_horsepower_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df["horsepower"] = pd.to_numeric(df["horsepower"])
    return df


def auto_summary(auto: pd.DataFrame) -> pd.DataFrame:
    return auto.describe()


def auto_reduced_summary(auto: pd.DataFrame) -> pd.DataFrame:
    return auto.drop(list(np.arange(10, 85))).describe()
