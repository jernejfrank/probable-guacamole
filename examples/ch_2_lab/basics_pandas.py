"""Collecting functions for getting familiar with pandas basics."""

import pandas as pd
from hamilton.function_modifiers import (
    config,
    extract_columns,
    pipe_output,
    step,
    value,
)


@extract_columns(
    "mpg",
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "year",
    "origin",
    "name",
)
@config.when(state="raw")
def auto__raw() -> pd.DataFrame:
    return pd.read_csv("../../datasets/Auto.csv")


def _replace_invalids_with_nan(
    df: pd.DataFrame, invalid_symbol: str = "?"
) -> pd.DataFrame:
    return df.replace({invalid_symbol: pd.NA})


def _drop_NA(df: pd.DataFrame) -> pd.DataFrame:  # noqa N802
    return df.dropna()


def _set_index(df: pd.DataFrame, name: str) -> pd.DataFrame:
    return df.set_index(name)


def _convert_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series)


@pipe_output(step(_convert_to_numeric).on_output("horsepower"))
@extract_columns(
    "mpg",
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "year",
    "origin",
    "name",
)
@pipe_output(
    step(_replace_invalids_with_nan, invalid_symbol=value("?")),
    step(_drop_NA),
    step(_set_index, name=value("name")),
)
@config.when(state="cleaned")
def auto__cleaned() -> pd.DataFrame:
    return pd.read_csv("../../datasets/Auto.csv")


def unique_horsepower(horsepower: pd.Series) -> pd.Series:
    return horsepower.unique()  # type: ignore


def total_horsepower(horsepower: pd.Series) -> float:
    return horsepower.sum()


def year_80(auto: pd.DataFrame, year: pd.Series) -> pd.DataFrame:
    return auto[year == 80]


def amc_and_ford(auto: pd.DataFrame) -> pd.DataFrame:
    rows = ["amc rebel sst", "ford torino"]
    return auto.loc[rows]


def row_3_4(auto: pd.DataFrame) -> pd.DataFrame:
    return auto.iloc[[3, 4]]


def col_0_2_3(auto: pd.DataFrame) -> pd.DataFrame:
    return auto.iloc[:, [0, 2, 3]]


def row_3_4_col_0_2_3(auto: pd.DataFrame) -> pd.DataFrame:
    return auto.iloc[[3, 4], [0, 2, 3]]


def ford_mpg_origin(auto: pd.DataFrame) -> pd.DataFrame:
    return auto.loc[["ford galaxie 500"], ["mpg", "origin"]]


def weight_and_origin_more_than_80_years(
    auto: pd.DataFrame, year: pd.Series
) -> pd.DataFrame:
    return auto.loc[year > 80, ["weight", "origin"]]


def ford_datsun_displacement_less_300(
    auto: pd.DataFrame, displacement: pd.Series
) -> pd.DataFrame:
    return auto.loc[
        (displacement < 300)
        & (auto.index.str.contains("ford") | auto.index.str.contains("datsun")),
        ["weight", "origin"],
    ]


if __name__ == "__main__":
    from hamilton import driver

    import __main__

    conf = {"state": "cleaned"}

    dr = driver.Builder().with_modules(__main__).with_config(conf).build()

    dr.display_all_functions(output_file_path="./examples/ch_2_lab/DAG.png")
