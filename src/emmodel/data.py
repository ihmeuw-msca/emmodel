"""
Data module
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Callable

import numpy as np
import pandas as pd


@dataclass
class DataProcessor:
    col_deaths: str
    col_year: str
    col_dtime: str
    col_covs: List[str] = field(default_factory=list)
    dtime_unit: str = "week"

    def __post_init__(self):
        if self.dtime_unit not in ["week", "month"]:
            raise ValueError("Unrecognized dtime unit, must be 'week' or 'month'.")
        self.dtimes_per_year = 52 if self.dtime_unit == "week" else 12
        self.cols = np.unique([self.col_deaths,
                               self.col_year,
                               self.col_dtime] + self.col_covs)

    def select_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.cols].copy()

    def rename_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        col_names_dict = {
            self.col_deaths: "deaths",
            self.col_year: "year",
            self.col_dtime: self.dtime_unit,
        }
        return df.rename(columns=col_names_dict)

    def add_time(self,
                 df: pd.DataFrame,
                 time_start: Tuple[int, int],
                 time_end: Tuple[int, int]) -> pd.DataFrame:
        df["time"] = (df["year"] - time_start[0])*self.dtimes_per_year + \
            (df[self.dtime_unit] - time_start[1]) + 1
        time_lb = 1
        time_ub = (time_end[0] - time_start[0])*self.dtimes_per_year + \
            (time_end[1] - time_start[1]) + 1
        df = df[(df["time"] >= time_lb) & (df["time"] <= time_ub)]
        return df.reset_index(drop=True)

    def add_offset(self,
                   df: pd.DataFrame,
                   offset_id: int,
                   offset_col: str,
                   offset_fun: Callable) -> pd.DataFrame:
        df[f"offset_{offset_id}"] = offset_fun(df[offset_col])
        return df

    def subset_group(self,
                     df: pd.DataFrame,
                     group_specs: Dict[str, List]) -> pd.DataFrame:
        for col, vals in group_specs.items():
            df = df[df[col].isin(vals)]
        return df.reset_index(drop=True)

    def get_time_min(self, df) -> Tuple[int, int]:
        year_min = df[self.col_year].min()
        tunit_min = df[df[self.col_year] == year_min][self.col_dtime].min()
        return (year_min, tunit_min)

    def get_time_max(self, df) -> Tuple[int, int]:
        year_max = df[self.col_year].max()
        tunit_max = df[df[self.col_year] == year_max][self.col_dtime].max()
        return (year_max, tunit_max)

    def process(self,
                df: pd.DataFrame,
                time_start: Tuple[int, int] = None,
                time_end: Tuple[int, int] = None,
                offset_id: int = 0,
                offset_col: str = None,
                offset_fun: Callable = None,
                group_specs: Dict[str, List] = None) -> pd.DataFrame:
        time_start = self.get_time_min(df) if time_start is None else time_start
        time_end = self.get_time_max(df) if time_end is None else time_end
        group_specs = dict() if group_specs is None else group_specs

        if offset_col is None:
            offset_col = f"offset_{offset_id}"
            df[offset_col] = 0.0
            offset_fun = None

        if offset_fun is None:
            def offset_fun(x): return x

        df = self.select_cols(df)
        df = self.rename_cols(df)
        df = self.add_time(df, time_start, time_end)
        df = self.add_offset(df, offset_id, offset_col, offset_fun)
        df = self.subset_group(df, group_specs)
        return df
