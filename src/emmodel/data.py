"""
Data module
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd


@dataclass
class DataProcessor:
    col_deaths: str
    col_year: str
    col_tunit: str
    col_population: str
    col_covs: List[str] = field(default_factory=list)

    def __post_init__(self):
        if "week" in self.col_tunit:
            self.tunit = "week"
            self.tunits_per_year = 52
        elif "month" in self.col_tunit:
            self.tunit = "month"
            self.tunits_per_year = 12
        else:
            raise ValueError("Unrecognized time unit, must be 'week' or 'month'.")

        self.cols = [self.col_deaths,
                     self.col_year,
                     self.col_tunit,
                     self.col_population] + self.col_covs

    def select_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.cols].copy()

    def rename_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        col_names_dict = {
            self.col_deaths: "deaths",
            self.col_year: "year",
            self.col_tunit: self.tunit,
            self.col_population: "population"
        }
        return df.rename(columns=col_names_dict)

    def add_time(self,
                 df: pd.DataFrame,
                 time_start: Tuple[int, int],
                 time_end: Tuple[int, int]) -> pd.DataFrame:
        df["time"] = (df["year"] - time_start[0])*self.tunits_per_year + \
            (df[self.tunit] - time_start[1])
        time_lb = 0
        time_ub = (time_end[0] - time_start[0])*self.tunits_per_year + \
            (time_end[1] - time_start[1])
        df = df[df["time"] >= time_lb & df["time"] <= time_ub]
        return df.reset_index(drop=True)

    def add_offset(self,
                   df: pd.DataFrame,
                   offset_id: int) -> pd.DataFrame:
        df[f"offset_{offset_id}"] = np.log(df.population)
        return df

    def subset_group(self,
                     df: pd.DataFrame,
                     group_specs: Dict[str, List]) -> pd.DataFrame:
        for col, vals in group_specs.items():
            df = df[df[col].isin(vals)]
        return df.reset_index(drop=True)

    def process(self,
                df: pd.DataFrame,
                time_start: Tuple[int, int] = (2010, 0),
                time_end: Tuple[int, int] = (2020, 8),
                offset_id: int = 0,
                group_specs: Dict[str, List] = dict()) -> pd.DataFrame:
        df = self.select_cols(df)
        df = self.rename_cols(df)
        df = self.add_time(df, time_start, time_end)
        df = self.add_offset(df, offset_id)
        df = self.subset_group(df, group_specs)
        return df
