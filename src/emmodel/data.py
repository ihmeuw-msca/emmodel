"""
Data module
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
import numpy as np
import pandas as pd


@dataclass
class DataManager:
    i_folder: str
    o_folder: str

    def __post_init__(self):
        self.i_folder = Path(self.i_folder)
        self.o_folder = Path(self.o_folder)

        if not (self.i_folder.exists() and self.i_folder.is_dir()):
            raise ValueError("`i_folder` must be a path to an existing folder.")
        if self.o_folder.exists() and not self.o_folder.is_dir():
            raise ValueError("`o_folder` must be a path to a folder.")

        if not self.o_folder.exists():
            self.o_folder.mkdir()

        self.meta = self.get_meta()
        self.locations = list(self.meta.keys())

    def get_meta(self):
        with open(self.i_folder / "meta.yaml", "r") as f:
            meta = yaml.load(f, Loader=yaml.FullLoader)
        default = meta.pop("default")
        for location in meta:
            meta[location] = {**default, **meta[location]}
        return meta

    def read_data(self,
                  group_specs: Dict,
                  exclude_locations: List[str] = None) -> Dict[str, pd.DataFrame]:
        exclude_locations = [] if exclude_locations is None else exclude_locations
        data = {}
        for location in self.locations:
            if location in exclude_locations:
                continue
            col_year = self.meta[location]["col_year"]
            col_time = self.meta[location]["col_time"]
            col_data = self.meta[location]["col_data"]
            df = pd.read_csv(self.i_folder / f"{location}.csv")
            df = select_cols(df, [col_year, col_time] + col_data)
            df = select_groups(df, group_specs)
            df = add_time(df,
                          col_year,
                          col_time,
                          self.meta[location]["time_start"],
                          self.meta[location]["time_unit"])
            df = df.fillna(0.0)
            data[location] = df
        return data

    def truncate_time(self,
                      data: Dict[str, pd.DataFrame],
                      time_end_id: int = 0) -> Dict[str, pd.DataFrame]:
        truncated_data = {}
        for location, df in data.items():
            col_year = self.meta[location]["col_year"]
            col_time = self.meta[location]["col_time"]
            time_end = self.meta[location][f"time_end_{time_end_id}"]
            time_ub = get_time_from_yeartime(
                time_end[0],
                time_end[1],
                get_time_min(df, col_year, col_time),
                self.meta[location]["time_unit"]
            )
            df = df[df["time"] <= time_ub]
            truncated_data[location] = df.reset_index(drop=True)
        return truncated_data

    def write_data(self, data: Dict[str, pd.DataFrame]):
        for name, df in data.items():
            df.to_csv(self.o_folder / f"{name}.csv", index=False)


def select_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    return df[cols].copy()


def select_groups(df: pd.DataFrame,
                  group_specs: Dict[str, List]) -> pd.DataFrame:
    for col, vals in group_specs.items():
        df = df[df[col].isin(vals)]
    return df.reset_index(drop=True)


def add_time(df: pd.DataFrame,
             col_year: str,
             col_time: str,
             time_start: Tuple[int, int] = (0, 0),
             time_unit: str = "week") -> pd.DataFrame:

    df["time"] = get_time_from_yeartime(df[col_year],
                                        df[col_time],
                                        time_start=time_start,
                                        time_unit=time_unit)
    df = df[df.time >= 1].reset_index(drop=True)
    df.time = df.time - df.time.min() + 1
    return df.reset_index(drop=True)


def get_time_min(df: pd.DataFrame,
                 col_year: str,
                 col_time: str) -> Tuple[int, int]:
    year_min = df[col_year].min()
    time_min = df.loc[df[col_year] == year_min, col_time].min()
    return (year_min, time_min)


def get_time_from_yeartime(year: np.ndarray,
                           time: np.ndarray,
                           time_start: Tuple[int, int],
                           time_unit: str) -> np.ndarray:
    if time_unit not in ["week", "month"]:
        raise ValueError("`time_unit` must be either 'week' or 'month'.")
    units_per_year = 52 if time_unit == "week" else 12
    return (year - time_start[0])*units_per_year + time - time_start[1] + 1
