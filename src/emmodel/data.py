"""
Data module
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import yaml
import numpy as np
import pandas as pd

from emmodel.utils import YearTime


@dataclass
class DataManager:
    i_folder: str
    o_folder: str
    locations: List[str] = None

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
        if self.locations is None:
            self.locations = list(self.meta.keys())
        else:
            for location in self.locations:
                if location not in self.meta:
                    raise ValueError(f"{location} not in meta file.")

    def get_meta(self):
        with open(self.i_folder / "meta.yaml", "r") as f:
            meta = yaml.load(f, Loader=yaml.FullLoader)
        default = meta.pop("default")
        for location in meta:
            meta[location] = {**default, **meta[location]}
            for group_key in ["age_groups", "sex_groups"]:
                group_info = meta[location][group_key]
                meta[location][group_key] = group_info if isinstance(group_info, list) else [group_info]
            for time_key in ["time_start", "time_end_0", "time_end_1"]:
                time_value = meta[location][time_key]
                meta[location][time_key] = YearTime(time_value["year"],
                                                    time_value["detailed"],
                                                    time_unit=meta[location]["time_unit"])
        return meta

    def read_data_location(self, location: str, group_specs: Dict) -> pd.DataFrame:
        col_year = self.meta[location]["col_year"]
        col_time = self.meta[location]["col_time"]
        col_data = self.meta[location]["col_data"]
        time_start = self.meta[location]["time_start"]
        df = pd.read_csv(self.i_folder / f"{location}.csv", low_memory=False)
        df = select_cols(df, [col_year, col_time] + col_data)
        df = select_groups(df, group_specs)
        df = df[~df.deaths.isna()].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"Location {location} has no matching data for {group_specs}.")
        df = add_time(df, col_year, col_time, time_start)
        return df.fillna(0.0)

    def read_data(self, group_specs: Dict) -> Dict[str, pd.DataFrame]:
        return {
            location: self.read_data_location(location, group_specs)
            for location in self.locations
        }

    def truncate_time_location(self,
                               location: str,
                               df: pd.DataFrame,
                               time_end_id: int = 0) -> pd.DataFrame:
        col_year = self.meta[location]["col_year"]
        col_time = self.meta[location]["col_time"]
        time_end = self.meta[location][f"time_end_{time_end_id}"]
        time_unit = self.meta[location]["time_unit"]
        year_time = get_yeartime(df, col_year, col_time, time_unit)
        time_ub = time_end - year_time.min()
        return df[df["time"] <= time_ub].reset_index(drop=True)

    def truncate_time(self,
                      data: Dict[str, pd.DataFrame],
                      time_end_id: int = 0) -> Dict[str, pd.DataFrame]:
        truncated_data = {}
        for location, df in data.items():
            truncated_data[location] = self.truncate_time_location(location, df, time_end_id)
        return truncated_data

    def write_data(self,
                   data: Dict[str, pd.DataFrame],
                   prefix: str = "",
                   suffix: str = ""):
        for name, df in data.items():
            name = name.replace(" ", "_")
            name = "_".join([prefix, name, suffix]).strip("_")
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
             time_start: YearTime) -> pd.DataFrame:

    yeartime = get_yeartime(df, col_year, col_time, time_start.time_unit)
    df = df[yeartime >= time_start].reset_index(drop=True)
    yeartime = get_yeartime(df, col_year, col_time, time_start.time_unit)
    df["time"] = (yeartime - yeartime.min()).astype(int) + 1
    return df


def get_yeartime(df: pd.DataFrame,
                 col_year: str,
                 col_time: str,
                 time_unit: str) -> np.ndarray:
    return np.array([YearTime(*t, time_unit=time_unit)
                     for t in zip(df[col_year], df[col_time])])
