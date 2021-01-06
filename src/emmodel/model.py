"""
Model module
"""
from typing import List

import numpy as np
import matplotlib.ticker as tkr
import matplotlib.pyplot as plt
import pandas as pd
import regmod

from emmodel.variable import ModelVariables


class ExcessMortalityModel:

    def __init__(self,
                 df: pd.DataFrame,
                 model_variables: List[ModelVariables],
                 col_obs: str = "deaths"):

        self.df = df
        self.model_variables = model_variables
        self.num_models = len(self.model_variables)
        col_covs = [
            var_name
            for variables in self.model_variables
            for var_name in variables.var_names
        ]
        self.data = [
            regmod.data.Data(col_obs=col_obs,
                             col_covs=col_covs,
                             col_offset=f"offset_{i}")
            for i in range(self.num_models)
        ]
        self.models = []
        self.results = []

    def run_models(self):
        for i in range(self.num_models):
            self.data[i].attach_df(self.df)
            self.models.append(self.model_variables[i].get_model(self.data[i]))
            self.results.append(regmod.optimizer.scipy_optimize(self.models[i]))
            pred = self.models[i].parameters[0].get_param(
                self.results[i]["coefs"], self.data[i]
            )
            if i + 1 == self.num_models:
                self.df["deaths_pred"] = pred
            elif self.model_variables[i + 1].model_type == "Linear":
                self.df[f"offset_{i + 1}"] = pred
            else:
                self.df[f"offset_{i + 1}"] = np.log(pred)
            self.data[i].detach_df()

    def predict(self, df: pd.DataFrame, col_pred: str = "deaths_pred") -> pd.DataFrame:
        """
        Predict expected deaths ('deaths_pred') from data and model fit
        """
        for i in range(self.num_models):
            self.data[i].attach_df(df)
            pred = self.models[i].parameters[0].get_param(
                self.results[i]["coefs"], self.data[i]
            )
            if i + 1 == self.num_models:
                df[col_pred] = pred
                df['trend_residual'] = np.log(df['deaths']) - df[f"offset_{i}"]
                df['time_trend'] = np.log(df[col_pred]) - df[f"offset_{i}"]
            elif self.model_variables[i + 1].model_type == "Linear":
                df[f"offset_{i + 1}"] = pred
            else:
                df[f"offset_{i + 1}"] = np.log(pred)
            self.data[i].detach_df()
        return df


def plot_data(df: pd.DataFrame,
              time_unit: str,
              col_year: str,
              col_deaths: str = "deaths") -> plt.Axes:
    if time_unit not in ["week", "month"]:
        raise ValueError("`time_unit` must be either 'week' or 'month'.")
    units_per_year = 52 if time_unit == "week" else 12

    years = df[col_year].unique()
    year_heads = (years - df[col_year].min())*units_per_year + 1

    axs = plt.subplots(2, figsize=(2.5*len(years), 10))[1]
    ax = axs[0]
    ax.scatter(df.time, df[col_deaths], color="gray")
    ax.set_xticks(year_heads)
    ax.set_xticklabels(years)
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(
        lambda x, p: format(int(x), ",")
    ))

    for time in year_heads:
        ax.axvline(time, linestyle="--", color="gray")
    ax.set_ylabel("deaths")
    ax.set_xlabel("time")

    return ax, axs


def plot_model(ax: plt.Axes,
               df: pd.DataFrame,
               col_pred: str,
               **options) -> plt.Axes:
    ax.plot(df.time, df[col_pred], label=col_pred, **options)
    ax.legend()
    return ax


def plot_time_trend(ax: plt.Axes, df: pd.DataFrame, 
        time_unit: str, col_year: str, **options) -> plt.Axes:
    ax.plot(df.time, df.trend_residual, label='residual', **options)
    ax.plot(df.time, df.time_trend, label='time trend', **options)
    ax.legend()

    if time_unit not in ["week", "month"]:
        raise ValueError("`time_unit` must be either 'week' or 'month'.")
    units_per_year = 52 if time_unit == "week" else 12
    years = df[col_year].unique()
    year_heads = (years - df[col_year].min())*units_per_year + 1
    ax.set_xticks(year_heads)
    ax.set_xticklabels(years)
    ax.set_xlabel("time")
    return ax