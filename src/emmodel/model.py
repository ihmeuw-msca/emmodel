"""
Model module
"""
import os
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import regmod

from emmodel.variable import ModelVariables


class ExcessMortalityModel:

    def __init__(self,
                 df: pd.DataFrame,
                 model_variables: List[ModelVariables]):

        self.df = df
        self.model_variables = model_variables
        self.num_models = len(self.model_variables)
        col_covs = [
            var_name
            for variables in self.model_variables
            for var_name in variables.var_names
        ]
        self.data = [
            regmod.data.Data(col_obs="deaths",
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
            self.df[f"offset_{i + 1}"] = np.log(self.models[i].parameters[0].get_param(
                self.results[i]["coefs"], self.data[i]
            ))
            self.data[i].detach_df()

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict expected deaths ('deaths_pred') from data and model fit
        """
        for i in range(self.num_models):
            self.data[i].attach_df(df)
            pred = np.log(self.models[i].parameters[0].get_param(
                self.results[i], self.data[i]
            ))
            df[f"offset_{i + 1}"] = pred
        df["deaths_pred"] = np.exp(pred)
        return df

    def plot_model(self, time_start=(2010, 0), ax=None, title=None, folder=None, name="unknown"):
        tunits_per_year = 52 if "week" in self.df.columns else 12
        tunit = "week" if "week" in self.df.columns else "month"

        years = self.df.year.unique()
        year_heads = (years - time_start[0])*tunits_per_year + \
            self.df.loc[self.df.year==years.min(), tunit].min() - time_start[1]

        if ax is None:
            ax = plt.subplots(1, figsize=(2.5*len(years), 5))[1]

        ax.scatter(self.df.time, self.df.deaths, color="gray")
        ax.plot(self.df.time,
                np.exp(self.df[f"offset_{self.num_models}"]),
                color="#008080")
        ax.set_xticks(year_heads)
        ax.set_xticklabels(years)
        for year_week in year_heads:
            ax.axvline(year_week, linestyle="--", color="gray")
        ax.set_ylabel("deaths")
        ax.set_xlabel("time")

        if title is not None:
            ax.set_title(title, loc="left")

        if folder is not None:
            folder = Path(folder)
            if not folder.exists():
                os.mkdir(folder)
            plt.savefig(folder / f"{name}.pdf", bbox_inches='tight')
        return ax
