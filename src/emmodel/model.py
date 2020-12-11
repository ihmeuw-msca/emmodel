"""
Model module
"""
from typing import List
import numpy as np
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
