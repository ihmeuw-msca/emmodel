"""
Model module
"""
from typing import List, Dict

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

    def predict(self, df: pd.DataFrame, col_pred: str = "deaths_pred", 
                prediction_interval: bool = False, num_draws: int = 1000) -> pd.DataFrame:
        """
        Predict expected deaths ('deaths_pred') from data and model fit
        """
        for i in range(self.num_models):
            self.data[i].attach_df(df)
            if not prediction_interval:
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
            else:
                coef, vcov = self.results[i]['coefs'], self.results[i]['vcov']
                parameters, data = self.models[i].parameters[0], self.data[i]
                pred_mean, pred_lower, pred_upper = \
                    simulate_uncertainty(coef, vcov, parameters, data, num_draws)
                if i + 1 == self.num_models:
                    df[f"{col_pred}_mean"] = pred_mean
                    df[f"{col_pred}_lower"] = pred_lower
                    df[f"{col_pred}_upper"] = pred_upper
                    df['trend_residual'] = np.log(df['deaths']) - df[f"offset_{i}"]
                    df['time_trend_mean'] = np.log(df[f"{col_pred}_mean"]) - df[f"offset_{i}"]
                    df['time_trend_lower'] = np.log(df[f"{col_pred}_lower"]) - df[f"offset_{i}"]
                    df['time_trend_upper'] = np.log(df[f"{col_pred}_upper"]) - df[f"offset_{i}"]
                elif self.model_variables[i + 1].model_type == "Linear":
                    df[f"offset_{i + 1}_mean"] = pred_mean
                    df[f"offset_{i + 1}_lower"] = pred_lower
                    df[f"offset_{i + 1}_upper"] = pred_upper
                else:
                    df[f"offset_{i + 1}_mean"] = np.log(pred_mean)
                    df[f"offset_{i + 1}_lower"] = np.log(pred_lower)
                    df[f"offset_{i + 1}_upper"] = np.log(pred_upper)
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
    for time in year_heads:
        ax.axvline(time, linestyle="--", color="gray")
    return ax


def simulate_uncertainty(coef, vcov, parameters, data, num_draws):
    lst_pred = []
    for _ in range(num_draws):
        coef_sim = np.random.multivariate_normal(coef, vcov)
        pred = parameters.get_param(coef_sim, data)
        lst_pred.append(pred)
    pred_mean = np.mean(lst_pred, axis=0)
    pred_lower = np.percentile(lst_pred, 2.5, axis=0)
    pred_upper = np.percentile(lst_pred, 97.5, axis=0)
    return pred_mean, pred_lower, pred_upper


class LinearExcessMortalityModel:
    """LinearExcessMortalityModel"""
    def __init__(self, df: pd.DataFrame, col_obs: str, 
                 stage1_col_covs: List[str], stage2_col_covs: List[str],
                 col_covs_priors: List[str]):
        self.df = df
        self.col_obs = col_obs
        self.stage1_col_covs = stage1_col_covs
        self.stage2_col_covs = stage2_col_covs
        self.col_covs_priors = col_covs_priors
        self.data_stage1 = regmod.data.Data(col_obs=self.col_obs, col_covs=self.stage1_col_covs)
        self.data_stage2 = regmod.data.Data(col_obs=self.col_obs, col_covs=self.stage2_col_covs)

    def run(self, prior_min_var: float = 0.1):
        self.run_stage_1()
        self.priors = self.result_to_priors(min_var=prior_min_var)
        self.run_stage_2()

    def run_stage_1(self):
        dropna_col_covs = list(set(self.stage1_col_covs) & set(self.df.columns))
        self.data_stage1.attach_df(self.df.dropna(subset=dropna_col_covs))
        self.variables_stage1 = [
            regmod.variable.Variable(self.stage1_col_covs[ix]) 
                for ix in range(len(self.stage1_col_covs))
            ]
        self.model_stage1 = regmod.model.LinearModel(self.data_stage1, self.variables_stage1)
        self.result_stage1 = regmod.optimizer.scipy_optimize(self.model_stage1)
        self.data_stage1.detach_df()

    def run_stage_2(self):
        dropna_col_covs = list(set(self.stage2_col_covs) & set(self.df.columns))
        self.data_stage2.attach_df(self.df.dropna(subset=dropna_col_covs))
        self.variables_stage2 = []
        # Add priors for specified covariates
        for cov in self.col_covs_priors:
            self.variables_stage2.append(regmod.variable.Variable(cov, priors=[self.priors[cov]]))
        # Include other covariates
        for cov in self.stage2_col_covs:
            if cov not in self.col_covs_priors:
                self.variables_stage2.append(regmod.variable.Variable(cov))
        self.model_stage2 = regmod.model.LinearModel(self.data_stage2, self.variables_stage2)
        self.result_stage2 = regmod.optimizer.scipy_optimize(self.model_stage2)
        self.data_stage2.detach_df()

    def result_to_priors(self, min_var: float = 0.1) -> Dict:
        mean = self.result_stage1["coefs"]
        var_names = [variable.name for variable in self.variables_stage1]
        sd = np.sqrt(np.maximum(min_var, np.diag(self.result_stage1["vcov"])))
        slices = regmod.utils.sizes_to_sclices([var.size for var in self.variables_stage1])
        priors = {
            var_name: regmod.prior.GaussianPrior(mean=mean[slices[i]], sd=sd[slices[i]]) 
            for i, var_name in enumerate(var_names) if var_name != 'intercept'
        }
        return priors

    def predict(self, df: pd.DataFrame, prediction_interval: bool = False, 
                num_draws: int = 1000) -> pd.DataFrame:
        dropna_col_covs = list(set(self.stage2_col_covs) & set(df.columns)) 
        df = df.dropna(subset=dropna_col_covs)
        data = regmod.data.Data(col_obs=self.col_obs, col_covs=self.stage2_col_covs)
        data.attach_df(df)
        if not prediction_interval:
            df['pred'] = self.model_stage2.parameters[0].get_param(self.result_stage2['coefs'], data)
        else:
            coef, vcov = self.result_stage2['coefs'], self.result_stage2['vcov']
            parameters = self.model_stage2.parameters[0]
            pred_mean, pred_lower, pred_upper = \
                simulate_uncertainty(coef, vcov, parameters, data, num_draws)
            df['pred_mean'] = pred_mean
            df['pred_lower'] = pred_lower
            df['pred_upper'] = pred_upper
        data.detach_df()
        return df
