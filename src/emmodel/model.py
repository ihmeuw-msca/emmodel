"""
Model module
"""
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.ticker as tkr
import matplotlib.pyplot as plt
import pandas as pd
from regmod.variable import Variable
from regmod.optimizer import scipy_optimize
from regmod.models import GaussianModel
from regmod.data import Data
from regmod.utils import sizes_to_sclices
from regmod.prior import GaussianPrior

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
            Data(col_obs=col_obs,
                 col_covs=col_covs,
                 col_offset=f"offset_{i}")
            for i in range(self.num_models)
        ]
        self.models = []
        self.coefs = []
        self.vcovs = []

    def run_models(self):
        for i in range(self.num_models):
            self.data[i].attach_df(self.df)
            self.models.append(self.model_variables[i].get_model(self.data[i]))
            self.coefs.append(scipy_optimize(self.models[i]))
            self.vcovs.append(self.models[i].opt_vcov)
            pred = self.models[i].params[0].get_param(
                self.coefs[i], self.data[i]
            )
            if i + 1 == self.num_models:
                self.df["deaths_pred"] = pred
            elif self.model_variables[i + 1].model_type == "Linear":
                self.df[f"offset_{i + 1}"] = pred
            else:
                self.df[f"offset_{i + 1}"] = np.log(pred)
            self.data[i].detach_df()

    def get_coefs_df(self) -> pd.DataFrame:
        dfs = []
        for i in range(self.num_models):
            variable_names = []
            for param in self.models[i].params:
                for variable in param.variables:
                    if variable.size == 1:
                        variable_names.append(variable.name)
                    else:
                        variable_names.extend([f"{variable.name}_{i}"
                                               for i in range(variable.size)])
            dfs.append(pd.DataFrame({
                "model_id": i,
                "name": variable_names,
                "coef": self.coefs[i],
                "coef_sd": np.sqrt(np.diag(self.vcovs[i]))
            }))
        return pd.concat(dfs, ignore_index=True)

    def get_coefs_samples(self, num_samples: int = 1000) -> List[List[Dict]]:
        """
        Get the samples of coefs.

        Parameters
        ----------
        num_samples : int, optional
            Number of samples, by default 1000

        Returns
        -------
        List[List[Dict]]
            List of samples of coefs, coefs will be in the form of list of
            dictionary.
        """
        coefs_samples = [
            np.random.multivariate_normal(
                self.coefs[i],
                self.vcovs[i],
                size=num_samples
            )
            for i in range(self.num_models)
        ]
        coefs_samples = np.array([
            [{"coefs": coefs_samples[i][j]} for j in range(num_samples)]
            for i in range(self.num_models)
        ])
        return coefs_samples.T

    def _predict(self,
                 df: pd.DataFrame,
                 col_pred: str = "deaths_pred",
                 coefs: List[Dict] = None) -> pd.DataFrame:
        """
        Inner predict function.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe that contains information of the model data.
        col_pred : str, optional
            Name of the final prediction column, by default "deaths_pred"
        coefs : List[Dict], optional
            Provided coefs, by default None. If None, use the instance
            coefs.

        Returns
        -------
        pd.DataFrame
            Predicted Dataframe.
        """
        coefs = self.coefs if coefs is None else coefs
        for i in range(self.num_models):
            self.data[i].attach_df(df)
            if isinstance(coefs[i], dict):
                pred = self.models[i].params[0].get_param(
                    coefs[i]['coefs'], self.data[i]
                    )
            else:
                pred = self.models[i].params[0].get_param(
                    coefs[i], self.data[i]
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

    def get_draws(self,
                  df: pd.DataFrame,
                  col_pred: str = "deaths_pred",
                  num_samples: int = 1000,
                  coefs_samples: List[List[Dict]] = None) -> Dict[str, np.ndarray]:
        """
        Get draws of the prediction.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe that contains information of the model data.
        col_pred : str, optional
            Name of the final prediction column, by default "deaths_pred"
        num_samples : int, optional
            Number of samples, by default 1000. Only use when coefs_samples
            is None.
        coefs_samples : List[List[Dict]], optional
            Samples of coefs, by default None. If None will automatically
            sample coefs.

        Returns
        -------
        Dict[str, np.ndarray]
            Draws of time_trend, time_residual and col_pred.
        """
        if coefs_samples is None:
            coefs_samples = self.get_coefs_samples(num_samples)

        draws = {"time_trend": [], "trend_residual": [], col_pred: []}
        for coefs in coefs_samples:
            df = self._predict(df.copy(), col_pred=col_pred, coefs=coefs)
            for key in draws.keys():
                draws[key].append(df[key].to_numpy())
        for key in draws.keys():
            draws[key] = np.vstack(draws[key])
        return draws

    def predict(self,
                df: pd.DataFrame,
                col_pred: str = "deaths_pred",
                include_ci: bool = False,
                ci_bounds: Tuple[float] = (0.025, 0.975),
                num_samples: int = 1000,
                return_draws: bool = False,
                **kwargs) -> pd.DataFrame:
        """
        Predict expected deaths('deaths_pred') from data and model fit
        """
        df = self._predict(df, col_pred=col_pred)
        if include_ci:
            draws = self.get_draws(df, col_pred=col_pred, num_samples=num_samples, **kwargs)
            for key in draws.keys():
                df[f"{key}_lower"] = np.quantile(draws[key], ci_bounds[0], axis=0)
                df[f"{key}_mean"] = np.mean(draws[key], axis=0)
                df[f"{key}_upper"] = np.quantile(draws[key], ci_bounds[1], axis=0)

                if return_draws:
                    df[[f"{key}_draw_{i}" for i in range(1, num_samples+1)]] = draws[key].T
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
    offset = df.loc[df.year_id == df[col_year].min(), time_unit].min()
    year_heads = year_heads - offset + 1

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
    offset = df.loc[df.year_id == df[col_year].min(), time_unit].min()
    year_heads = year_heads - offset + 1
    ax.set_xticks(year_heads)
    ax.set_xticklabels(years)
    ax.set_xlabel("time")
    for time in year_heads:
        ax.axvline(time, linestyle="--", color="gray")
    return ax


def summarize_uncertainty(lst, ci_bounds):
    pred_mean = np.mean(lst, axis=0)
    pred_lower = np.percentile(lst, ci_bounds[0], axis=0)
    pred_upper = np.percentile(lst, ci_bounds[1], axis=0)
    return pred_mean, pred_lower, pred_upper


def simulate_uncertainty(coef, vcov, parameters, data, num_samples, ci_bounds):
    coef_sims = np.random.multivariate_normal(coef, vcov, num_samples)
    lst_pred = [parameters.get_param(coef_sim, data) for coef_sim in coef_sims]
    pred_mean, pred_lower, pred_upper = summarize_uncertainty(lst_pred, ci_bounds)
    return pred_mean, pred_lower, pred_upper, lst_pred


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
        self.data_stage1 = Data(col_obs=self.col_obs, col_covs=self.stage1_col_covs)
        self.data_stage2 = Data(col_obs=self.col_obs, col_covs=self.stage2_col_covs)

    def run(self, prior_min_var: float = 0.1):
        self.run_stage_1()
        self.priors = self.result_to_priors(min_var=prior_min_var)
        self.run_stage_2()

    def run_stage_1(self):
        dropna_col_covs = list(set(self.stage1_col_covs) & set(self.df.columns))
        self.data_stage1.attach_df(self.df.dropna(subset=dropna_col_covs))
        self.variables_stage1 = [
            Variable(self.stage1_col_covs[ix])
            for ix in range(len(self.stage1_col_covs))
        ]
        self.model_stage1 = GaussianModel(self.data_stage1,
                                          param_specs={"mu": {"variables": self.variables_stage1}})
        self.result_stage1 = scipy_optimize(self.model_stage1)
        self.data_stage1.detach_df()

    def run_stage_2(self):
        dropna_col_covs = list(set(self.stage2_col_covs) & set(self.df.columns))
        self.data_stage2.attach_df(self.df.dropna(subset=dropna_col_covs))
        self.variables_stage2 = []
        # Add priors for specified covariates
        for cov in self.col_covs_priors:
            self.variables_stage2.append(Variable(cov, priors=[self.priors[cov]]))
        # Include other covariates
        for cov in self.stage2_col_covs:
            if cov not in self.col_covs_priors:
                self.variables_stage2.append(Variable(cov))
        self.model_stage2 = GaussianModel(self.data_stage2,
                                          param_specs={"mu": {"variables": self.variables_stage2}})
        self.result_stage2 = scipy_optimize(self.model_stage2)
        self.data_stage2.detach_df()

    def result_to_priors(self, min_var: float = 0.1) -> Dict:
        mean = self.result_stage1["coefs"]
        var_names = [variable.name for variable in self.variables_stage1]
        sd = np.sqrt(np.maximum(min_var, np.diag(self.result_stage1["vcov"])))
        slices = sizes_to_sclices([var.size for var in self.variables_stage1])
        priors = {
            var_name: GaussianPrior(mean=mean[slices[i]], sd=sd[slices[i]])
            for i, var_name in enumerate(var_names) if var_name != 'intercept'
        }
        return priors

    def predict(self, df: pd.DataFrame, include_ci: bool = False,
                num_samples: int = 1000, ci_bounds: Tuple[float] = (0.025, 0.975),
                return_draws: bool = False) -> pd.DataFrame:
        dropna_col_covs = list(set(self.stage2_col_covs) & set(df.columns))
        df = df.dropna(subset=dropna_col_covs)
        data = Data(col_obs=self.col_obs, col_covs=self.stage2_col_covs)
        data.attach_df(df)
        if include_ci:
            coef, vcov = self.result_stage2['coefs'], self.result_stage2['vcov']
            parameters = self.model_stage2.params[0]
            pred_mean, pred_lower, pred_upper, lst_pred = \
                simulate_uncertainty(coef, vcov, parameters, data, num_samples, ci_bounds)
            df['pred_mean'], df['pred_lower'], df['pred_upper'] = pred_mean, pred_lower, pred_upper

            if return_draws:
                df[[f'pred_{i}' for i in range(1, num_samples+1)]] = np.array(lst_pred).T
        else:
            df['pred'] = self.model_stage2.params[0].get_param(self.result_stage2['coefs'], data)
        data.detach_df()
        return df
