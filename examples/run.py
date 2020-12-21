"""
Main running script
"""
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from emmodel.cascade import Cascade, CascadeSpecs
from emmodel.data import DataManager
from emmodel.model import ExcessMortalityModel, plot_data, plot_model
from emmodel.variable import (ModelVariables, SeasonalityModelVariables,
                              TimeModelVariables)
from regmod.prior import UniformPrior
from regmod.utils import SplineSpecs
from regmod.variable import SplineVariable, Variable


def get_model_mp(data: Dict[str, pd.DataFrame],
                 col_time: str = "time_start") -> Dict[str, ExcessMortalityModel]:
    seas_spline_specs = SplineSpecs(knots=np.linspace(0.0, 1.0, 5),
                                    degree=3,
                                    r_linear=True,
                                    knots_type="rel_domain")
    time_spline_specs = SplineSpecs(knots=np.linspace(0.0, 1.0, 5),
                                    degree=3,
                                    knots_type="rel_domain")
    models = {}
    for name, df in data.items():
        df["offset_0"] = np.log(df.population)
        seas_var = SplineVariable(col_time, spline_specs=seas_spline_specs)
        time_var = SplineVariable("time", spline_specs=time_spline_specs)
        variables = [
            SeasonalityModelVariables([seas_var], col_time),
            TimeModelVariables([time_var])
        ]
        models[name] = ExcessMortalityModel(df, variables)
    return models


def run_model_mp(models: Dict[str, ExcessMortalityModel],
                 data_pred: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    results = {}
    for name, model in models.items():
        model.run_models()
        data_pred[name]["offset_0"] = np.log(data_pred[name].population)
        df_pred = model.predict(data_pred[name], col_pred="mortality_pattern")
        results[name] = df_pred
    return results


def get_model_cc(data: Dict[str, Dict[str, pd.DataFrame]],
                 dmanager: DataManager,
                 cascade_specs: Dict,
                 model_type: str = "Linear",
                 use_death_rate_covid: bool = False) -> Tuple[Cascade]:
    cov = "death_rate_covid" if use_death_rate_covid else "deaths_covid"
    if model_type == "Poisson":
        for location_df in data.values():
            for df in location_df.values():
                df["offset_0"] = df["offset_2"]
    elif model_type == "Linear":
        for location_df in data.values():
            for df in location_df.values():
                df["offset_0"] = df["mortality_pattern"]
    else:
        raise Exception(f"Not valid model_type: {model_type}")

    covid_var = Variable(cov, priors=[UniformPrior(lb=0.0, ub=np.inf)])
    variables = [ModelVariables([covid_var], model_type=model_type)]
    specs = CascadeSpecs(variables, **cascade_specs)

    # create level 0 model
    df_all = pd.concat([data[location]["0 to 125"] for location in dmanager.locations])
    cmodel_lvl0 = Cascade(df_all, specs, level_id=0, name="all")

    # create level 1 model
    cmodel_lvl1 = {
        location: Cascade(data[location]["0 to 125"], specs, level_id=1, name=location)
        for location in data.keys()
    }

    # create level 2 model
    cmodel_lvl2 = {
        location: {
            age_group: Cascade(data[location][age_group], specs, level_id=2, name=age_group)
            for age_group in dmanager.meta[location]["age_groups"]
        }
        for location in dmanager.locations
    }

    # link models
    cmodel_lvl0.add_children(list(cmodel_lvl1.values()))
    for location in dmanager.locations:
        cmodel_lvl1[location].add_children(list(cmodel_lvl2[location].values()))

    return cmodel_lvl0, cmodel_lvl1, cmodel_lvl2


def run_model_cc(*cmodels: Tuple[Cascade]) -> Dict[str, pd.DataFrame]:
    cmodels[0].run_models()
    names = ["all"]
    coefs = [cmodels[0].model.results[0]["coefs"][0]]
    results = {"all": cmodels[0].model.df}

    for level in range(1, len(cmodels)):
        level_results = flatten_dict(cmodels[level])
        level_names = list(level_results.keys())
        level_coefs = [level_results[name].model.results[0]["coefs"][0]
                       for name in level_names]
        names.extend(level_names)
        coefs.extend(level_coefs)
        results.update({name: level_results[name].model.df
                        for name in level_names})
    results["cascade_coefs"] = pd.DataFrame({
        "location": names,
        "coef": coefs
    })
    results["cascade_coefs"].sort_values("coef", inplace=True)
    return results


def plot_models(cmodels: Dict[str, Cascade], dmanager: DataManager):
    for name, cmodel in cmodels.items():
        df = cmodel.model.df
        name = name.replace(" ", "_")
        location = name.split("_")[0]
        ax = plot_data(df,
                       dmanager.meta[location]["time_unit"],
                       dmanager.meta[location]["col_year"])
        ax = plot_model(ax, df, "deaths_pred", color="#008080")
        ax = plot_model(ax, df, "mortality_pattern", color="#E7A94D",
                        linestyle="--")
        ax.set_title(name, loc="left")
        ax.legend()
        plt.savefig(dmanager.o_folder / f"{name}.pdf",
                    bbox_inches="tight")
        plt.close("all")


def flatten_dict(d: Dict, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def fit_age_mp_location(location: str, dmanager: DataManager) -> Dict[str, pd.DataFrame]:
    age_all = "0 to 125"
    age_groups = dmanager.meta[location]["age_groups"]
    df_all = dmanager.read_data_location(location, group_specs={"age_name": [age_all], "sex": ["all"]})
    # compute the covid deaths rate
    df_all["death_rate_covid"] = df_all["deaths_covid"]/df_all["population"]
    death_rate_covid = df_all[["time", "death_rate_covid"]].copy()
    data_0 = {age_all: dmanager.truncate_time_location(location, df_all, time_end_id=0)}
    data_1 = {age_all: dmanager.truncate_time_location(location, df_all, time_end_id=1)}
    for age_group in age_groups:
        df = dmanager.read_data_location(location, group_specs={"age_name": [age_group], "sex": ["all"]})
        df = df.merge(death_rate_covid, on="time")
        df["deaths_covid"] = df["death_rate_covid"]*df["population"]
        data_0[age_group] = dmanager.truncate_time_location(location, df, time_end_id=0)
        data_1[age_group] = dmanager.truncate_time_location(location, df, time_end_id=1)
    models = get_model_mp(data_0)
    return run_model_mp(models, data_1)


def fit_age_mp(dmanager: DataManager) -> Dict[str, Dict[str, pd.DataFrame]]:
    results = {}
    for location in dmanager.locations:
        results[location] = fit_age_mp_location(location, dmanager)
    return results


def fit_age_cc(data: Dict[str, Dict[str, pd.DataFrame]],
               dmanager: DataManager,
               cascade_specs: Dict,
               model_type: str = "Linear",
               use_death_rate_covid: bool = False) -> Tuple[Tuple[Cascade], Dict[str, pd.DataFrame]]:
    cmodels = get_model_cc(data, dmanager, cascade_specs, model_type, use_death_rate_covid)
    results = run_model_cc(*cmodels)
    return cmodels, results


if __name__ == "__main__":
    # process inputs -----------------------------------------------------------
    i_folder = "examples/data"
    o_folder = "examples/results"

    cascade_specs = {
        "prior_masks": {},
        "level_masks": [100.0, 1e-2]
    }
    model_type = "Linear"
    use_death_rate_covid = False

    # workflow -----------------------------------------------------------------
    dmanager = DataManager(i_folder, o_folder)
    data_age_mp = fit_age_mp(dmanager)
    data_age_cc = fit_age_cc(data_age_mp, dmanager, cascade_specs, model_type, use_death_rate_covid)
    dmanager.write_data(data_age_cc[1])
    leaf_cmodels = data_age_cc[0][1]
    leaf_cmodels.update(flatten_dict(data_age_cc[0][2]))
    plot_models(leaf_cmodels, dmanager)
