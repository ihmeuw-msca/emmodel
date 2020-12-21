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


def get_model_mortality_pattern(data: Dict[str, pd.DataFrame],
                                meta: Dict) -> Dict[str, ExcessMortalityModel]:
    seas_spline_specs = SplineSpecs(knots=np.linspace(0.0, 1.0, 5),
                                    degree=3,
                                    r_linear=True,
                                    knots_type="rel_domain")
    time_spline_specs = SplineSpecs(knots=np.linspace(0.0, 1.0, 5),
                                    degree=3,
                                    knots_type="rel_domain")
    models = {}
    for location, df in data.items():
        df["offset_0"] = np.log(df.population)
        seas_var = SplineVariable(meta[location]["col_time"], spline_specs=seas_spline_specs)
        time_var = SplineVariable("time", spline_specs=time_spline_specs)
        variables = [
            SeasonalityModelVariables([seas_var], meta[location]["col_time"]),
            TimeModelVariables([time_var])
        ]
        models[location] = ExcessMortalityModel(df, variables)
    return models


def run_model_mortality_pattern(models: Dict[str, ExcessMortalityModel],
                                data_pred: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    results = {}
    for location, model in models.items():
        model.run_models()
        data_pred[location]["offset_0"] = np.log(data_pred[location].population)
        df_pred = model.predict(data_pred[location], col_pred="mortality_pattern")
        results[location] = df_pred
    return results


def get_model_cascade(data: Dict[str, pd.DataFrame],
                      cascade_specs: Dict,
                      model_type: str = "Linear",
                      use_death_rate_covid: bool = False) -> Tuple[Cascade, Cascade]:
    cov = "deaths_covid"
    if use_death_rate_covid:
        cov = "death_rate_covid"
        for df in data.values():
            df[cov] = df.deaths_covid/df.population

    if model_type == "Poisson":
        for df in data.values():
            df["offset_0"] = df["offset_2"]
    elif model_type == "Linear":
        for df in data.values():
            df["offset_0"] = df["mortality_pattern"]
    else:
        raise Exception(f"Not valid model_type: {model_type}")

    covid_var = Variable(cov, priors=[UniformPrior(lb=0.0, ub=np.inf)])
    variables = [ModelVariables([covid_var], model_type=model_type)]
    specs = CascadeSpecs(variables, **cascade_specs)

    # create level 0 model
    df_all = pd.concat(list(data.values()))
    cmodel_lvl0 = Cascade(df_all, specs, level_id=0, name="all")

    # create level 1 model
    cmodel_lvl1 = [
        Cascade(df, specs, level_id=1, name=location)
        for location, df in data.items() if not location.startswith("USA")
    ]
    cmodel_USA = Cascade(data["USA"], specs, level_id=1, name="USA")
    cmodel_lvl1.append(cmodel_USA)

    # create level 2 model
    cmodel_lvl2 = [
        Cascade(df, specs, level_id=2, name=location)
        for location, df in data.items() if location.startswith("USA_")
    ]

    # link models
    cmodel_lvl0.add_children(cmodel_lvl1)
    cmodel_USA.add_children(cmodel_lvl2)

    return cmodel_lvl0, cmodel_lvl1 + cmodel_lvl2


def run_model_cascade(root_cmodel: Cascade, leaf_cmodels: List[Cascade]) -> Dict[str, pd.DataFrame]:
    root_cmodel.run_models()
    names = [cmodel.name for cmodel in leaf_cmodels]
    coefs = [cmodel.model.results[0]["coefs"][0] for cmodel in leaf_cmodels]
    final_results = {
        cmodel.name: cmodel.model.df.drop(
            columns=[col for col in cmodel.model.df.columns if "offset" in col]
        )
        for cmodel in leaf_cmodels
    }
    final_results["cascade_coefs"] = pd.DataFrame({
        "location": names,
        "coef": coefs
    })
    final_results["cascade_coefs"].sort_values("coef", inplace=True)
    return final_results


def plot_models(cmodels: List[Cascade], dmanager: DataManager):
    for cmodel in cmodels:
        df = cmodel.model.df
        ax = plot_data(df,
                       dmanager.meta[cmodel.name]["time_unit"],
                       dmanager.meta[cmodel.name]["col_year"])
        ax = plot_model(ax, df, "deaths_pred", color="#008080")
        ax = plot_model(ax, df, "mortality_pattern", color="#E7A94D",
                        linestyle="--")
        ax.set_title(cmodel.name, loc="left")
        ax.legend()
        plt.savefig(dmanager.o_folder / f"{cmodel.name}.pdf",
                    bbox_inches="tight")
        plt.close("all")


if __name__ == "__main__":
    # process inputs -----------------------------------------------------------
    i_folder = "examples/data"
    o_folder = "examples/results"
    exclude_locations = []

    group_specs = {
        "age_name": ["0 to 125"],
        "sex": ["all"]
    }  # not location specific

    cascade_specs = {
        "prior_masks": {},
        "level_masks": [100.0, 0.01]
    }
    model_type = "Linear"
    use_death_rate_covid = False

    # workflow -----------------------------------------------------------------
    # load data
    dmanager = DataManager(i_folder, o_folder)
    data = dmanager.read_data(group_specs,
                              exclude_locations=exclude_locations)
    data_0 = dmanager.truncate_time(data, time_end_id=0)
    data_1 = dmanager.truncate_time(data, time_end_id=1)

    # fit mortality patterns
    models_0 = get_model_mortality_pattern(data_0, dmanager.meta)
    data_1 = run_model_mortality_pattern(models_0, data_1)

    # fit cascade on covid covariate
    models_1 = get_model_cascade(data_1,
                                 cascade_specs,
                                 model_type=model_type,
                                 use_death_rate_covid=use_death_rate_covid)
    final_results = run_model_cascade(*models_1)

    # save result
    dmanager.write_data(final_results)

    # plot model
    plot_models(models_1[1], dmanager)
