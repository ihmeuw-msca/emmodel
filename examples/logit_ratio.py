"""
Logit ratio analysis between excess mortality and covid deaths

- use covaraite idr_lagged
- pre-analysis getting prior
- cascade model infer location variation
"""
from typing import List, Dict, Union
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from regmod.variable import Variable, SplineVariable
from regmod.utils import SplineSpecs
from regmod.prior import UniformPrior
from emmodel.model import ExcessMortalityModel
from emmodel.variable import ModelVariables
from emmodel.cascade import CascadeSpecs, Cascade


# data file path
data_path = "./examples/data/stage_2_input_data.csv"

# result folder
results_path = Path("./examples/results")

# create variables
variables = ModelVariables(
    [Variable("intercept"),
     Variable("log_death_rate_covid"),
     SplineVariable("idr_lagged",
                    spline_specs=SplineSpecs(
                        knots=np.linspace(0.0, 1.0, 3),
                        degree=2,
                        knots_type="rel_domain",
                        include_first_basis=False
                    ))],
    model_type="Linear"
)

# construct the cascade model specification
cascade_specs = CascadeSpecs(
    model_variables=[variables],
    prior_masks={"intercept": [1.0],
                 "log_death_rate_covid": [0.01],
                 "idr_lagged": [1.0, 1.0, 1.0]},
    level_masks=[1.0, 1.0, 100.0],
    col_obs="logit_ratio_0_7"
)


# prediction function
def predict(df_pred: pd.DataFrame,
            model: ExcessMortalityModel) -> pd.DataFrame:
    df_pred = df_pred.copy()
    model.data[0].attach_df(df_pred)
    df_pred["logit_ratio_0_7"] = model.models[0].params[0].get_param(
        model.results[0]["coefs"], model.data[0]
    )
    return df_pred


# link cascade model
def link_cascade_models(root_model: Cascade,
                        leaf_models: List[List[Cascade]],
                        model_structure: Union[Dict, List]):
    if isinstance(model_structure, dict):
        sub_model_names = model_structure.keys()
    else:
        sub_model_names = model_structure
    sub_models = [model for model in leaf_models[0] if model.name in sub_model_names]
    root_model.add_children(sub_models)

    if isinstance(model_structure, dict):
        for model in sub_models:
            link_cascade_models(model, leaf_models[1:], model_structure[model.name])


# plot result
def plot_model(df, pred_dfs, locations):
    _, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df.idr_lagged, df.logit_ratio_0_7, color="gray")
    colors = ["red", "#E88734", "#008080", "#172128"]
    for i, location in enumerate(locations):
        if location not in pred_dfs:
            continue
        ax.plot(pred_dfs[location].idr_lagged,
                pred_dfs[location].logit_ratio_0_7, color=colors[i], label=location)
    if len(locations) == 1:
        index = [True]*df.shape[0]
    elif len(locations) == 2:
        index = df.super_region_name == locations[-1]
    elif len(locations) == 3:
        index = df.region_name == locations[-1]
    else:
        index = df.ihme_loc_id == locations[-1]
    ax.scatter(df.idr_lagged[index], df.logit_ratio_0_7[index], color="#38ACEC")
    ax.set_xlabel("idr_lagged")
    ax.set_ylabel("logit_ratio_0_7")
    ax.set_title(locations[-1])
    ax.legend()
    plt.savefig(results_path / f"{locations[-1]}.pdf", bbox_inches="tight")


def main():
    # loading data
    df = pd.read_csv(data_path)
    df = df[df.include].reset_index(drop=True)

    # pre-analysis getting prior
    # exclude Russian for better curve
    pre_model = ExcessMortalityModel(
        df[~df.ihme_loc_id.str.contains("RUS")].reset_index(drop=True),
        [variables],
        col_obs="logit_ratio_0_7"
    )
    pre_model.run_models()

    # getting location structure
    location_structure = {}
    for super_region in df.super_region_name.unique():
        regions = df[df.super_region_name == super_region].region_name.unique()
        if len(regions) > 1:
            location_structure[super_region] = {}
            for region in regions:
                location_structure[super_region][region] = list(
                    df[df.region_name == region].ihme_loc_id
                )
        else:
            location_structure[super_region] = list(
                df[df.super_region_name == super_region].ihme_loc_id
            )

    # fixed the spline shape
    coefs = pre_model.results[0]["coefs"][2:]
    variables.variables[-1].add_priors(UniformPrior(lb=coefs, ub=coefs))
    cascade_specs.model_variables = [variables]

    # construct cascade model
    # global model
    global_model = Cascade(df, cascade_specs, level_id=0, name="global")

    # super region model
    super_region_models = [
        Cascade(df[df.super_region_name == super_region].reset_index(drop=True),
                cascade_specs,
                level_id=1,
                name=super_region)
        for super_region in df.super_region_name.unique()
    ]

    # region model
    region_models = [
        Cascade(df[df.region_name == region].reset_index(drop=True),
                cascade_specs,
                level_id=2,
                name=region)
        for region in df.region_name.unique()
    ]

    # country and subnational model
    leaf_models = [
        Cascade(df[df.ihme_loc_id.str.contains(country)].reset_index(drop=True),
                cascade_specs,
                level_id=3,
                name=country)
        for country in df.ihme_loc_id.unique()
    ]

    # link all models together
    link_cascade_models(global_model,
                        [super_region_models, region_models, leaf_models],
                        location_structure)

    # fit model
    global_model.run_models()

    # create plots
    model_list = global_model.to_list()

    # predict
    df_pred = pd.DataFrame({
        "idr_lagged": np.linspace(df.idr_lagged.min(),
                                  df.idr_lagged.max(), 100)
    })
    pred_dfs = {}
    for model in model_list:
        for cov in ["log_death_rate_covid"]:
            df_pred[cov] = model.df[cov].mean()
        pred_dfs[model.name] = predict(df_pred, model.model)

    # plot
    for loc_id in df.ihme_loc_id.unique():
        df_sub = df[df.ihme_loc_id == loc_id]
        super_region = df_sub.super_region_name.values[0]
        region = df_sub.region_name.values[0]
        plot_model(df, pred_dfs, ["global", super_region, region, loc_id])
        plt.close("all")

    # create results dataframe
    coefs = pd.concat([model.model.get_coefs_df() for model in model_list])
    coefs["location"] = [model.name for model in model_list]

    coefs.to_csv(results_path / "coefs.csv", index=False)

    return model_list


if __name__ == "__main__":
    models = main()
