"""
Logit ratio analysis between excess mortality and covid deaths

- use covaraite idr_lagged
- pre-analysis getting prior
- cascade model infer location variation
"""
from typing import List, Dict, Union
from pathlib import Path
from scipy.optimize import LinearConstraint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from regmod.data import Data
from regmod.variable import Variable, SplineVariable
from regmod.utils import SplineSpecs
from regmod.prior import UniformPrior, SplineUniformPrior, SplineGaussianPrior
from regmod.optimizer import scipy_optimize
from emmodel.model import ExcessMortalityModel
from emmodel.variable import ModelVariables
from emmodel.cascade import CascadeSpecs, Cascade


# data file path
data_path = Path("./examples/data_debug/2020-04-22/stage2_input.csv")

# result folder
results_path = Path("./examples/results_debug")

# define all variables
intercept_variable = Variable("intercept")

idr_spline_specs = SplineSpecs(
     knots=np.linspace(0.0, 1.0, 5),
     degree=2,
     knots_type="rel_domain",
     include_first_basis=False
)
idr_variable = SplineVariable("idr_lagged",
                              spline_specs=idr_spline_specs,
                              priors=[SplineUniformPrior(order=1, lb=-np.inf, ub=0.0),
                                      SplineGaussianPrior(order=1, mean=0.0, sd=1e-4,
                                                          domain_lb=0.4, domain_ub=1.0)])

time_spline_specs = SplineSpecs(
    knots=np.linspace(0.0, 1.0, 10),
    degree=2,
    knots_type="rel_domain",
    include_first_basis=False,
    r_linear=True
)
time_variable = SplineVariable("time_id",
                               spline_specs=time_spline_specs,
                               priors=[SplineGaussianPrior(order=1, mean=0.0, sd=1e-4,
                                                           domain_lb=0.9, domain_ub=1.0, size=2)])



# create variables for IDR global model
idr_model_variables = ModelVariables(
    [intercept_variable,
     idr_variable],
    model_type="Linear"
)

# create variables for cascade
cascade_model_variables = ModelVariables(
    [intercept_variable,
     idr_variable,
     time_variable],
    model_type="Linear"
)

# construct the cascade model specification
cascade_specs = CascadeSpecs(
    model_variables=[cascade_model_variables],
    prior_masks={"intercept": [np.inf],
                 "idr_lagged": [1.0]*idr_variable.size,
                 "time_id": [1.0]*(time_variable.size - 1) + [0.1]},
    level_masks=[1.0, 1.0, 10.0, 10.0],
    col_obs="logit_ratio"
)

# sample setting
num_samples = 1000
np.random.seed(123)


# prediction function
def predict(df_pred: pd.DataFrame,
            model: ExcessMortalityModel,
            col_pred: str = "logit_ratio") -> pd.DataFrame:
    df_pred = df_pred.copy()
    pred = np.zeros(df_pred.shape[0])
    for i in range(model.num_models):
        model.data[i].attach_df(df_pred)
        pred = pred + model.models[i].params[0].get_param(
            model.results[i]["coefs"], model.data[i]
        )
        model.data[i].detach_df()
    df_pred[col_pred] = pred
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
def plot_model(df, pred_dfs, locations) -> plt.Axes:
    _, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df.time_id, df.logit_ratio, color="gray")
    colors = ["red", "#E88734", "#008080", "#02639B", "#172128"]
    for i, location in enumerate(locations):
        if location not in pred_dfs:
            continue
        pred_dfs[location].sort_values("time_id", inplace=True)
        ax.plot(pred_dfs[location].time_id,
                pred_dfs[location].logit_ratio,
                color=colors[i], label=location)
    if len(locations) == 1:
        index = [True]*df.shape[0]
    elif len(locations) == 2:
        index = df.super_region_name == locations[-1]
    elif len(locations) == 3:
        index = df.region_name == locations[-1]
    else:
        index = df.ihme_loc_id == locations[-1]
    ax.scatter(df.time_id[index], df.logit_ratio[index], color="#38ACEC")
    ax.set_xlabel("time_id")
    ax.set_ylabel("logit_ratio")
    ax.set_title(locations[-1])
    ax.legend()
    return ax


def sample_coefs(cmodel: Cascade) -> pd.DataFrame:
    model = cmodel.model
    coefs = np.random.multivariate_normal(
        mean=model.results[-1]["coefs"],
        cov=model.results[-1]["vcov"],
        size=num_samples
    )
    return pd.DataFrame(
        coefs,
        columns=[variable.name
                 for variable in model.models[-1].params[0].variables]
    )


def main():
    # load data
    df_all = pd.read_csv(data_path)
    df_all = df_all[df_all.include].reset_index(drop=True)

    national_index = df_all.ihme_loc_id.str.len() == 3
    df_national = df_all[national_index].reset_index(drop=True)
    df_subnational = df_all[~national_index].reset_index(drop=True)

    df = df_national

    # create results folder
    if not results_path.exists():
        results_path.mkdir()

   # Fit global IDR model
    idr_model = ExcessMortalityModel(df, [idr_model_variables], col_obs="logit_ratio")
    idr_model.run_models()


    # attach data to create spline
    data = Data(
        col_obs="logit_ratio",
        col_covs=[intercept_variable.name,
                idr_variable.name,
                time_variable.name]
    )
    data.df = df
    idr_variable.check_data(data)
    time_variable.check_data(data)

    # fix idr coefficients
    coefs = idr_model.results[0]["coefs"][1:]
    idr_variable.add_priors(UniformPrior(lb=coefs, ub=coefs))

    # getting location structure
    location_structure = {}
    for super_region in df.super_region_name.unique():
        regions = df[df.super_region_name == super_region].region_name.unique()
        location_structure[super_region] = {}
        for region in regions:
            nationals = list(
                df_national[df_national.region_name == region].ihme_loc_id.unique()
            )
            location_structure[super_region][region] = {}
            for national in nationals:
                subnational_index = df_subnational.ihme_loc_id.str.startswith(national)
                location_structure[super_region][region][national] = list(
                    df_subnational.ihme_loc_id[subnational_index].unique()
                )

    # construct cascade model
    # global model
    global_model = Cascade(df, cascade_specs, level_id=0, name="Global")

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

    # national model
    national_models = [
        Cascade(df_national[df_national.ihme_loc_id == national].reset_index(drop=True),
                cascade_specs,
                level_id=3,
                name=national)
        for national in df_national.ihme_loc_id.unique()
    ]

    # subnational model
    subnational_models = [
        Cascade(df_subnational[df_subnational.ihme_loc_id == subnational].reset_index(drop=True),
                cascade_specs,
                level_id=4,
                name=subnational)
        for subnational in df_subnational.ihme_loc_id.unique()
    ]

    # link all models together
    link_cascade_models(global_model,
                        [super_region_models, region_models, national_models, subnational_models],
                        location_structure)

    # fit model
    global_model.run_models()

    # create plots
    model_list = global_model.to_list()

    # predict
    pred_dfs = {}
    for cmodel in model_list:
        pred_dfs[cmodel.name] = predict(cmodel.df, cmodel.model)

    # plot
    for loc_id in df_all.ihme_loc_id.unique():
        df_sub = df_all[df_all.ihme_loc_id == loc_id]
        loc_structure = [
            "Global",
            df_sub.super_region_name.values[0],
            df_sub.region_name.values[0]
        ]
        if len(loc_id) > 3:
            loc_structure.extend([loc_id[:3], loc_id])
        else:
            loc_structure.append(loc_id)
        plot_model(df_all, pred_dfs, loc_structure)
        plt.savefig(results_path / f"{loc_id}.pdf", bbox_inches="tight")
        plt.close("all")

    # # create results dataframe
    # coefs = pd.concat([model.model.get_coefs_df() for model in model_list])
    # coefs["location"] = [model.name for model in model_list]

    # coefs.to_csv(results_path / "coefs.csv", index=False)

    # # create samples of the coefficient
    # for cmodel in model_list:
    #     df_coefs = sample_coefs(cmodel)
    #     df_coefs.to_csv(results_path / f"cdraws_{cmodel.level_id}_{cmodel.name}.csv", index=False)

    return model_list


if __name__ == "__main__":
    models = main()
