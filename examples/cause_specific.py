from typing import Dict, List, Tuple
from emmodel.model import ExcessMortalityModel, plot_data, plot_model
from emmodel.variable import (ModelVariables, SeasonalityModelVariables,
                              TimeModelVariables)
from regmod.prior import UniformPrior
from regmod.utils import SplineSpecs
from regmod.variable import SplineVariable, Variable
from emmodel.cascade import Cascade, CascadeSpecs
from pathlib import Path 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

level = 0.1
cascade_specs = {
            "prior_masks": {},
            "level_masks": [100.0, level]
    }
use_death_rate_covid = True 
model_type = "Poisson"
cov = "death_rate_covid" if use_death_rate_covid else "deaths_covid"

o_folder = Path(f"/Users/jiaweihe/Downloads/mortality/results/test_cause_specific_{cov}_{model_type}_{level}/")
if not o_folder.exists():
    o_folder.mkdir()

def flatten_dict(d: Dict, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


df = pd.read_csv("~/Downloads/doc/data/cause_specific/for_Peng_non_natural_cause.csv")
df_2020 = df.loc[:,['location_name','year_x','week','pop_2020',
    'covid_rate', 'death_rate_2020']]
df_2019 = df.loc[:,['location_name','year_y','week','pop_2019', 'death_rate_2019']]
df_2020 = df_2020.rename(columns={'year_x':'year','pop_2020':'population',
	'death_rate_2020':'death_rate', 'covid_rate':'death_rate_covid'})
df_2019 = df_2019.rename(columns={'year_y':'year','pop_2019':'population',
	'death_rate_2019':'death_rate'})
df_2019['death_rate_covid'] = 0
df_all = df_2019.append(df_2020)
df_all['deaths'] = df_all['death_rate'] * df_all['population'] / 1000
df_all['deaths_covid'] = df_all['death_rate_covid'] * df_all['population'] / 1000

time_start = (2019, 1)
time_end_0 = (2020, 9)
time_end_1 = (2020, 50)
time_ub_0 = (time_end_0[0] - time_start[0])*52 + time_end_0[1] - time_start[1] + 1
time_ub_1 = (time_end_1[0] - time_start[0])*52 + time_end_1[1] - time_start[1] + 1
data_0 = {}
data_1 = {}

for location in df_all.location_name.unique():
	df_loc = df_all.loc[df_all.location_name==location]
	df_loc = df_loc.sort_values(['year','week'])
	df_loc['time'] = (df_loc['year'] - time_start[0])*52 + df_loc['week'] - time_start[1] + 1
	data_0[location] = df_loc.loc[df_loc.time < time_ub_0]
	data_1[location] = df_loc.loc[df_loc.time < time_ub_1]

seas_spline_specs = SplineSpecs(knots=np.linspace(1, 52, 3),
                                degree=2,
                                r_linear=True,
                                knots_type="abs")
time_spline_specs = SplineSpecs(knots=np.linspace(0.0, 1.0, 5),
                                degree=3,
                                knots_type="rel_domain")
col_time = 'week'
models = {}
for name, df in data_0.items():
    df["offset_0"] = np.log(df.population)
    seas_var = SplineVariable(col_time, spline_specs=seas_spline_specs)
    time_var = SplineVariable("time", spline_specs=time_spline_specs)
    # variables = [
    #     SeasonalityModelVariables([seas_var], col_time),
    #     TimeModelVariables([time_var])
    # ]
    variables = [
        SeasonalityModelVariables([seas_var], col_time, smooth_order=1)
    ]
    models[name] = ExcessMortalityModel(df, variables)

results = {}
for name, model in models.items():
    print(f"  fit {name}")
    model.run_models()
    data_1[name]["offset_0"] = np.log(data_1[name].population)
    df_pred = model.predict(data_1[name], col_pred="mortality_pattern")
    results[name] = df_pred


if model_type == "Poisson":
    for df in results.values():
        df["offset_0"] = np.log(df["mortality_pattern"])
elif model_type == "Linear":
    for df in results.values():
        df["offset_0"] = df["mortality_pattern"]
else:
    raise Exception(f"Not valid model_type: {model_type}")


covid_var = Variable(cov, priors=[UniformPrior(lb=-np.inf, ub=np.inf)])

variables = [ModelVariables([covid_var], model_type=model_type)]
specs = CascadeSpecs(variables, **cascade_specs)

# create level 0 model
df_all = pd.concat([results[location] for location in results.keys()])
cmodel_lvl0 = Cascade(df_all, specs, level_id=0, name="all")

# create level 1 model
cmodel_lvl1 = {
    location: Cascade(results[location], specs, level_id=1, name=location)
    for location in results.keys()
}

# link models
cmodel_lvl0.add_children(list(cmodel_lvl1.values()))

cmodels = cmodel_lvl0, cmodel_lvl1

cmodels[0].run_models()
names = ["all"]
coefs = [cmodels[0].model.results[0]["coefs"][0]]
results = {"all": cmodels[0].model.df}

print(level)

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

data_age_cc = cmodels, results
leaf_cmodels = data_age_cc[0][1]
# leaf_cmodels.update(flatten_dict(data_age_cc[0][2]))

results["cascade_coefs"].to_csv(o_folder / "coefs.csv", index=False)

for name, cmodel in leaf_cmodels.items():
    df = cmodel.model.df
    name = name.replace(" ", "_")
    location = name.split("_")[0]
    ax, axs = plot_data(df,
                   'week',
                   'year')
    plt.delaxes(axs[1])
    ax = plot_model(ax, df, "deaths_pred", color="#008080")
    ax = plot_model(ax, df, "mortality_pattern", color="#E7A94D",
                    linestyle="--")
    ax.set_title(name, loc="left")
    ax.legend()
    plt.savefig(o_folder / f"{name}.pdf",
                bbox_inches="tight")
    plt.close("all")
