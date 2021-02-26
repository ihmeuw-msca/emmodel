"""
Example for Cause of Death Team
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from emmodel.data import add_time
from emmodel.model import ExcessMortalityModel, plot_data, plot_model
from emmodel.utils import YearTime
from emmodel.variable import SeasonalityModelVariables, TimeModelVariables
from regmod.utils import SplineSpecs
from regmod.variable import SplineVariable

# file settings
data_path = "./examples/data/for_Peng_non_natural_cause.csv"
results_folder = Path("./examples/results")

# time unit
time_unit = "week"

# start time
time_start = YearTime(2019, 1, time_unit=time_unit)
# end time for fit and prediction
time_end = {
    "fit": YearTime(2020, 9, time_unit=time_unit),
    "pred": YearTime(2020, 52, time_unit=time_unit)
}

# model setting
# spline specification for seasonality and time trend
seas_spline_specs = SplineSpecs(knots=np.linspace(1, 52, 3),
                                degree=2,
                                r_linear=True,
                                knots_type="abs")
time_spline_specs = SplineSpecs(knots=np.array([0.0, 1.0]),
                                degree=1,
                                knots_type="rel_domain")


def main():
    # process input data
    df = pd.read_csv(data_path)
    df_2020 = df[["location_name", "year_x", "week", "pop_2020", "death_rate_2020"]].copy()
    df_2019 = df[["location_name", "year_y", "week", "pop_2019", "death_rate_2019"]].copy()
    df_2020 = df_2020.rename(columns={"year_x": "year",
                                      "pop_2020": "population",
                                      "death_rate_2020": "death_rate"})
    df_2019 = df_2019.rename(columns={"year_y": "year",
                                      "pop_2019": "population",
                                      "death_rate_2019": "death_rate"})
    df = pd.concat([df_2019, df_2020])
    df["deaths"] = df["death_rate"] * df["population"] / 100000

    time_ub = {k: v - time_start + 1
               for k, v in time_end.items()}

    locations = df.location_name.unique()
    data = {k: {} for k in time_end.keys()}
    for location in locations:
        df_loc = df.loc[df.location_name == location]
        df_loc = df_loc.sort_values(["year", "week"])
        df_loc = add_time(df_loc, "year", "week", time_start)

        for k in time_end.keys():
            data[k][location] = df_loc.loc[df_loc.time < time_ub[k]].copy()
            data[k][location]["offset_0"] = np.log(data[k][location].population)

    # create models
    models = {}
    for location, d in data["fit"].items():
        seas_var = SplineVariable(time_unit, spline_specs=seas_spline_specs)
        time_var = SplineVariable("time", spline_specs=time_spline_specs)
        variables = [
            SeasonalityModelVariables([seas_var], time_unit, smooth_order=1),
            TimeModelVariables([time_var])
        ]
        models[location] = ExcessMortalityModel(d, variables)

    # run models
    results = {}
    for location, model in models.items():
        model.run_models()
        d_pred = model.predict(data["pred"][location],
                               col_pred="mortality_pattern")
        results[location] = d_pred

    # save results
    df_result = pd.concat(results.values())
    df_result.to_csv(results_folder / "prediction.csv", index=False)

    # plot results
    for location, result in results.items():
        ax, axs = plot_data(result, "week", "year")
        plt.delaxes(axs[1])
        ax = plot_model(ax, result, "mortality_pattern", color="#008080")
        ax.set_title(location, loc="left")
        ax.legend()
        plt.savefig(results_folder / f"{location}.pdf", bbox_inches="tight")
        plt.close("all")

    return models, results


if __name__ == "__main__":
    model_dict, result_dict = main()
