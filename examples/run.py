"""
Main running script
"""
from itertools import product
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from emmodel.data import DataManager
from emmodel.model import (ExcessMortalityModel, plot_data, plot_model,
                           plot_time_trend)
from emmodel.variable import SeasonalityModelVariables, TimeModelVariables
from pandas import DataFrame
from regmod.utils import SplineSpecs
from regmod.variable import SplineVariable


def get_group_specific_data(dm: DataManager,
                            location: str,
                            age_group: str,
                            sex_group: str) -> List[DataFrame]:
    df = dm.read_data_location(location,
                               group_specs={"age_name": [age_group],
                                            "sex": [sex_group]})
    data = []
    for i in range(2):
        df_sub = dm.truncate_time_location(location, df, time_end_id=i)
        df_sub["offset_0"] = np.log(df_sub.population)
        data.append(df_sub)
    return data


def get_data(dm: DataManager) -> Dict[str, List[DataFrame]]:
    data = {}
    for location in dm.locations:
        for age_group, sex_group in product(dm.meta[location]["age_groups"],
                                            dm.meta[location]["sex_groups"]):
            dfs = get_group_specific_data(dm, location, age_group, sex_group)
            age_group = age_group.replace(" ", "_")
            data[f"{location}-{age_group}-{sex_group}"] = dfs
    return data


def get_time_knots(time_min: int,
                   time_max: int,
                   units_per_year: int,
                   knots_per_year: float,
                   tail_size: int) -> np.ndarray:
    body_size = time_max - time_min - tail_size + 1
    num_body_knots = int(knots_per_year*body_size/units_per_year) + 1
    if num_body_knots < 2:
        time_knots = np.array([time_min, time_max])
    else:
        time_knots = np.hstack([
            np.linspace(time_min, time_max - tail_size, num_body_knots),
            time_max
        ])
    return time_knots


def get_mortality_pattern_model(df: DataFrame,
                                col_time: str = "time_start",
                                units_per_year: int = 12,
                                knots_per_year: float = 0.5,
                                tail_size: int = 18) -> ExcessMortalityModel:
    seas_spline_specs = SplineSpecs(knots=np.linspace(0.0, 1.0, 5),
                                    degree=3,
                                    knots_type="rel_domain")
    time_knots = get_time_knots(df.time.min(),
                                df.time.max(),
                                units_per_year,
                                knots_per_year,
                                tail_size)
    time_spline_specs = SplineSpecs(knots=time_knots,
                                    degree=1,
                                    knots_type="abs")
    seas_var = SplineVariable(col_time, spline_specs=seas_spline_specs)
    time_var = SplineVariable("time", spline_specs=time_spline_specs)
    variables = [
        SeasonalityModelVariables([seas_var], col_time),
        TimeModelVariables([time_var])
    ]
    return ExcessMortalityModel(df, variables)


def get_mortality_pattern_models(dm: DataManager,
                                 data: Dict[str, DataFrame]) -> Dict[str, ExcessMortalityModel]:
    models = {}
    for name, dfs in data.items():
        location = name.split("-")[0]
        col_time = dm.meta[location]["col_time"]
        units_per_year = dm.meta[location]["time_start"].units_per_year
        tail_size = dm.meta[location]["tail_size"]
        knots_per_year = dm.meta[location]["knots_per_year"]
        models[name] = get_mortality_pattern_model(dfs[0],
                                                   col_time,
                                                   units_per_year,
                                                   knots_per_year,
                                                   tail_size)
    return models


def plot_models(dm: DataManager,
                results: Dict[str, DataFrame]):
    for name, df in results.items():
        location = name.split("-")[0]

        time_unit = dm.meta[location]["time_unit"]
        col_year = dm.meta[location]["col_year"]

        ax, axs = plot_data(df, time_unit, col_year)
        ax = plot_model(ax, df, "mortality_pattern", color="#008080")
        ax.set_title(name, loc="left")
        ax.legend()
        ax = plot_time_trend(axs[1], df, time_unit, col_year)
        plt.savefig(dm.o_folder / f"{name}.pdf", bbox_inches="tight")
        plt.close("all")


def main(dm: DataManager):
    # get dataframes for each location, age_group and sex_group combination
    data = get_data(dm)

    # get mortality pattern models
    mortality_pattern_models = get_mortality_pattern_models(dm, data)

    # fit mortality pattern models and predict results
    for name, model in mortality_pattern_models.items():
        model.run_models()
        data[name][1] = model.predict(data[name][1],
                                      col_pred="mortality_pattern")
    results = {name: dfs[1] for name, dfs in data.items()}

    # save the mortality pattern results
    dm.write_data(results)

    # plot results and save figures
    plot_models(dm, results)


if __name__ == "__main__":
    # inputs
    i_folder = "examples/data"
    o_folder = "examples/results"
    locations = ["AUT"]

    main(DataManager(i_folder, o_folder, locations))
