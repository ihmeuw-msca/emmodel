import os
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from regmod.variable import Variable, SplineVariable
from regmod.utils import SplineSpecs

from emmodel.cascade import Cascade, CascadeSpecs
from emmodel.data import DataProcessor
from emmodel.model import ExcessMortalityModel
from emmodel.variable import (ModelVariables,
                              SeasonalityModelVariables,
                              TimeModelVariables)


time_start = (2010, 1)
time_end_mortality_pattern = {
    "week": (2020, 8),
    "month": (2020, 2)
}
time_end_deaths_covid = {
    "week": (2020, 36),
    "month": (2020, 9)
}
default_spline_specs = SplineSpecs(
    knots=np.linspace(0.0, 1.0, 5),
    degree=3,
    knots_type="rel_domain"
)
level_masks = [10.0]
prior_masks = {}


@dataclass
class DataInterface:
    data_folder: str
    results_folder: str = field(default="./results")
    locations: List[str] = field(init=False)
    location_dtime_units: Dict[str, str] = field(init=False)

    def __post_init__(self):
        self.data_folder = Path(self.data_folder)
        self.results_folder = Path(self.results_folder)

        if not (self.data_folder.exists() and self.data_folder.is_dir()):
            raise ValueError("`data_folder` must be a path to an existing folder.")
        if self.results_folder.exists() and not self.results_folder.is_dir():
            raise ValueError("`result_folder` must be a path to a folder.")

        if not self.results_folder.exists():
            self.results_folder.mkdir()

        self.locations = self.get_locations()
        self.location_dtime_units = self.get_location_dtime_units()

    def get_locations(self) -> List[str]:
        data_files = os.listdir(self.data_folder)
        return [data_file.split(".csv")[0] for data_file in data_files]

    def get_location_dtime_units(self) -> Dict[str, str]:
        location_dtime_units = {}
        for location in self.locations:
            df = pd.read_csv(self.data_folder / f"{location}.csv", nrows=1)
            location_dtime_units[location] = df.time_unit[0]
        return location_dtime_units

    def get_dp(self, location: str) -> DataProcessor:
        return DataProcessor(
            col_deaths="deaths",
            col_year="year_start",
            col_dtime="time_start",
            col_covs=["age_name", "sex", "population", "deaths_covid", "location_id"],
            dtime_unit=self.location_dtime_units[location]
        )

    def read_data(self, locations: List[str] = None) -> Dict[str, pd.DataFrame]:
        locations = self.locations if locations is None else locations
        data = {}
        for location in locations:
            dp = self.get_dp(location)
            df = pd.read_csv(self.data_folder / f"{location}.csv")
            df = dp.select_cols(df)
            df = dp.subset_group(df, {"age_name": ["0 to 125"], "sex": ["all"]})
            df = df.fillna(0.0)
            data[location] = df
        return data if len(data) > 1 else data[locations[0]]

    def write_results(self, results: Dict[str, pd.DataFrame]):
        for location, df in results.items():
            df.to_csv(self.results_folder / f"{location}.csv", index=False)


def fit_mortality_patterns(di: DataInterface,
                           exclude_locations: List[str] = None) -> Dict[str, pd.DataFrame]:
    exclude_locations = [] if exclude_locations is None else exclude_locations
    selected_locations = [location for location in di.locations if location not in exclude_locations]
    data = di.read_data(selected_locations)

    mortality_patterns = {}
    for location in selected_locations:
        dp = di.get_dp(location)
        df = dp.process(data[location],
                        time_start=time_start,
                        time_end=time_end_mortality_pattern[dp.dtime_unit],
                        offset_col="population",
                        offset_fun=np.log)
        seasonality_variable = SplineVariable(dp.dtime_unit, spline_specs=default_spline_specs)
        time_variable = SplineVariable("time", spline_specs=default_spline_specs)
        variables = [
            SeasonalityModelVariables([seasonality_variable]),
            TimeModelVariables([time_variable])
        ]
        model = ExcessMortalityModel(df, variables)
        model.run_models()
        df_pred = dp.process(data[location],
                             time_start=time_start,
                             time_end=time_end_deaths_covid[dp.dtime_unit],
                             offset_col="population",
                             offset_fun=np.log)
        df_pred = model.predict(df_pred)
        df_pred["death_rate_covid"] = df_pred.deaths_covid/df_pred.population
        df_pred["mortality_pattern"] = df_pred.deaths_pred
        df_pred["offset_0"] = df_pred.offset_2
        mortality_patterns[location] = df_pred

    return mortality_patterns


def fit_deaths_covid(di: DataInterface,
                     mortality_patterns: Dict[str, pd.DataFrame],
                     save_plots: bool = True,
                     save_coefs: bool = True,
                     save_results: bool = True) -> Dict[str, pd.DataFrame]:
    df_all = pd.concat(list(mortality_patterns.values()))
    covid_variable = Variable("death_rate_covid")
    variables = [ModelVariables([covid_variable])]
    specs = CascadeSpecs(variables,
                         prior_masks=prior_masks,
                         level_masks=level_masks)

    # create level 0 model
    cmodel_all = Cascade(df_all, specs, level_id=0, name="all")

    # create level 1 model
    cmodel_locations = [
        Cascade(df, specs, level_id=1, name=location)
        for location, df in mortality_patterns.items()
    ]
    cmodel_all.add_children(cmodel_locations)
    cmodel_all.run_models()

    final_results = {}
    names = []
    coefs = []
    for cmodel in cmodel_locations:
        df = cmodel.model.df
        df = df.drop(columns=[col for col in df.columns if "offset" in col])

        if save_plots:
            ax = cmodel.model.plot_model(title=cmodel.name, name=cmodel.name)
            ax.plot(df.time, df.mortality_pattern, color="#E7A94D")
            plt.savefig(di.results_folder / f"{cmodel.name}.pdf", bbox_inches="tight")
            plt.close("all")
        final_results[cmodel.name] = df
        names.append(cmodel.name)
        coefs.append(cmodel.model.results[0]["coefs"][0])

    final_coefs = pd.DataFrame({
        "location": names,
        "coef": coefs
    })

    if save_coefs:
        final_coefs.to_csv(di.results_folder / "coefs.csv", index=False)

    if save_results:
        di.write_results(final_results)

    return final_results, final_coefs


def main():
    data_folder = ""
    results_folder = ""
    di = DataInterface(data_folder, results_folder)
    mortality_patterns = fit_mortality_patterns(di)
    fit_deaths_covid(di, mortality_patterns,
                     save_plots=True, save_coefs=True, save_results=True)


if __name__ == '__main__':
    main()
