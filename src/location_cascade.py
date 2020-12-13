import os
import numpy as np 
import pandas as pd 
import pickle
import regmod 

from emmodel.cascade import Cascade, CascadeSpecs
from emmodel.variable import ModelVariables, TimeModelVariables, YearModelVariables
from emmodel.data import DataProcessor


def location_specific_cascade(df):
    dp = DataProcessor(
        col_deaths='deaths',
        col_year='year_start', 
        col_tunit='week_start',
        col_population='population',
        col_covs=["age_name", "sex", "deaths_covid"]
        )
    df_all_age = dp.process(df, group_specs={"age_name": ["0 to 125"], "sex": ["all"]})
    spline_specs = regmod.utils.SplineSpecs(
        knots=np.linspace(0, 1, 5),
        degree=3,
        knots_type="rel_domain"
        )
    intercept_variable = regmod.variable.Variable("intercept")
    time_variable = regmod.variable.SplineVariable("time", spline_specs=spline_specs)
    year_variable = regmod.variable.SplineVariable("week", spline_specs=spline_specs)
    year_variable.add_priors(regmod.prior.UniformPrior(
        lb=np.array([0.0] + [-np.inf]*(year_variable.size - 1)),
        ub=np.array([0.0] + [np.inf]*(year_variable.size - 1)))
    )
    variables = [
        YearModelVariables([intercept_variable, year_variable]),
        TimeModelVariables([time_variable])
    ]
    prior_masks = {'time': np.repeat(np.inf, time_variable.size),
                   'week': np.repeat(1.0, year_variable.size)}
    level_masks = [0.1]

    specs = CascadeSpecs(variables,
                         prior_masks=prior_masks,
                         level_masks=level_masks)
    # create level 0 model
    cascade = Cascade(df_all_age, specs, level_id=0)
    cascade.run_models()

    num_models = cascade.model.num_models
    # Prediction of location specific model
    df_all_age['offset'] = df_all_age['offset_{}'.format(cascade.model.num_models)]
    df_all_age = df_all_age.drop([f'offset_{i}' for i in range(num_models+1)], axis=1)
    df_all_age['location_id'] = df.location_id.unique()[0]
    return df_all_age


def first_stage(data_folder):
    """Read data and run model for each location separately.
       Use the prediction as offset for later stages.
    """
    data_files = os.listdir(data_folder)
    lst = []
    for data_file in data_files:
        loc = data_file.split(".csv")[0]
        df = pd.read_csv(os.path.join(data_folder, f"{loc}.csv"))
        time_unit = df.time_unit.unique()
        if len(time_unit) == 1:
            # Process locations with weekly data for now
            if time_unit[0] == 'week':
                df = df.rename(columns={'time_start': 'week_start'})
            # Deal with monthly data later
    #         elif time_unit[0] == 'month':
    #             df = df.rename(columns={'time_start': 'month_start'})
            else:
                continue
        # Fill in the NaNs with zero.
        df['deaths_covid'] = df['deaths_covid'].fillna(0)
        # Drop rows where deaths data are NaNs.
        df = df[~df.deaths.isnull()]
        # Run model for each location separately to get the prediction as offset 
        df_all_location = location_specific_cascade(df)
        lst.append(df_all_location)
    df_all_location = pd.concat(lst)
    return df_all_location


def second_stage(df_all_location):
    # Create children of each location.
    location_ids = df_all_location.location_id.unique()
    df_location_specific = {}
    for loc in location_ids:
        df_location_specific[loc] = df_all_location.loc[df_all_location.location_id == loc]

    spline_specs = regmod.utils.SplineSpecs(
        knots=np.linspace(0, 1, 5),
        degree=3,
        knots_type="rel_domain"
    )
    intercept_variable = regmod.variable.Variable("intercept")
    time_variable = regmod.variable.SplineVariable("time", spline_specs=spline_specs)
    year_variable = regmod.variable.SplineVariable("week", spline_specs=spline_specs)
    offset_variable = regmod.variable.Variable("offset")
    covid_variable = regmod.variable.Variable("deaths_covid")
    year_variable.add_priors(
        regmod.prior.UniformPrior(lb=np.array([0.0] + [-np.inf]*(year_variable.size - 1)),
            ub=np.array([0.0] + [np.inf]*(year_variable.size - 1)))
        )
    variables = [
        YearModelVariables([intercept_variable, year_variable]),
        TimeModelVariables([time_variable, offset_variable])
    ]

    prior_masks = {'time': np.repeat(np.inf, time_variable.size),
                   'week': np.repeat(1.0, year_variable.size),
                   'offset': np.repeat(1.0, offset_variable.size)}
    level_masks = [0.1]

    specs = CascadeSpecs(variables,
                         prior_masks=prior_masks,
                         level_masks=level_masks)

    # create level 0 model
    cascade_all_location = Cascade(df_all_location, specs, level_id=0)

    # create level 1 model
    cascade_location_specifics = [
        Cascade(df_location_specific[location_group], specs, level_id=1)
        for location_group in location_ids
    ]

    cascade_all_location.add_children(cascade_location_specifics)
    cascade_all_location.run_models()
    return cascade_all_location, cascade_location_specifics


def save_results(cascade_all_location, cascade_location_specifics, result_folder, suffix):
    # Plot model and save results
    for cascade_loc in cascade_location_specifics:
        mae = np.mean(np.abs(
            np.exp(cascade_loc.df[f"offset_{cascade_loc.model.num_models}"]) \
                - cascade_loc.df.deaths
            )
        )
        location_id = cascade_loc.df.location_id.unique()[0]
        name = f"{location_id}_{suffix}.pdf"
        title = f"location_id: {location_id}; MAE: {round(mae)}"
        cascade_loc.model.plot_model(folder=result_folder, name=name, title=title)
        cascade_loc.df.to_csv(os.path.join(result_folder, f"{location_id}_{suffix}.csv"), 
            index=False)
        # Save location specific model
        with open(os.path.join(result_folder, 
            f"location_specific_{location_id}_{suffix}.pkl"), "wb") as f_write:
            pickle.dump(cascade_loc, f_write)
    # Save overall model
    with open(os.path.join(result_folder, f"all_location_{suffix}.pkl"), "wb") as f_write:
            pickle.dump(cascade_all_location, f_write)

def main():
    data_folder = "/home/j/temp/jiaweihe/mortality/2020-12-10-12-52/outputs"
    result_folder = "/home/j/temp/jiaweihe/mortality/2020-12-10-12-52/results"
    # data_folder = "/Users/jiaweihe/Downloads/mortality/data"
    # result_folder = "/Users/jiaweihe/Downloads/mortality/results"

    if not os.path.exists(result_folder):
        os.mkdir(result_folder)

    df_all_location = first_stage(data_folder)
    cascade_all_location, cascade_location_specifics = second_stage(df_all_location)
    save_results(cascade_all_location, cascade_location_specifics, 
        result_folder, suffix="with_offset")


if __name__ == '__main__':
    main()
