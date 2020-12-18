import os
import numpy as np 
import pandas as pd 
import pickle
import matplotlib.pyplot as plt
import regmod 

from emmodel.model import ExcessMortalityModel
from emmodel.cascade import Cascade, CascadeSpecs
from emmodel.variable import ModelVariables, TimeModelVariables, SeasonalityModelVariables
from emmodel.data import DataProcessor


def run_first_stage(data_folder, result_folder, suffix, is_saving_first_stage=False):
    data_files = os.listdir(data_folder)
    lst_df = []
    for data_file in data_files:
        loc = data_file.split(".csv")[0]
        if loc in ['AUS', 'BRA', 'NZL']:
            continue
        df = pd.read_csv(os.path.join(data_folder, f"{loc}.csv"))
        time_unit = df.time_unit.unique()
        if len(time_unit) == 1:
            # Process locations with weekly data for now
            if time_unit[0] == 'month':
                continue
        else:
            raise Exception("More than one time unit.")
        # Fill in the NaNs with zero.
        df['deaths_covid'] = df['deaths_covid'].fillna(0)
        df = df[~df.deaths.isnull()]
        df_pred, model1 = get_location_specific_prediction(df)
        df_pred['loc_name'] = loc
        if is_saving_first_stage:
            save_first_stage(df_pred, model1, result_folder, suffix, loc)
        lst_df.append(df_pred)
    df_all_location = pd.concat(lst_df)
    return df_all_location


def get_location_specific_prediction(df):
    dp1 = DataProcessor(
        col_deaths="deaths",
        col_year="year_start",
        col_dtime="time_start",
        col_covs=["age_name", "sex", "population", "deaths_covid", "location_id"],
        dtime_unit="week"
    )
    year_start = df[dp1.col_year].min()
    tunit_start = df.loc[df[dp1.col_year]==year_start, dp1.col_dtime].min()
    if year_start < 2010:
        year_start = 2010
        tunit_start = 1
    time_start = (year_start, tunit_start)
    df1 = dp1.process(df,
                      time_end=(2020, 8),
                      group_specs={"age_name": ["0 to 125"], "sex": ["all"]},
                      offset_col="population",
                      offset_fun=np.log)
    # Use time_start
    # df1 = dp1.process(df,
    #                   time_start=time_start,
    #                   time_end=(2020, 8),
    #                   group_specs={"age_name": ["0 to 125"], "sex": ["all"]},
    #                   offset_col="population",
    #                   offset_fun=np.log)
    df1['deaths_covid'] = df1['deaths_covid'].fillna(0)
    spline_specs = regmod.utils.SplineSpecs(
        knots=np.linspace(0.0, 1.0, 5),
        degree=3,
        knots_type="rel_domain"
    )
    week_variable = regmod.variable.SplineVariable("week", spline_specs=spline_specs)
    time_variable = regmod.variable.SplineVariable("time", spline_specs=spline_specs)
    variables = [
        SeasonalityModelVariables([week_variable]),
        TimeModelVariables([time_variable])
    ]

    model1 = ExcessMortalityModel(df1, variables)
    model1.run_models()
    df1_pred = dp1.process(df,
                           time_end=(2020, 36),
                           group_specs={"age_name": ["0 to 125"], "sex": ["all"]},
                           offset_col="population",
                           offset_fun=np.log)
    df1_pred = model1.predict(df1_pred)
    df1_pred["death_rate_covid"] = df1_pred.deaths_covid/df1_pred.population
    return df1_pred, model1


def process_first_stage(df1_pred):
    dp = DataProcessor(
            col_deaths="deaths",
            col_year="year",
            col_dtime="week",
            col_covs=["age_name", "sex", "offset_2", 
            "death_rate_covid", "location_id", "loc_name"],
            dtime_unit="week"
        )
    # year_start = df1_pred[dp.col_year].min()
    # tunit_start = df1_pred.loc[df1_pred[dp.col_year]==year_start, dp.col_dtime].min()
    # if year_start < 2010:
    #     year_start = 2010
    #     tunit_start = 1
    # time_start = (year_start, tunit_start)
    df = dp.process(df1_pred, offset_col="offset_2")
    return df


def run_second_stage(df_all_location, covid_variable="death_rate_covid"):
    location_ids = df_all_location.location_id.unique()
    df_location_specific = {}
    df_all_location = process_first_stage(df_all_location)
    for loc in location_ids:
        df_location_specific[loc] = df_all_location.loc[df_all_location.location_id == loc]
    covid_variable = regmod.variable.Variable(covid_variable)
    variables = [ModelVariables([covid_variable])]
    prior_masks = {}
    level_masks = [10.0]

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


def plot_residual(df, result_folder, loc):
    df['week_yr'] = pd.to_datetime(df['year'].astype(str) \
        + " " + df['week'].astype(str) + ' 1', format='%Y %U %w')
    plt.figure(figsize=(12,8))
    plt.plot(df['week_yr'], df['residuals'])
    plt.axhline(y=0,color='k')
    plt.title(loc)
    plt.savefig(os.path.join(result_folder, f"{loc}_residuals.pdf"))


def save_first_stage(df_all_location, model, result_folder, suffix, loc):
    df_all_location['fitted_values'] = np.exp(df_all_location[f"offset_{model.num_models}"])
    df_all_location['residuals'] = df_all_location['fitted_values'] - df_all_location['deaths']
    df_all_location.to_csv(os.path.join(result_folder, f"{loc}_{suffix}.csv"), index=False)
    plot_residual(df_all_location, result_folder, loc)
    df_all_location = df_all_location.drop(columns=['fitted_values', 'residuals'], axis=1)
    return


def save_results(cascade_all_location, cascade_location_specifics, 
    result_folder, suffix, save_model=False, save_prediction=False):
    """Plot model and save results."""
    lst_coef = []
    for cascade_loc in cascade_location_specifics:
        cascade_loc.df['fitted_values'] = np.exp(cascade_loc.df[f"offset_{cascade_loc.model.num_models}"])

        mae = np.mean(np.abs(cascade_loc.df.fitted_values - cascade_loc.df.deaths))
        # Extract coefficients
        coef = cascade_loc.model.results[0]['coefs'][0]
        loc = cascade_loc.df.loc_name.unique()[0]
        df_coef = pd.DataFrame({'loc':[loc], 'coef':[coef], 'mae': [mae]})
        lst_coef.append(df_coef)
        # Plot
        name = f"{loc}_{suffix}"
        title = f"loc: {loc}; MAE: {round(mae)}; coef: {round(coef, 4)}"
        cascade_loc.model.plot_model(folder=result_folder, name=name, title=title)
        if save_prediction:
            cascade_loc.df.to_csv(os.path.join(result_folder, f"{loc}_{suffix}.csv"), 
                index=False)
        # Save location specific model
        if save_model:
            with open(os.path.join(result_folder, 
                f"location_specific_{loc}_{suffix}.pkl"), "wb") as f_write:
                pickle.dump(cascade_loc, f_write)

    df_coef = pd.concat(lst_coef)
    df_coef.sort_values('coef').to_csv(
        os.path.join(result_folder, f"coefs_{suffix}.csv"), index=False
        )


def main():
    # data_folder = "/Users/jiaweihe/Downloads/mortality/data"
    data_folder = "/home/j/temp/jiaweihe/mortality/2020-12-10-12-52/outputs"
    time_stamp = "2020-12-10-12-52"
    level_mask = 10
    use_cov = True
    #["deaths_covid", "deaths_covid_log", "mortality_covid", "mortality_covid_log"]:
    for covid_variable in ["death_rate_covid"]: 
        if not use_cov:
            suffix = "no_covid_cov"
            # result_folder = f"/Users/jiaweihe/Downloads/mortality/results/{suffix}"
            result_folder = f"/ihme/mortality/covid_em_estimate/{time_stamp}/emmodel/{suffix}"
        else:
            suffix = f"{covid_variable}_{level_mask}"
            # result_folder = f"/Users/jiaweihe/Downloads/mortality/results/{covid_variable}_{level_mask}"
            result_folder = f"/ihme/mortality/covid_em_estimate/{time_stamp}/emmodel/{covid_variable}_{level_mask}"

        if not os.path.exists(result_folder):
            os.mkdir(result_folder)

        df_all_location = run_first_stage(data_folder, result_folder, suffix, 
            is_saving_first_stage=False)

        if use_cov:
            cascade_all_location, cascade_location_specifics = \
                run_second_stage(df_all_location, covid_variable=covid_variable)
            save_results(cascade_all_location, cascade_location_specifics, 
                result_folder, suffix=suffix, save_prediction=True)


if __name__ == '__main__':
    main()
