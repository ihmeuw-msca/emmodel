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
        # df = df.dropna()
        df = df[~df.deaths.isnull()]
        # if df.location_id.unique()[0] == 48:
        #     import pdb; pdb.set_trace()
        df_pred, model1 = get_location_specific_prediction(df)
        df_pred['loc_name'] = loc
        if is_saving_first_stage:
            save_first_stage(df_pred, model1, result_folder, suffix, loc)
        lst_df.append(df_pred)
        # import pdb;pdb.set_trace()
    # import pdb;pdb.set_trace()
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
    # time_end = (2020, 52)
    # print(time_start)
    # import pdb; pdb.set_trace()
    df1 = dp1.process(df,
                      time_end=(2020, 8),
                      group_specs={"age_name": ["0 to 125"], "sex": ["all"]},
                      offset_col="population",
                      offset_fun=np.log)
    # df1 = dp1.process(df,
    #                   time_start=time_start,
    #                   time_end=(2020, 8),
    #                   group_specs={"age_name": ["0 to 125"], "sex": ["all"]},
    #                   offset_col="population",
    #                   offset_fun=np.log)
    # import pdb; pdb.set_trace()
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
    # Drop NaNs; otherwise run_models would give an error
    # df1 = df1.dropna()
    model1 = ExcessMortalityModel(df1, variables)
    try:
        model1.run_models()
    except:
        import pdb; pdb.set_trace()
    # df = df.dropna()
    df1_pred = dp1.process(df,
                           time_end=(2020, 36),
                           group_specs={"age_name": ["0 to 125"], "sex": ["all"]},
                           offset_col="population",
                           offset_fun=np.log)
    df1_pred = model1.predict(df1_pred)
    df1_pred["death_rate_covid"] = df1_pred.deaths_covid/df1_pred.population
    # import pdb; pdb.set_trace()
    return df1_pred, model1


def process_first_stage(df1_pred):
    # offset_last = f"offset_2"
    dp = DataProcessor(
            col_deaths="deaths",
            col_year="year",
            col_dtime="week",
            col_covs=["age_name", "sex", "offset_2", 
            "death_rate_covid", "location_id", "loc_name"],
            dtime_unit="week"
        )
    # import pdb; pdb.set_trace()
    # year_start = df1_pred[dp.col_year].min()
    # tunit_start = df1_pred.loc[df1_pred[dp.col_year]==year_start, dp.col_dtime].min()
    # if year_start < 2010:
    #     year_start = 2010
    #     tunit_start = 1
    # time_start = (year_start, tunit_start)
    df = dp.process(df1_pred, offset_col="offset_2")
    return df


def run_second_stage(df_all_location, covid_variable="death_rate_covid"):
    # import pdb; pdb.set_trace()
    location_ids = df_all_location.location_id.unique()
    df_location_specific = {}
    df_all_location = process_first_stage(df_all_location)
    for loc in location_ids:
        # tmp = df_all_location.loc[df_all_location.location_id == loc]
        # df_location_specific[loc] = process_first_stage(tmp)
        df_location_specific[loc] = df_all_location.loc[df_all_location.location_id == loc]
    # import pdb; pdb.set_trace()
    covid_variable = regmod.variable.Variable(covid_variable)
    # covid_variable = regmod.variable.Variable(covid_variable)
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
    # import pdb; pdb.set_trace()

    try:
        cascade_all_location.add_children(cascade_location_specifics)
        cascade_all_location.run_models()
    except:
        # cascade_all_location.add_children(cascade_location_specifics[:19])
        cascade_all_location.run_models()
        # import pdb; pdb.set_trace()
        # for ix in range(len(cascade_location_specifics)):
        #     child = cascade_location_specifics[ix]
        #     try:
        #         child.model.models[0].data.attach_df(child.df)
        #         print(child.model.models[0].gradient(np.array([1.0])))
        #     except: 
        #         continue
                # import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    return cascade_all_location, cascade_location_specifics


# def run_second_stage(df1_pred):
#     dp2 = DataProcessor(
#         col_deaths="deaths",
#         col_year="year",
#         col_dtime="week",
#         col_covs=["age_name", "sex", "offset_2", "death_rate_covid"],
#         dtime_unit="week"
#     )
#     df2 = dp2.process(df1_pred,
#                       offset_col="offset_2")
#     covid_variable = Variable("death_rate_covid")
#     model2 = ExcessMortalityModel(df2, [ModelVariables([covid_variable])])
#     model2.run_models()
#     model2.plot_model(folder=result_folder, name=name, title=title)


# def get_location_specific_prediction(df):
#     """Run location specifiic model to get the 
#        prediction as offset for next stage."""
#     dp = DataProcessor(
#         col_deaths='deaths',
#         col_year='year_start', 
#         dtime_unit='week',
#         col_dtime='time_start',
#         col_covs=["population", "age_name", "sex", "deaths_covid", 
#         "deaths_covid_log", "mortality_covid", "mortality_covid_log"]
#         )
#     year_start = df[dp.col_year].min()
#     # import pdb; pdb.set_trace()
#     tunit_start = df.loc[df[dp.col_year]==year_start, dp.col_dtime].min()
#     if year_start < 2010:
#         year_start = 2010
#         tunit_start = 0
#     time_start = (year_start, tunit_start)
#     time_end = (2020, 52)
#     # import pdb; pdb.set_trace()
#     df_all_age = dp.process(df, time_start=time_start, time_end=time_end, 
#         offset_col='population', offset_fun=np.log, 
#         group_specs={"age_name": ["0 to 125"], "sex": ["all"]})

#     # import pdb; pdb.set_trace()
#     spline_specs = regmod.utils.SplineSpecs(
#         knots=np.linspace(0, 1, 5),
#         degree=3,
#         knots_type="rel_domain"
#         )

#     year_variable = regmod.variable.SplineVariable("week", spline_specs=spline_specs)
#     time_variable = regmod.variable.SplineVariable("time", spline_specs=spline_specs)

#     variables = [
#         SeasonalityModelVariables([year_variable]),
#         TimeModelVariables([time_variable])
#     ]

#     emm = ExcessMortalityModel(df_all_age, variables)
#     emm.run_models()
#     num_models = emm.num_models
#     # Save prediction of location specific model as offset_0.
#     df_all_age['offset_0'] = df_all_age['offset_{}'.format(emm.num_models)]
#     df_all_age = df_all_age.drop([f'offset_{i}' for i in range(1, num_models+1)], axis=1)
#     df_all_age['location_id'] = df.location_id.unique()[0]
#     # print(df.location_id.unique()[0], df_all_age.deaths_covid.sum())

#     # Correlation of covid deaths and the residual of location specific model.
#     # df_all_age['resid'] = df_all_age['deaths'] - np.exp(df_all_age['offset_0'])
#     # print(df_all_age.corr().loc['resid', 'deaths_covid'])

#     return df_all_age


def plot_residual(df, result_folder, loc):
    # cats['week_yr'] = pd.to_datetime(cats['year'].astype(str) + ' ' + cats['week'].astype(str) + ' 1',
                                # format='%Y %U %w')
    df['week_yr'] = pd.to_datetime(df['year'].astype(str) \
        + " " + df['week'].astype(str) + ' 1', format='%Y %U %w')
    plt.figure(figsize=(12,8))
    plt.plot(df['week_yr'], df['residuals'])
    plt.axhline(y=0,color='k')
    # df.plot.line(x='week_yr', y='residuals')
    # df.plot.scatter(x='week_yr', y='residuals')
    plt.title(loc)
    plt.savefig(os.path.join(result_folder, f"{loc}_residuals.pdf"))


def save_first_stage(df_all_location, model, result_folder, suffix, loc):
    df_all_location['fitted_values'] = np.exp(df_all_location[f"offset_{model.num_models}"])
    df_all_location['residuals'] = df_all_location['fitted_values'] - df_all_location['deaths']
    df_all_location.to_csv(os.path.join(result_folder, f"{loc}_{suffix}.csv"), index=False)
    plot_residual(df_all_location, result_folder, loc)
    # import pdb; pdb.set_trace()
    df_all_location = df_all_location.drop(columns=['fitted_values', 'residuals'], axis=1)
    return


# def run_first_stage(data_folder, result_folder, suffix, is_saving_first_stage=False):
#     """Read data and run model for each location separately.
#        Use the prediction as offset for later stages.
#     """
#     data_files = os.listdir(data_folder)
#     lst_df = []
#     for data_file in data_files:
#         loc = data_file.split(".csv")[0]
#         if loc in ['AUS', 'BRA', 'NZL']:
#             continue
#         df = pd.read_csv(os.path.join(data_folder, f"{loc}.csv"))
#         # if df.time_unit.unique()[0] == 'week':
#         #     import pdb; pdb.set_trace()
#         time_unit = df.time_unit.unique()
#         # import pdb; pdb.set_trace()
#         if len(time_unit) == 1:
#             # Process locations with weekly data for now
#             if time_unit[0] == 'month':
#                 continue
#                 # df = df.rename(columns={'time_start': 'week_start'})
#             # Deal with monthly data later
#     #         elif time_unit[0] == 'month':
#     #             df = df.rename(columns={'time_start': 'month_start'})
#             # else:
#             #     continue
#         # Fill in the NaNs with zero.
#         df['deaths_covid'] = df['deaths_covid'].fillna(0)
#         # import pdb; pdb.set_trace()
#         df['deaths_covid_log'] = df['deaths_covid'].map(lambda x: np.log(x) if x > 0 else 0)
#         df['mortality_covid'] = df['deaths_covid']/df['population']
#         df['mortality_covid_log'] = df['mortality_covid'].map(lambda x: np.log(x) if x > 0 else 0)
#         # Drop rows where deaths data are NaNs.
#         df = df[~df.deaths.isnull()]
#         # Run model for each location separately to get the prediction as offset 
#         df_all_location = get_location_specific_prediction(df)
#         df_all_location['loc_name'] = loc
#         if is_saving_first_stage:
#             save_first_stage(df_all_location, result_folder, suffix, loc)
#         lst_df.append(df_all_location)
#     df_all_location = pd.concat(lst_df)
#     return df_all_location


# def run_second_stage(df_all_location, level_mask, covid_variable="deaths_covid"):
#     """Run cascade."""
#     # Create children of each location.
#     location_ids = df_all_location.location_id.unique()
#     df_location_specific = {}
#     for loc in location_ids:
#         df_location_specific[loc] = df_all_location.loc[df_all_location.location_id == loc]
#     covid_variable = regmod.variable.Variable(covid_variable)
#     # covid_variable = regmod.variable.Variable("deaths_covid")
#     # covid_variable = regmod.variable.Variable("deaths_covid_log")
#     # covid_variable = regmod.variable.Variable("mortality_covid")
#     # covid_variable = regmod.variable.Variable("mortality_covid_log")
#     variables = [ModelVariables([covid_variable])]
#     prior_masks = {}
#     level_masks = [level_mask]

#     specs = CascadeSpecs(variables,
#                          prior_masks=prior_masks,
#                          level_masks=level_masks)

#     # create level 0 model
#     cascade_all_location = Cascade(df_all_location, specs, level_id=0)

#     # create level 1 model
#     cascade_location_specifics = [
#         Cascade(df_location_specific[location_group], specs, level_id=1)
#         for location_group in location_ids
#     ]

#     cascade_all_location.add_children(cascade_location_specifics)
#     cascade_all_location.run_models()
#     return cascade_all_location, cascade_location_specifics


def save_results(cascade_all_location, cascade_location_specifics, 
    result_folder, suffix, save_model=False, save_prediction=False):
    """Plot model and save results."""
    lst_coef = []
    for cascade_loc in cascade_location_specifics:
        # import pdb; pdb.set_trace()
        cascade_loc.df['fitted_values'] = np.exp(cascade_loc.df[f"offset_{cascade_loc.model.num_models}"])
        # import pdb;pdb.set_trace()

        dtimes_per_year = 52
        dtime_unit = 'week'
        year_start = cascade_loc.df['year'].min()
        tunit_start = cascade_loc.df.loc[cascade_loc.df['year']==year_start, dtime_unit].min()
        if year_start < 2010:
            year_start = 2010
            tunit_start = 1
        time_start = (year_start, tunit_start)
        time_end = (2020, 36)
        cascade_loc.df["time"] = (cascade_loc.df["year"] - time_start[0])*dtimes_per_year + \
            (cascade_loc.df[dtime_unit] - time_start[1]) + 1
        time_lb = 1
        time_ub = (time_end[0] - time_start[0])*dtimes_per_year + \
            (time_end[1] - time_start[1]) + 1
        # import pdb;pdb.set_trace()
        cascade_loc.df = cascade_loc.df[(cascade_loc.df["time"] >= time_lb) & (cascade_loc.df["time"] <= time_ub)]

        mae = np.mean(np.abs(cascade_loc.df.fitted_values - cascade_loc.df.deaths))
        # Extract coefficients
        coef = cascade_loc.model.results[0]['coefs'][0]
        loc = cascade_loc.df.loc_name.unique()[0]
        df_coef = pd.DataFrame({'loc':[loc], 'coef':[coef], 'mae': [mae]})
        lst_coef.append(df_coef)
        # Plot
        name = f"{loc}_{suffix}"
        title = f"loc: {loc}; MAE: {round(mae)}; coef: {round(coef, 4)}"
        # import pdb; pdb.set_trace()
        cascade_loc.model.plot_model(folder=result_folder, name=name, title=title)
        if save_prediction:
            cascade_loc.df.to_csv(os.path.join(result_folder, f"{loc}_{suffix}.csv"), 
                index=False)
        # Save location specific model
        if save_model:
            with open(os.path.join(result_folder, 
                f"location_specific_{loc}_{suffix}.pkl"), "wb") as f_write:
                pickle.dump(cascade_loc, f_write)

    # Save overall model
    # with open(os.path.join(result_folder, f"all_location_{suffix}.pkl"), "wb") as f_write:
    #     pickle.dump(cascade_all_location, f_write)

    df_coef = pd.concat(lst_coef)
    df_coef.sort_values('coef').to_csv(
        os.path.join(result_folder, f"coefs_{suffix}.csv"), index=False
        )


def main():
    data_folder = "/Users/jiaweihe/Downloads/mortality/data"
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
        # import pdb; pdb.set_trace()
        print(df_all_location.columns)
        if use_cov:
            cascade_all_location, cascade_location_specifics = \
                run_second_stage(df_all_location, covid_variable=covid_variable)
            save_results(cascade_all_location, cascade_location_specifics, 
                result_folder, suffix=suffix, save_prediction=True)


if __name__ == '__main__':
    main()

# child = cascade_location_specifics[0]
# child.model.models[0].data.attach_df(child.df)
# child.model.models[0].hessian(np.array([1.0]))
# child.model.models[0].gradient(np.array([1.0]))
# child.model.models[0].parameters[0].variables[0].gprior.mean
# child.get_priors()