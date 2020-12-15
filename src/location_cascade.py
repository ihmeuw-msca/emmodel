import os
import numpy as np 
import pandas as pd 
import pickle
import regmod 

from emmodel.model import ExcessMortalityModel
from emmodel.cascade import Cascade, CascadeSpecs
from emmodel.variable import ModelVariables, TimeModelVariables, SeasonalityModelVariables
from emmodel.data import DataProcessor


def get_location_specific_prediction(df):
    """Run location specifiic model to get the 
       prediction as offset for next stage."""
    dp = DataProcessor(
        col_deaths='deaths',
        col_year='year_start', 
        col_tunit='week_start',
        col_population='population',
        col_covs=["age_name", "sex", "deaths_covid", 
        "deaths_covid_log", "mortality_covid", "mortality_covid_log"]
        )
    year_start = df[dp.col_year].min()
    tunit_start = df.loc[df[dp.col_year]==year_start, dp.col_tunit].min()
    if year_start < 2010:
        year_start = 2010
        tunit_start = 0
    time_start = (year_start, tunit_start)
    time_end = (2020, 52)
    # import pdb; pdb.set_trace()
    df_all_age = dp.process(df, time_start=time_start, time_end=time_end,
        group_specs={"age_name": ["0 to 125"], "sex": ["all"]})

    # import pdb; pdb.set_trace()
    spline_specs = regmod.utils.SplineSpecs(
        knots=np.linspace(0, 1, 5),
        degree=3,
        knots_type="rel_domain"
        )

    year_variable = regmod.variable.SplineVariable("week", spline_specs=spline_specs)
    time_variable = regmod.variable.SplineVariable("time", spline_specs=spline_specs)

    variables = [
        SeasonalityModelVariables([year_variable]),
        TimeModelVariables([time_variable])
    ]

    emm = ExcessMortalityModel(df_all_age, variables)
    emm.run_models()
    num_models = emm.num_models
    # Save prediction of location specific model as offset_0.
    df_all_age['offset_0'] = df_all_age['offset_{}'.format(emm.num_models)]
    df_all_age = df_all_age.drop([f'offset_{i}' for i in range(1, num_models+1)], axis=1)
    df_all_age['location_id'] = df.location_id.unique()[0]
    # print(df.location_id.unique()[0], df_all_age.deaths_covid.sum())

    # Correlation of covid deaths and the residual of location specific model.
    # df_all_age['resid'] = df_all_age['deaths'] - np.exp(df_all_age['offset_0'])
    # print(df_all_age.corr().loc['resid', 'deaths_covid'])

    return df_all_age


def run_first_stage(data_folder):
    """Read data and run model for each location separately.
       Use the prediction as offset for later stages.
    """
    data_files = os.listdir(data_folder)
    lst_df = []
    for data_file in data_files:
        loc = data_file.split(".csv")[0]
        df = pd.read_csv(os.path.join(data_folder, f"{loc}.csv"))
        # if df.time_unit.unique()[0] == 'week':
        #     import pdb; pdb.set_trace()
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
        # import pdb; pdb.set_trace()
        df['deaths_covid_log'] = df['deaths_covid'].map(lambda x: np.log(x) if x > 0 else 0)
        df['mortality_covid'] = df['deaths_covid']/df['population']
        df['mortality_covid_log'] = df['mortality_covid'].map(lambda x: np.log(x) if x > 0 else 0)
        # Drop rows where deaths data are NaNs.
        df = df[~df.deaths.isnull()]
        # Run model for each location separately to get the prediction as offset 
        df_all_location = get_location_specific_prediction(df)
        df_all_location['loc_name'] = loc
        # import pdb; pdb.set_trace()
        lst_df.append(df_all_location)
    df_all_location = pd.concat(lst_df)
    return df_all_location


def run_second_stage(df_all_location, level_mask, covid_variable="deaths_covid"):
    """Run cascade."""
    # Create children of each location.
    location_ids = df_all_location.location_id.unique()
    df_location_specific = {}
    for loc in location_ids:
        df_location_specific[loc] = df_all_location.loc[df_all_location.location_id == loc]
    covid_variable = regmod.variable.Variable(covid_variable)
    # covid_variable = regmod.variable.Variable("deaths_covid")
    # covid_variable = regmod.variable.Variable("deaths_covid_log")
    # covid_variable = regmod.variable.Variable("mortality_covid")
    # covid_variable = regmod.variable.Variable("mortality_covid_log")
    variables = [ModelVariables([covid_variable])]
    prior_masks = {}
    level_masks = [level_mask]

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
        # import pdb; pdb.set_trace()
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
    # if system == 'cluster':
    #     data_folder = "/home/j/temp/jiaweihe/mortality/2020-12-10-12-52/outputs"
    # result_folder = "/home/j/temp/jiaweihe/mortality/2020-12-10-12-52/results"
    data_folder = "/Users/jiaweihe/Downloads/mortality/data"
    time_stamp = "2020-12-10-12-52"
    level_mask = 2
    result_folder = f"/Users/jiaweihe/Downloads/mortality/results/covid_death_log_{level_mask}"
    # data_folder = "/home/j/temp/jiaweihe/mortality/2020-12-10-12-52/outputs"
    # for level_mask in [level_mask]: #[0.001, 0.01, 0.1, 1, 2]:
    for covid_variable in ["deaths_covid"]: #["deaths_covid", "deaths_covid_log", "mortality_covid", "mortality_covid_log"]:
        # result_folder = f"/home/j/temp/jiaweihe/mortality/2020-12-10-12-52/results/{covid_variable}_{level_mask}"
        suffix = f"level_masks_{level_mask}"
        # result_folder = f"/ihme/mortality/covid_em_estimate/{time_stamp}/emmodel/{suffix}"

        # if not os.path.exists(result_folder):
        #     os.mkdir(result_folder)

        df_all_location = run_first_stage(data_folder)
        cascade_all_location, cascade_location_specifics = \
            run_second_stage(df_all_location, level_mask, covid_variable=covid_variable)
        save_results(cascade_all_location, cascade_location_specifics, 
            result_folder, suffix=suffix, save_prediction=True)
    # import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()

# child.model.models[0].hessian(np.array([1.0]))
# child.model.models[0].gradient(np.array([1.0]))
# child.model.models[0].data.attach_df(child.df)
# child.model.models[0].parameters[0].variables[0].gprior.mean
# child.get_priors()