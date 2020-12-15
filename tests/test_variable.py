"""
Test variable module
"""
import pytest
import numpy as np
import pandas as pd
import regmod
from emmodel.data import DataProcessor
from emmodel.variable import SeasonalityModelVariables, TimeModelVariables


@pytest.fixture
def seasonality_model_variables():
    variables = [
        regmod.variable.SplineVariable("week",
                                       spline_specs=regmod.utils.SplineSpecs(knots=np.linspace(0.0, 1.0, 5),
                                                                             degree=3))
    ]
    return SeasonalityModelVariables(variables)


@pytest.fixture
def time_model_variables():
    variables = [
        regmod.variable.Variable("time"),
        regmod.variable.Variable("covid_deaths")
    ]
    return TimeModelVariables(variables)


@pytest.fixture
def data():
    df = pd.DataFrame({
        "deaths": [5]*104,
        "population": [100]*104,
        "year": [2019]*52 + [2020]*52,
        "week": list(range(1, 53))*2,
        "time": list(range(1, 105)),
        "offset_0": [2]*104,
        "age": [20]*52 + [30]*52,
        "sex": ["male"]*52 + ["female"]*52,
        "loc": ["Mars"]*104,
        "covid_deaths": [0]*104
    })

    return regmod.data.Data(
        col_obs="deaths",
        col_covs=["covid_deaths", "time", "week"],
        col_offset="offset_0",
        df=df
    )


def test_seasonality_model_variables(data, seasonality_model_variables):
    model = seasonality_model_variables.get_model(data)
    assert isinstance(model, regmod.model.PoissonModel)


def test_time_model_variables(data, time_model_variables):
    model = time_model_variables.get_model(data)
    assert isinstance(model, regmod.model.PoissonModel)
