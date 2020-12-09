"""
Test Data module
"""
import pytest
import numpy as np
import pandas as pd
from emmodel.data import DataProcessor


@pytest.fixture
def df():
    return pd.DataFrame({
        "test_deaths": [5]*104,
        "test_population": [100]*104,
        "test_year": [2019]*52 + [2020]*52,
        "test_week": list(range(1, 53))*2,
        "test_age": [20]*52 + [30]*52,
        "test_sex": ["male"]*52 + ["female"]*52,
        "test_loc": ["Mars"]*104,
        "test_random": ["random"]*104
    })


@pytest.fixture
def data_processor():
    return DataProcessor(
        "test_deaths",
        "test_year",
        "test_week",
        "test_population",
        ["test_age", "test_sex", "test_loc"]
    )


def test_select_cols(df, data_processor):
    df = data_processor.select_cols(df)
    assert all([col in df.columns for col in data_processor.cols])
    assert "test_random" not in df.columns


def test_rename_cols(df, data_processor):
    df = data_processor.select_cols(df)
    df = data_processor.rename_cols(df)
    assert all([col.replace("test_", "") in df.columns
                for col in data_processor.cols
                if col not in data_processor.col_covs])


def test_add_time(df, data_processor):
    df = data_processor.select_cols(df)
    df = data_processor.rename_cols(df)
    df = data_processor.add_time(df, (2019, 1), (2020, 52))
    assert np.allclose(df.time, np.arange(104))


def test_add_offset(df, data_processor):
    df = data_processor.select_cols(df)
    df = data_processor.rename_cols(df)
    df = data_processor.add_time(df, (2019, 1), (2020, 52))
    df = data_processor.add_offset(df, 0, "population")
    assert np.allclose(df["offset_0"], np.log(df["population"]))


def test_subset_group(df, data_processor):
    df = data_processor.select_cols(df)
    df = data_processor.rename_cols(df)
    df = data_processor.add_time(df, (2019, 1), (2020, 52))
    df = data_processor.add_offset(df, 0, "population")
    df = data_processor.subset_group(df, {"test_sex": ["male"]})
    assert all(df["test_sex"] == "male")


def test_process(df, data_processor):
    df = data_processor.process(df, time_end=(2020, 52))
    assert df.shape[0] == 104
