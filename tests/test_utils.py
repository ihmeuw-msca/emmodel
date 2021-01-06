"""
Test utility module
"""
import pytest
from emmodel.utils import YearTime


@pytest.mark.parametrize("year", [-2010])
def test_year_validate(year):
    with pytest.raises(AssertionError):
        YearTime(year, 0)


@pytest.mark.parametrize("time", [-10])
def test_time_validate(time):
    with pytest.raises(AssertionError):
        YearTime(0, time)


@pytest.mark.parametrize("time_unit", ["day", "year"])
def test_time_unit_validate(time_unit):
    with pytest.raises(AssertionError):
        YearTime(0, 0, time_unit=time_unit)


@pytest.mark.parametrize("yt0", [YearTime(2010, 0)])
@pytest.mark.parametrize(("yt1", "result"),
                         [(YearTime(2010, 10), 10),
                          (YearTime(2009, 10), -42)])
def test_sub(yt0, yt1, result):
    assert (yt1 - yt0) == result


@pytest.mark.parametrize("yt0", [YearTime(2010, 0)])
@pytest.mark.parametrize(("yt1", "result"),
                         [(YearTime(2009, 51), True),
                          (YearTime(2009, 52), False),
                          (YearTime(2010, 1), False)])
def test_lt(yt0, yt1, result):
    assert (yt1 < yt0) == result


@pytest.mark.parametrize("yt0", [YearTime(2010, 0)])
@pytest.mark.parametrize(("yt1", "result"),
                         [(YearTime(2009, 51), True),
                          (YearTime(2009, 52), True),
                          (YearTime(2010, 1), False)])
def test_le(yt0, yt1, result):
    assert (yt1 <= yt0) == result


@pytest.mark.parametrize("yt0", [YearTime(2010, 0)])
@pytest.mark.parametrize(("yt1", "result"),
                         [(YearTime(2009, 51), False),
                          (YearTime(2009, 52), False),
                          (YearTime(2010, 1), True)])
def test_gt(yt0, yt1, result):
    assert (yt1 > yt0) == result


@pytest.mark.parametrize("yt0", [YearTime(2010, 0)])
@pytest.mark.parametrize(("yt1", "result"),
                         [(YearTime(2009, 51), False),
                          (YearTime(2009, 52), True),
                          (YearTime(2010, 1), True)])
def test_ge(yt0, yt1, result):
    assert (yt1 >= yt0) == result


@pytest.mark.parametrize("yt0", [YearTime(2010, 0)])
@pytest.mark.parametrize(("yt1", "result"),
                         [(YearTime(2010, 0), True),
                          (YearTime(2009, 52), True),
                          (YearTime(2010, 1), False)])
def test_eq(yt0, yt1, result):
    assert (yt1 == yt0) == result


@pytest.mark.parametrize("yt", [YearTime(2010, 1)])
@pytest.mark.parametrize(("key", "value"),
                         [(0, 2010),
                          ("year", 2010),
                          (1, 1),
                          ("time", 1)])
def test_getitem(yt, key, value):
    assert yt[key] == value
