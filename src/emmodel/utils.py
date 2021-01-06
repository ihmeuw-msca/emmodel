"""
Utility module
"""
from typing import Union


UNITS_PER_YEAR = {
    "week": 52,
    "month": 12
}


class YearTime:
    """
    Year time class, contains year and detailed time information, used for
    computation of time axis

    Attributes
    ----------
    year : int
        Year variable.
    time : int
        Detailed time variable.
    time_unit : str
        Time unit, options are "week" and "month".
    units_per_year : int
        Number of time units per year.
    """

    def __init__(self, year: int, time: int, time_unit: str = "week"):
        """
        Parameters
        ----------
        year : int
            Year variable, required to be a non-negative integer.
        time : int
            Detailed time variable, required to be a non-negative integer.
        time_unit : str, optional
            Time unit, options are "week" and "month", by default "week"
        """
        year = int(year)
        time = int(time)
        assert year >= 0, "`year` has to be a non-negative integer."
        assert time >= 0, "`time` has to be a non-negative integer."
        assert time_unit in ["week", "month"], \
            "`time_unit` has to be selected from 'week' or 'month'."

        self.time_unit = time_unit
        self.units_per_year = UNITS_PER_YEAR[self.time_unit]
        self.year = year + (time - 1) // self.units_per_year
        self.time = 1 + (time - 1) % self.units_per_year

    def _validate_other(self, other: "YearTime"):
        if not isinstance(other, YearTime):
            raise TypeError("Can only operate with instance of `YearTime`.")
        if self.time_unit != other.time_unit:
            raise ValueError("Can only compare when have same `time_unit`.")

    def __sub__(self, other: "YearTime") -> int:
        self._validate_other(other)
        return (self.year - other.year)*self.units_per_year + (self.time - other.time)

    def __lt__(self, other: "YearTime") -> bool:
        self._validate_other(other)
        return (self.year < other.year) or ((self.year == other.year) and
                                            (self.time < other.time))

    def __le__(self, other: "YearTime") -> bool:
        self._validate_other(other)
        return (self.year < other.year) or ((self.year == other.year) and
                                            (self.time <= other.time))

    def __gt__(self, other: "YearTime") -> bool:
        self._validate_other(other)
        return (self.year > other.year) or ((self.year == other.year) and
                                            (self.time > other.time))

    def __ge__(self, other: "YearTime") -> bool:
        self._validate_other(other)
        return (self.year > other.year) or ((self.year == other.year) and
                                            (self.time >= other.time))

    def __eq__(self, other: "YearTime") -> bool:
        self._validate_other(other)
        return (self.year == other.year) and (self.time == other.time)

    def __getitem__(self, key: Union[int, str]) -> int:
        if key in [0, "year"]:
            value = self.year
        elif key in [1, "time"]:
            value = self.time
        else:
            raise ValueError("Invalid key for `YearTime`.")
        return value

    def __repr__(self) -> str:
        return f"YearTime(year={self.year}, time={self.time}, time_unit={self.time_unit})"
