import itertools
from typing import Union, Any

import numpy as np
import pandas as pd
from .constants import YEAR_HOURS, DAY_LENGTH, MAX_HIGH_HOURS


def shift_array(arr, num, fill_value=np.nan):
    """
    shift the array num places to the right

    :param arr: the array
    :param num: number of places to shift
    :param fill_value: a value to fill empty spaces in shifted array (should be a value of the same type

    :return: the shifted array
    """
    result = np.empty_like(arr)
    if fill_value is np.nan and num != 0 and result.dtype != np.float64:
        result = result.astype('float64')
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def year_diff(end_date: pd.DatetimeIndex | pd.Timestamp, start_date: pd.DatetimeIndex | pd.Timestamp):
    """
    calculate the difference in years between 2 dateTime or timestamp (a series and a single date or 2 series)

    :param end_date: the end date
    :param start_date: the start date
    """
    years = end_date.year - start_date.year
    years -= np.where((end_date.month < start_date.month) | ((end_date.month == start_date.month) &
                                                             (end_date.day < start_date.day)), 1, 0)
    return years


def month_diff(end_date: pd.DatetimeIndex | pd.Timestamp, start_date: pd.DatetimeIndex | pd.Timestamp):
    """
    calculate the difference in months between 2 series of dateTime or timestamp of the same length (a series and a
    single date or 2 series)

    :param end_date: the end date
    :param start_date: the start date
    """
    return end_date.month - start_date.month + (end_date.year - start_date.year) * 12 - (end_date.day < start_date.day)


def check_cover_no_overlap(cover_range: Union[list[int], tuple[int, ...]], overlap_error_msg: str,
                           cover_error_msg: str, *args: Union[list[int], tuple[int, ...]]):
    """
    check the given lists/tuples cover the given cover range and has no overlapping elements

    :param cover_range: the range the args should cover
    :param overlap_error_msg: error message when there is overlap
    :param cover_error_msg: error message where lists/tuples doesn't cover
    :param args: a number of lists/tuples
    """
    combined = list(itertools.chain(*args))
    if len(set(combined)) != len(combined):
        raise ValueError(overlap_error_msg)
    if sorted(combined) != cover_range:
        raise ValueError(cover_error_msg)


def _get_hour_division(low_hours, first_high_hour, high_hour_num):
    """
    get the division to low and high number from the given parameters
    :param low_hours: tuple with low hour (or none)
    :param first_high_hour: the first of the ihg hours
    :param high_hour_num: the number of high hours
    """
    if low_hours is None:
        bound_high_hour = (first_high_hour + high_hour_num) % DAY_LENGTH
        if first_high_hour < bound_high_hour:
            low_hours = tuple(range(0, first_high_hour)) + tuple(range(bound_high_hour, DAY_LENGTH))
            high_hours = tuple(range(first_high_hour, bound_high_hour))
        else:
            low_hours = tuple(range(bound_high_hour, first_high_hour))
            high_hours = tuple(range(0, bound_high_hour)) + tuple(range(first_high_hour, DAY_LENGTH))
    else:
        high_hours = tuple(x for x in range(0, DAY_LENGTH) if x not in low_hours)

    return low_hours, high_hours


def get_seasonal_hour_division(winter_low_hours: tuple[int, ...] | None = None,
                               transition_low_hours: tuple[int, ...] | None = None,
                               summer_low_hours: tuple[int, ...] | None = None,
                               first_high_hour: int = 17,
                               high_hour_num_winter: int = 5,
                               high_hour_num_transition: int = 5,
                               high_hour_num_summer: int = 6):
    """
    returns the low and high hours for each season (winter, transition, summer) according to the given parameters
    :param winter_low_hours: hours in the winter when there is a low price (if none determined by discharge start hour)
    :param transition_low_hours: hours in the transition seasons when there is a low price (if none determined by
        discharge start hour)
    :param summer_low_hours: hours in the summer when there is a low price (if none determined by discharge start hour)
    :param first_high_hour: the first hour of with high price
    :param high_hour_num_winter: the number of hours in a day with high price in the winter
    :param high_hour_num_transition: the number of hour in a day with high price in transition seasons
    :param high_hour_num_summer: the number of hours in a day with high prices in the summer
    """
    # check value of parameters
    if winter_low_hours is not None:
        for x in winter_low_hours:
            if not 0 <= x < DAY_LENGTH:
                raise ValueError(f"winter low hour should be between 0 and {DAY_LENGTH}")
    if transition_low_hours is not None:
        for x in transition_low_hours:
            if not 0 <= x < DAY_LENGTH:
                raise ValueError(f"transition low hour should be between 0 and {DAY_LENGTH}")
    if summer_low_hours is not None:
        for x in summer_low_hours:
            if not 0 <= x < DAY_LENGTH:
                raise ValueError(f"summer low hour should be between 0 and {DAY_LENGTH}")
    if not 0 <= first_high_hour < DAY_LENGTH:
        raise ValueError(f"first high hour should be between 0 and {DAY_LENGTH}")
    if not 0 < high_hour_num_winter <= MAX_HIGH_HOURS or not 0 < high_hour_num_transition <= MAX_HIGH_HOURS or \
            not 0 < high_hour_num_summer <= MAX_HIGH_HOURS:
        raise ValueError(f"number of high hours in a day should be between 1 and {MAX_HIGH_HOURS} (inclusive)")

    # create hours division
    winter_low_hours, winter_high_hours = _get_hour_division(winter_low_hours, first_high_hour, high_hour_num_winter)
    transition_low_hours, transition_high_hours = _get_hour_division(transition_low_hours, first_high_hour,
                                                                     high_hour_num_transition)
    summer_low_hours, sumer_high_hours = _get_hour_division(summer_low_hours, first_high_hour, high_hour_num_summer)

    return (winter_low_hours, winter_high_hours, transition_low_hours, transition_high_hours, summer_low_hours,
            sumer_high_hours)


def build_tariff_table(winter_low: float, winter_high_week: float, winter_high_weekend: float, transition_low: float,
                       transition_high_week: float, transition_high_weekend: float, summer_low: float,
                       summer_high_week: float, summer_high_weekend: float,
                       week_days: tuple[int, ...] = (0, 1, 2, 3, 4), winter_months: tuple[int, ...] = (0, 1, 11),
                       transition_months: tuple[int, ...] = (2, 3, 4, 9, 10),
                       summer_months: tuple[int, ...] = (5, 6, 7, 8),
                       winter_low_hours: tuple[int, ...] | None = None,
                       transition_low_hours: tuple[int, ...] | None = None,
                       summer_low_hours: tuple[int, ...] | None = None,
                       first_high_hour: int = 17, high_hour_num_winter: int = 5, high_hour_num_transition: int = 5,
                       high_hour_num_summer: int = 6):
    """
    create a tariff table containg the tariff for each hour of the day in each month using the tariff and prices for
    winter, transition and summer

    :param winter_low: tariff for winter low hours
    :param winter_high_week: tariff for winter high hours in week days
    :param winter_high_weekend: tariff for winter high hours in weekend days
    :param transition_low: tariff for low hours in transition seasons
    :param transition_high_week: tariff for high hours in transition seasons in the week days
    :param transition_high_weekend: tariff for high hours in transition seasons in the weekend days
    :param summer_low: tariff for low hours in summer
    :param summer_high_week: tariff for summer high hour in week days
    :param summer_high_weekend: tariff for summer high hour in weekend days
    :param week_days: regular week days
    :param winter_months: months that are part of winter
    :param transition_months: months that are part of transition seasons
    :param summer_months: months that are part of summer
    :param winter_low_hours: winter hours when the tariff is low
    :param transition_low_hours: transition hours when the tariff is low
    :param summer_low_hours: summer hours when the tariff is low
    :param first_high_hour: the first hour of with high price
    :param high_hour_num_winter: the number of hours in a day with high price in the winter
    :param high_hour_num_transition: the number of hour in a day with high price in transition seasons
    :param high_hour_num_summer: the number of hours in a day with high prices in the summer
    """
    tariff_table = np.zeros((7, 12, 24))

    # check months and hours doesn't overlap and cover every month/hour
    check_cover_no_overlap(list(range(0, 12)), "Seasons should have different months", "Months doesn't cover all year",
                           winter_months, transition_months, summer_months)
    # get hours division for each season
    winter_low_hours, winter_high_hours, transition_low_hours, transition_high_hours, summer_low_hours, \
        summer_high_hours = get_seasonal_hour_division(winter_low_hours, transition_low_hours, summer_low_hours,
                                                       first_high_hour, high_hour_num_winter, high_hour_num_transition,
                                                       high_hour_num_summer)
    # get weekend days
    weekend_days = tuple(x for x in range(0, 7) if x not in week_days)

    # winter tariffs
    tariff_table[np.ix_(week_days, winter_months, winter_low_hours)] = winter_low
    tariff_table[np.ix_(weekend_days, winter_months, winter_low_hours)] = winter_low
    tariff_table[np.ix_(week_days, winter_months, winter_high_hours)] = winter_high_week
    tariff_table[np.ix_(weekend_days, winter_months, winter_high_hours)] = winter_high_weekend
    # transition tariffs
    tariff_table[np.ix_(week_days, transition_months, transition_low_hours)] = transition_low
    tariff_table[np.ix_(weekend_days, transition_months, transition_low_hours)] = transition_low
    tariff_table[np.ix_(week_days, transition_months, transition_high_hours)] = transition_high_week
    tariff_table[np.ix_(weekend_days, transition_months, transition_high_hours)] = transition_high_weekend
    # summer tariffs
    tariff_table[np.ix_(week_days, summer_months, summer_low_hours)] = summer_low
    tariff_table[np.ix_(weekend_days, summer_months, summer_low_hours)] = summer_low
    tariff_table[np.ix_(week_days, summer_months, summer_high_hours)] = summer_high_week
    tariff_table[np.ix_(weekend_days, summer_months, summer_high_hours)] = summer_high_weekend
    return tariff_table


def tariff_table_to_hourly(tariff_table: np.ndarray[Any, np.dtype[np.float64]], year: int):
    """
    Create hourly prices for tariff table in the given year

    :param tariff_table: the tariff table
    :param year: the year to calculate for (4 digits)
    """
    times = pd.date_range(start=f'{year}-01-01 00:00', end=f'{year}-12-31 23:00', freq='h', tz='Asia/Jerusalem')

    def f(x): return tariff_table[(x.day_of_week + 1) % 7, x.month - 1, x.hour]

    return f(times)


def relu(x):
    """
    calc relu (max(x, 0) for numpy array
    :param x: the array
    """
    return x * (x > 0)


def clamp(value, low=0, high=1, old_low=-1, old_high=1):
    """
    clamp value from one range to a different range (preserving proportions)
    :param value: the value to clamp
    :param low: low end of new range
    :param high: high end of new range
    :param old_low: low end of old range
    :param old_high: high range of old range
    :return: the clamped value
    """
    old_range = old_high - old_low
    new_range = high - low
    return np.clip(low + (value - old_low) * new_range / old_range, low, high)


def is_leap_year(year: int):
    """
    return True if the given year is a leap year, otherwise false
    """
    return (year % 400 == 0) or ((year % 100 != 0) and (year % 4 == 0))


def hour_num_in_range(start_year: int, number_of_years: int):
    """
    calculate the number of hours in the year range stating from start year with length number of years (accounting for
    leap years)
    :param start_year: the year the range start in
    :param number_of_years: the number of years in the range
    :return: the total number of hours in range and a list with te number of hours in each year
    """
    # get hours without accounting for leap years
    hour_num = [YEAR_HOURS] * number_of_years
    # get first year that is divisible by 4
    first_potential_leap_year = start_year if start_year % 4 == 0 else start_year + 4 - (start_year % 4)
    # add hour for years that are leap years
    for y in range(first_potential_leap_year, start_year + number_of_years, 4):
        if is_leap_year(y):
            hour_num[y - start_year] += 24
    return sum(hour_num), hour_num


def get_yearly_prices(year, sell_prices, buy_prices, tariff_table, start_year, yearly_hour_num):
    """
    get prices in current year according to init inputs
    :param year: the year to calculate for (starting at 0 for the first year of simulation)
    :param sell_prices: array with prices for selling power
    :param buy_prices: array with prices for buying power
    :param tariff_table: array with a tariff table of prices for every hour in each day of the week in every month
    :param start_year: the first year of the project
    :param yearly_hour_num: number of hours in each year
    """
    if year > len(yearly_hour_num) or year < 0:
        raise ValueError("Year should be in range of number of years of the project (string at 0)")
    if sell_prices is None:
        sell_output = tariff_table_to_hourly(tariff_table, start_year + year)
        buy_output = sell_output
    elif sell_prices.shape == (YEAR_HOURS,):
        if is_leap_year(start_year + year):
            sell_output = np.concatenate((sell_prices, sell_prices[-DAY_LENGTH:]))
            buy_output = np.concatenate((buy_prices, buy_prices[-DAY_LENGTH:]))
        else:
            sell_output = sell_prices
            buy_output = buy_prices
    else:
        year_first_hour = sum(yearly_hour_num[:year])
        year_last_hour = yearly_hour_num[year] + year_first_hour
        sell_output = sell_prices[year_first_hour: year_last_hour]
        buy_output = buy_prices[year_first_hour: year_last_hour]
    return sell_output, buy_output


def is_real_numbers(arr: np.ndarray[Any, Any]):
    """
    check if the dtype of the given array is a type corresponding to real numbers
    :param arr: the numpy array
    """
    return np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer)
