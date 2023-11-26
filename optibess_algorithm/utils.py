import itertools
from typing import Union

import numpy as np
import pandas as pd


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


def _check_cover_no_overlap(cover_range: Union[list[int], tuple[int, ...]], overlap_error_msg: str,
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


def build_tariff_table(
        winter_months: Union[list[int], tuple[int, ...]] = (0, 1, 11),
        transition_months: Union[list[int], tuple[int, ...]] = (2, 3, 4, 9, 10),
        summer_months: Union[list[int], tuple[int, ...]] = (5, 6, 7, 8),
        winter_low_hours: Union[list[int], tuple[int, ...]] = tuple(range(1, 17)) + (22, 23, 0),
        winter_high_hours: Union[list[int], tuple[int, ...]] = tuple(range(17, 22)),
        transition_low_hours: Union[list[int], tuple[int, ...]] = tuple(range(1, 17)) + (22, 23, 0),
        transition_high_hours: Union[list[int], tuple[int, ...]] = tuple(range(17, 22)),
        summer_low_hours: Union[list[int], tuple[int, ...]] = tuple(range(1, 17)) + (23, 0),
        summer_high_hours: Union[list[int], tuple[int, ...]] = tuple(range(17, 23)),
):
    """
    create a tariff table containing the tariff in each hour for each month
    :param winter_months: months considered winter
    :param transition_months: months considered transition
    :param summer_months: months considered summer
    :param winter_low_hours: winter hours with low tariff
    :param winter_high_hours: winter hours with high tariff
    :param transition_low_hours: transition hours with low tariff
    :param transition_high_hours: transition hours with high tariff
    :param summer_low_hours: summer hours with low tariff
    :param summer_high_hours: summer hours with high tariff
    """
    "Seasons should have different months"
    # check months and hours doesn't overlap and cover every month/hour
    _check_cover_no_overlap(list(range(0, 12)), "Seasons should have different months", "Months doesn't cover all year",
                            winter_months, transition_months, summer_months)
    _check_cover_no_overlap(list(range(0, 24)), "Winter low and high hours should not overlap",
                            "Winter low and high hours doesn't cover the day", winter_low_hours, winter_high_hours)
    _check_cover_no_overlap(list(range(0, 24)), "Transition low and high hours should not overlap",
                            "Transition low and high hours doesn't cover the day", transition_low_hours,
                            transition_high_hours)
    _check_cover_no_overlap(list(range(0, 24)), "Summer low and high hours should not overlap",
                            "Summer low and high hours doesn't cover the day", summer_low_hours, summer_high_hours)

    tariff_table = np.zeros((12, 24))
    # winter tariffs
    tariff_table[np.ix_(winter_months, winter_low_hours)] = 1
    tariff_table[np.ix_(winter_months, winter_high_hours)] = 2
    # transition tariffs
    tariff_table[np.ix_(transition_months, transition_low_hours)] = 3
    tariff_table[np.ix_(transition_months, transition_high_hours)] = 4
    # summer tariffs
    tariff_table[np.ix_(summer_months, summer_low_hours)] = 5
    tariff_table[np.ix_(summer_months, summer_high_hours)] = 6

    return tariff_table.tolist()
