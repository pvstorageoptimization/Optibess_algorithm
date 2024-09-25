import os
import unittest
from numpy import testing as nptesting
import pandas as pd
import numpy as np

from optibess_algorithm.utils import (shift_array, year_diff, month_diff, build_tariff_table, tariff_table_to_hourly,
                                      relu, clamp, is_leap_year, hour_num_in_range, get_yearly_prices,
                                      get_seasonal_hour_division, is_real_numbers)
from optibess_algorithm.constants import DAY_LENGTH, MAX_HIGH_HOURS

test_folder = os.path.dirname(os.path.abspath(__file__))


class TestShiftArray(unittest.TestCase):

    def test_shift_array_positive(self):
        # call function
        result = shift_array([0, 1, 2, 3, 4, 5], 2)
        # check output
        nptesting.assert_array_equal(result, [np.nan, np.nan, 0, 1, 2, 3])

    def test_shift_array_non_numeric_values_no_fill_value(self):
        # check error raised
        with self.assertRaises(ValueError):
            shift_array(["a", "b", "c", "d"], 1)

    def test_shift_array_non_numeric_values_num_zero(self):
        # call function
        result = shift_array(["a", "b", "c", "d"], 0)
        # check output
        nptesting.assert_array_equal(result, ["a", "b", "c", "d"])

    def test_shift_array_non_numeric_values_with_fill_value(self):
        # call function
        result = shift_array(["a", "b", "c", "d"], 1, "C")
        # check output
        nptesting.assert_array_equal(result, ["C", "a", "b", "c"])

    def test_shift_array_negative(self):
        # call function
        result = shift_array([0, 1, 2, 3, 4, 5], -3)
        # check output
        nptesting.assert_array_equal(result, [3, 4, 5, np.nan, np.nan, np.nan])


class TestYearDiff(unittest.TestCase):

    def test_year_diff_single_values(self):
        # call function
        result = year_diff(pd.to_datetime("2025-5-1"), pd.to_datetime("2023-1-1"))
        # check output
        self.assertEqual(result, 2)

    def test_year_diff_series_single_value(self):
        # call function
        result = year_diff(pd.to_datetime("2025-6-5"), pd.date_range("2023-5-1", "2023-5-10"))
        # check output
        nptesting.assert_array_equal(result, (2, 2, 2, 2, 2, 2, 2, 2, 2, 2))

    def test_year_diff_2_series(self):
        # call function
        result = year_diff(pd.date_range("2025-6-2", "2025-6-5"), pd.date_range("2023-5-1", "2023-5-4"))
        # check output
        nptesting.assert_array_equal(result, (2, 2, 2, 2))

    def test_year_diff_single_value_series(self):
        # call function
        result = year_diff(pd.date_range("2026-7-30", "2026-8-3"), pd.to_datetime("2023-7-31"))
        # check output
        nptesting.assert_array_equal(result, (2, 3, 3, 3, 3))

    def test_year_diff_earlier_month(self):
        # call function
        result = year_diff(pd.to_datetime("2027-6-1"), pd.to_datetime("2023-7-1"))
        # check output
        self.assertEqual(result, 3)

    def test_year_diff_earlier_day(self):
        # call function
        result = year_diff(pd.to_datetime("2024-9-2"), pd.to_datetime("2023-9-5"))
        # check output
        self.assertEqual(result, 0)


class TestMonthDiff(unittest.TestCase):

    def test_month_diff_single_values(self):
        # call function
        result = month_diff(pd.to_datetime("2025-4-5"), pd.to_datetime("2023-1-1"))
        # check output
        self.assertEqual(result, 27)

    def test_month_diff_series_single_value(self):
        # call function
        result = month_diff(pd.to_datetime("2024-7-4"), pd.date_range("2023-7-1", "2023-7-3"))
        # check output
        nptesting.assert_array_equal(result, (12, 12, 12))

    def test_month_diff_2_series(self):
        # call function
        result = month_diff(pd.date_range("2025-10-20", "2025-10-24"), pd.date_range("2023-8-23", "2023-8-27"))
        # check output
        nptesting.assert_array_equal(result, (25, 25, 25, 25, 25))

    def test_month_diff_single_value_series(self):
        # call function
        result = month_diff(pd.date_range("2025-3-2", "2025-3-5"), pd.to_datetime("2023-3-3"))
        # check output
        nptesting.assert_array_equal(result, (23, 24, 24, 24))

    def test_month_diff_earlier_month(self):
        # call function
        result = month_diff(pd.to_datetime("2025-9-25"), pd.to_datetime("2023-10-20"))
        # check output
        self.assertEqual(result, 23)

    def test_month_diff_earlier_day(self):
        # call function
        result = month_diff(pd.to_datetime("2027-3-15"), pd.to_datetime("2023-2-19"))
        # check output
        self.assertEqual(result, 48)


class TestGetSeasonalHourDivision(unittest.TestCase):

    def test_regular(self):
        results = get_seasonal_hour_division()
        self.assertEqual(results[0], tuple(range(0, 17)) + (22, 23))
        self.assertEqual(results[1], tuple(range(17, 22)))
        self.assertEqual(results[2], tuple(range(0, 17)) + (22, 23))
        self.assertEqual(results[3], tuple(range(17, 22)))
        self.assertEqual(results[4], tuple(range(0, 17)) + (23,))
        self.assertEqual(results[5], tuple(range(17, 23)))

    def test_low_hours_given(self):
        results = get_seasonal_hour_division(tuple(range(0, 18)), tuple(range(0, 19)), tuple(range(0, 16)))
        self.assertEqual(results[0], tuple(range(0, 18)))
        self.assertEqual(results[1], tuple(range(18, 24)))
        self.assertEqual(results[2], tuple(range(0, 19)))
        self.assertEqual(results[3], tuple(range(19, 24)))
        self.assertEqual(results[4], tuple(range(0, 16)))
        self.assertEqual(results[5], tuple(range(16, 24)))

    def test_first_high_hour(self):
        results = get_seasonal_hour_division(first_high_hour=19)
        self.assertEqual(results[0], tuple(range(0, 19)))
        self.assertEqual(results[1], tuple(range(19, 24)))
        self.assertEqual(results[2], tuple(range(0, 19)))
        self.assertEqual(results[3], tuple(range(19, 24)))
        self.assertEqual(results[4], tuple(range(1, 19)))
        self.assertEqual(results[5], (0,) + tuple(range(19, 24)))

    def test_high_hours_num(self):
        results = get_seasonal_hour_division(high_hour_num_winter=4, high_hour_num_transition=3,
                                             high_hour_num_summer=8)
        self.assertEqual(results[0], tuple(range(0, 17)) + tuple(range(21, 24)))
        self.assertEqual(results[1], tuple(range(17, 21)))
        self.assertEqual(results[2], tuple(range(0, 17)) + tuple(range(20, 24)))
        self.assertEqual(results[3], tuple(range(17, 20)))
        self.assertEqual(results[4], tuple(range(1, 17)))
        self.assertEqual(results[5], (0,) + tuple(range(17, 24)))

    def test_out_of_range_winter_low_hours(self):
        with self.assertRaises(ValueError) as e:
            get_seasonal_hour_division(winter_low_hours=(25,))
        self.assertEqual(str(e.exception), f"winter low hour should be between 0 and {DAY_LENGTH}")

    def test_out_of_range_transition_low_hours(self):
        with self.assertRaises(ValueError) as e:
            get_seasonal_hour_division(transition_low_hours=(25,))
        self.assertEqual(str(e.exception), f"transition low hour should be between 0 and {DAY_LENGTH}")

    def test_out_of_range_summer_low_hours(self):
        with self.assertRaises(ValueError) as e:
            get_seasonal_hour_division(summer_low_hours=(25,))
        self.assertEqual(str(e.exception), f"summer low hour should be between 0 and {DAY_LENGTH}")

    def test_out_of_range_first_high_hour(self):
        with self.assertRaises(ValueError) as e:
            get_seasonal_hour_division(first_high_hour=25)
        self.assertEqual(str(e.exception), f"first high hour should be between 0 and {DAY_LENGTH}")

    def test_out_of_range_high_hour_num(self):
        with self.assertRaises(ValueError) as e:
            get_seasonal_hour_division(high_hour_num_winter=9)
        self.assertEqual(str(e.exception), f"number of high hours in a day should be between 1 and {MAX_HIGH_HOURS} "
                                           f"(inclusive)")


class TestBuildTariffTable(unittest.TestCase):

    def test_build_tariff_table_default(self):
        result = build_tariff_table(0.15, 0.55, 0.55, 0.14, 0.17, 0.14, 0.17, 0.88, 0.17)
        # check for error in setter
        data = np.loadtxt(os.path.join(test_folder, "financial_calculator/build_tariff_table_result.csv"),
                          delimiter=",")
        nptesting.assert_array_almost_equal(result, data.reshape((7, 12, 24)), 2)


class TestTariffTableToHourly(unittest.TestCase):

    def test_tariff_table_to_hourly(self):
        tariff_table = np.loadtxt(os.path.join(test_folder, "financial_calculator/build_tariff_table_result.csv"),
                                  delimiter=",").reshape((7, 12, 24))
        nptesting.assert_allclose(tariff_table_to_hourly(tariff_table, 2023),
                                  np.loadtxt(os.path.join(test_folder,
                                                          "financial_calculator/hourly_tariff_output.csv")),
                                  atol=0.01)


class TestRelu(unittest.TestCase):

    def test_relu_positive_value(self):
        self.assertEqual(relu(4), 4)

    def test_relu_negative_value(self):
        self.assertEqual(relu(-4), 0)


class TestClamp(unittest.TestCase):

    def test_clamp_default_range(self):
        self.assertEqual(clamp(0), 0.5)

    def test_clamp_different_range(self):
        self.assertEqual(clamp(3, 4, 5, 2, 6), 4.25)


class TestIsLeapYear(unittest.TestCase):

    def test_is_leap_year_not_leap(self):
        self.assertFalse(is_leap_year(2023))

    def test_is_leap_year_leap(self):
        self.assertTrue(is_leap_year(2024))


class TestHourNumInRange(unittest.TestCase):

    def test_start_not_leap_year(self):
        result = hour_num_in_range(2023, 10)
        self.assertEqual(result[0], 87672)
        self.assertEqual(result[1], [8760, 8784, 8760, 8760, 8760, 8784, 8760, 8760, 8760, 8784])

    def test_start_leap_year(self):
        result = hour_num_in_range(2024, 10)
        self.assertEqual(result[0], 87672)
        self.assertEqual(result[1], [8784, 8760, 8760, 8760, 8784, 8760, 8760, 8760, 8784, 8760])


class TestGetYearlyPrices(unittest.TestCase):

    def setUp(self):
        self.yearly_hour_num = [8760, 8784, 8760, 8760, 8760, 8784, 8760, 8760, 8760, 8784]

    def test_get_hourly_tariff_values(self):
        tariff_table = np.loadtxt(os.path.join(test_folder, "financial_calculator/build_tariff_table_result.csv"),
                                  delimiter=",").reshape((7, 12, 24))
        # call function
        result = get_yearly_prices(0, None, None, tariff_table, 2023, self.yearly_hour_num)
        # check output
        nptesting.assert_allclose(result[0], np.loadtxt(os.path.join(test_folder,
                                                                     "financial_calculator/hourly_tariff_output.csv")),
                                  atol=0.01)

    def test_get_hourly_tariff_short_prices_not_leap(self):
        prices = np.ones((8760,))
        # call function and check result
        result = get_yearly_prices(0, prices, prices, None, 2023, self.yearly_hour_num)
        nptesting.assert_array_equal(result[0], np.ones((8760,)))

    def test_get_hourly_tariff_short_prices_leap(self):
        prices = np.ones((8760,))
        # call function and check result
        result = get_yearly_prices(1, prices, prices, None, 2023, self.yearly_hour_num)
        nptesting.assert_array_equal(result[0], np.ones((8784,)))

    def test_get_hourly_prices_tariff_long_prices_first_year(self):
        prices = np.concatenate([(i + 1) * np.ones((x,)) for i, x in enumerate(self.yearly_hour_num)])
        # call function and check result
        result = get_yearly_prices(0, prices, prices, None, 2023, self.yearly_hour_num)
        nptesting.assert_array_equal(result[0], np.ones((8760,)))

    def test_get_hourly_prices_tariff_long_prices_third_year(self):
        prices = np.concatenate([(i + 1) * np.ones((x,)) for i, x in enumerate(self.yearly_hour_num)])
        # call function and check result
        result = get_yearly_prices(2, prices, prices, None, 2023, self.yearly_hour_num)
        nptesting.assert_array_equal(result[0], 3 * np.ones((8760,)))


class TestIsRealNumbers(unittest.TestCase):

    def test_float_array(self):
        # call function
        result = is_real_numbers(np.array([1.5, 3.5]))
        # check result
        self.assertTrue(result)

    def test_int_array(self):
        # call function
        result = is_real_numbers(np.array([1, 3]))
        # check result
        self.assertTrue(result)

    def test_non_numeric_array(self):
        # call function
        result = is_real_numbers(np.array(["1.5", "3.5"]))
        # check result
        self.assertFalse(result)
