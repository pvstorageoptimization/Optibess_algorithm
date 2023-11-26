import unittest
from numpy import testing as nptesting
import pandas as pd
import numpy as np

from optibess_algorithm.utils import shift_array, year_diff, month_diff, build_tariff_table


class TestUtils(unittest.TestCase):

    def test_shift_array_positive(self):
        result = shift_array([0, 1, 2, 3, 4, 5], 2)
        nptesting.assert_array_equal(result, [np.nan, np.nan, 0, 1, 2, 3])

    def test_shift_array_non_numeric_values_no_fill_value(self):
        with self.assertRaises(ValueError):
            shift_array(["a", "b", "c", "d"], 1)

    def test_shift_array_non_numeric_values_num_zero(self):
        result = shift_array(["a", "b", "c", "d"], 0)
        nptesting.assert_array_equal(result, ["a", "b", "c", "d"])

    def test_shift_array_non_numeric_values_with_fill_value(self):
        result = shift_array(["a", "b", "c", "d"], 1, "C")
        nptesting.assert_array_equal(result, ["C", "a", "b", "c"])

    def test_shift_array_negative(self):
        result = shift_array([0, 1, 2, 3, 4, 5], -3)
        nptesting.assert_array_equal(result, [3, 4, 5, np.nan, np.nan, np.nan])

    def test_year_diff_single_values(self):
        result = year_diff(pd.to_datetime("2025-5-1"), pd.to_datetime("2023-1-1"))
        self.assertEqual(result, 2)

    def test_year_diff_series_single_value(self):
        result = year_diff(pd.to_datetime("2025-6-5"), pd.date_range("2023-5-1", "2023-5-10"))
        nptesting.assert_array_equal(result, (2, 2, 2, 2, 2, 2, 2, 2, 2, 2))

    def test_year_diff_2_series(self):
        result = year_diff(pd.date_range("2025-6-2", "2025-6-5"), pd.date_range("2023-5-1", "2023-5-4"))
        nptesting.assert_array_equal(result, (2, 2, 2, 2))

    def test_year_diff_single_value_series(self):
        result = year_diff(pd.date_range("2026-7-30", "2026-8-3"), pd.to_datetime("2023-7-31"))
        nptesting.assert_array_equal(result, (2, 3, 3, 3, 3))

    def test_year_diff_earlier_month(self):
        result = year_diff(pd.to_datetime("2027-6-1"), pd.to_datetime("2023-7-1"))
        self.assertEqual(result, 3)

    def test_year_diff_earlier_day(self):
        result = year_diff(pd.to_datetime("2024-9-2"), pd.to_datetime("2023-9-5"))
        self.assertEqual(result, 0)

    def test_month_diff_single_values(self):
        result = month_diff(pd.to_datetime("2025-4-5"), pd.to_datetime("2023-1-1"))
        self.assertEqual(result, 27)

    def test_month_diff_series_single_value(self):
        result = month_diff(pd.to_datetime("2024-7-4"), pd.date_range("2023-7-1", "2023-7-3"))
        nptesting.assert_array_equal(result, (12, 12, 12))

    def test_month_diff_2_series(self):
        result = month_diff(pd.date_range("2025-10-20", "2025-10-24"), pd.date_range("2023-8-23", "2023-8-27"))
        nptesting.assert_array_equal(result, (25, 25, 25, 25, 25))

    def test_month_diff_single_value_series(self):
        result = month_diff(pd.date_range("2025-3-2", "2025-3-5"), pd.to_datetime("2023-3-3"))
        nptesting.assert_array_equal(result, (23, 24, 24, 24))

    def test_month_diff_earlier_month(self):
        result = month_diff(pd.to_datetime("2025-9-25"), pd.to_datetime("2023-10-20"))
        self.assertEqual(result, 23)

    def test_month_diff_earlier_day(self):
        result = month_diff(pd.to_datetime("2027-3-15"), pd.to_datetime("2023-2-19"))
        self.assertEqual(result, 48)

    def test_build_tariff_table_default(self):
        result = build_tariff_table()
        nptesting.assert_array_equal(result, ((1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1),
                                              (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1),
                                              (3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 3, 3),
                                              (3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 3, 3),
                                              (3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 3, 3),
                                              (5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 5),
                                              (5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 5),
                                              (5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 5),
                                              (5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 5),
                                              (3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 3, 3),
                                              (3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 3, 3),
                                              (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1)))

    def test_build_tariff_table_months_change(self):
        result = build_tariff_table(transition_months=(3, 4, 9, 10), winter_months=(0, 1, 2, 11))
        nptesting.assert_array_equal(result, ((1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1),
                                              (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1),
                                              (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1),
                                              (3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 3, 3),
                                              (3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 3, 3),
                                              (5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 5),
                                              (5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 5),
                                              (5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 5),
                                              (5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 5),
                                              (3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 3, 3),
                                              (3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 3, 3),
                                              (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1)))

    def test_tariff_table_hours_change(self):
        result = build_tariff_table(summer_low_hours=tuple(range(1, 15)) + (16, 23, 0),
                                    summer_high_hours=(15,) + tuple(range(17, 23)))
        nptesting.assert_array_equal(result, ((1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1),
                                              (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1),
                                              (3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 3, 3),
                                              (3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 3, 3),
                                              (3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 3, 3),
                                              (5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 5, 6, 6, 6, 6, 6, 6, 5),
                                              (5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 5, 6, 6, 6, 6, 6, 6, 5),
                                              (5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 5, 6, 6, 6, 6, 6, 6, 5),
                                              (5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 5, 6, 6, 6, 6, 6, 6, 5),
                                              (3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 3, 3),
                                              (3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 3, 3),
                                              (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1)))

    def test_build_tariff_table_overlapping_months(self):
        with self.assertRaises(ValueError) as e:
            build_tariff_table(summer_months=(5, 6, 7, 8, 9))
        self.assertEqual(str(e.exception), "Seasons should have different months")

    def test_build_tariff_table_missing_months(self):
        with self.assertRaises(ValueError) as e:
            build_tariff_table(winter_months=(0, 1))
        self.assertEqual(str(e.exception), "Months doesn't cover all year")

    def test_build_tariff_table_overlapping_hours(self):
        with self.assertRaises(ValueError) as e:
            build_tariff_table(winter_high_hours=list(range(16, 23)))
        self.assertEqual(str(e.exception), "Winter low and high hours should not overlap")

    def test_build_tariff_table_missing_hours(self):
        with self.assertRaises(ValueError) as e:
            build_tariff_table(transition_low_hours=list(range(1, 17)))
        self.assertEqual(str(e.exception), "Transition low and high hours doesn't cover the day")
