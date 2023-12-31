import unittest
from unittest.mock import Mock, patch

import numpy as np
from numpy import testing as nptesting
from optibess_algorithm.power_storage import LithiumPowerStorage


class PowerStorageTest(unittest.TestCase):

    def test_creation_minimal(self):
        # create power storage
        result = LithiumPowerStorage(num_of_years=25, connection_size=5000)
        # check aug table value
        nptesting.assert_array_equal(result.aug_table, np.array([[0, 83, 30937.088], ]))

    def test_creation_default_aug(self):
        # create power storage
        result = LithiumPowerStorage(num_of_years=25, connection_size=5000, use_default_aug=True)
        # check aug table value
        nptesting.assert_array_equal(result.aug_table, np.array([[0, 83, 30937.088], [96, 16, 5963.776],
                                                                 [192, 11, 4100.096]]))

    def test_creation_aug_table(self):
        # create power storage
        result = LithiumPowerStorage(num_of_years=25, connection_size=5000, aug_table=((0, 50), (80, 20), (150, 30)))
        # check aug table value
        nptesting.assert_array_almost_equal(result.aug_table, np.array([[0, 50, 18636.8], [80, 20, 7454.72],
                                                                        [150, 30, 11182.08]]), 10)

    def test_creation_incorrect_num_of_years(self):
        # check for error in creation
        with self.assertRaises(ValueError) as e:
            LithiumPowerStorage(num_of_years=0, connection_size=5000)
        self.assertEqual(str(e.exception), "Number of years should be positive")

    def test_creation_incorrect_connection_size(self):
        # check for error in creation
        with self.assertRaises(ValueError) as e:
            LithiumPowerStorage(num_of_years=25, connection_size=-200)
        self.assertEqual(str(e.exception), "Battery connection size should be positive")

    def test_degradation_table_incorrect_value1(self):
        # create power storage
        result = LithiumPowerStorage(num_of_years=25, connection_size=5000)
        # check for error in setter
        with self.assertRaises(ValueError) as e:
            result.degradation_table = (1, 0.95, 0.92, 0.91)
        self.assertEqual(str(e.exception), f"Battery deg table should have at least {result.num_of_years + 1} entries")

    def test_degradation_table_incorrect_value2(self):
        # create power storage
        result = LithiumPowerStorage(num_of_years=5, connection_size=5000)
        # check for error in setter
        with self.assertRaises(ValueError) as e:
            result.degradation_table = (1, 0.5, 0.3, 0.2, 0.1, 0)
        self.assertEqual(str(e.exception), "Battery deg table should be between 0 (exclusive) and 1 (inclusive)")

    def test_degradation_table_incorrect_value3(self):
        # create power storage
        result = LithiumPowerStorage(num_of_years=5, connection_size=5000)
        # check for error in setter
        with self.assertRaises(ValueError) as e:
            result.degradation_table = (1, 0.5, 0.5, 0.4, 0.3, 0.2)
        self.assertEqual(str(e.exception), "Battery deg table values should be strictly descending")

    def test_dod_table_incorrect_value1(self):
        # create power storage
        result = LithiumPowerStorage(num_of_years=25, connection_size=5000)
        # check for error in setter
        with self.assertRaises(ValueError) as e:
            result.dod_table = (0.9, 0.9, 0.9)
        self.assertEqual(str(e.exception), f"Battery dod table should have at least {result.num_of_years + 1} entries")

    def test_dod_table_incorrect_value2(self):
        # create power storage
        result = LithiumPowerStorage(num_of_years=5, connection_size=5000)
        # check for error in setter
        with self.assertRaises(ValueError) as e:
            result.dod_table = (1.1, 0.95, 0.95, 0.9, 0.9, 0.9)
        self.assertEqual(str(e.exception), "Battery dod table should be between 0 (exclusive) and 1 (inclusive)")

    def test_dod_table_incorrect_value3(self):
        # create power storage
        result = LithiumPowerStorage(num_of_years=5, connection_size=5000)
        # check for error in setter
        with self.assertRaises(ValueError) as e:
            result.dod_table = (0.9, 0.9, 0.89, 0.9, 0.9, 0.9)
        self.assertEqual(str(e.exception), "Battery dod table values should be descending")

    def test_rte_table_incorrect_value1(self):
        # create power storage
        result = LithiumPowerStorage(num_of_years=25, connection_size=5000)
        # check for error in setter
        with self.assertRaises(ValueError) as e:
            result.rte_table = (0.95, 0.95, 0.94, 0.94, 0.93)
        self.assertEqual(str(e.exception), f"Battery rte table should have at least {result.num_of_years + 1} entries")

    def test_rte_table_incorrect_value2(self):
        # create power storage
        result = LithiumPowerStorage(num_of_years=5, connection_size=5000)
        # check for error in setter
        with self.assertRaises(ValueError) as e:
            result.rte_table = (1.01, 0.95, 0.94, 0.93, 0.92, 0.91)
        self.assertEqual(str(e.exception), "Battery rte table should be between 0 (exclusive) and 1 (inclusive)")

    def test_rte_table_incorrect_value3(self):
        # create power storage
        result = LithiumPowerStorage(num_of_years=5, connection_size=5000)
        # check for error in setter
        with self.assertRaises(ValueError) as e:
            result.rte_table = (0.95, 0.95, 0.94, 0.95, 0.93, 0.93)
        self.assertEqual(str(e.exception), "Battery rte table values should be descending")

    def test_creation_incorrect_block_size(self):
        # check for error in creation
        with self.assertRaises(ValueError) as e:
            LithiumPowerStorage(num_of_years=25, connection_size=5000, block_size=-600)
        self.assertEqual(str(e.exception), "Battery block size should be positive")

    def test_idle_self_consumption_incorrect_value(self):
        # create power storage
        result = LithiumPowerStorage(num_of_years=25, connection_size=5000)
        # check for error in setter
        with self.assertRaises(ValueError) as e:
            result.idle_self_consumption = 0
        self.assertEqual(str(e.exception), "Idle battery self consumption should be between 0 and 1 (exclusive)")

    def test_active_self_consumption_incorrect_value(self):
        # create power storage
        result = LithiumPowerStorage(num_of_years=25, connection_size=5000)
        # check for error in setter
        with self.assertRaises(ValueError) as e:
            result.active_self_consumption = 1
        self.assertEqual(str(e.exception), "Active battery self consumption should be between 0 and 1 (exclusive)")

    def test_battery_hours_incorrect_value(self):
        # create power storage
        result = LithiumPowerStorage(num_of_years=25, connection_size=5000)
        # check for error in setter
        with self.assertRaises(ValueError) as e:
            result.battery_hours = 10
        self.assertEqual(str(e.exception), "Battery hours should be between 0 and 8")

    def test_aug_table_incorrect_value1(self):
        # create power storage
        result = LithiumPowerStorage(num_of_years=25, connection_size=5000)
        # check for error in setter
        with self.assertRaises(ValueError) as e:
            result.aug_table = ()
        self.assertEqual(str(e.exception), "Augmentation table should have at least 1 entry")

    def test_aug_table_incorrect_value2(self):
        # create power storage
        result = LithiumPowerStorage(num_of_years=25, connection_size=5000)
        # check for error in setter
        with self.assertRaises(ValueError) as e:
            result.aug_table = ((0, 10, 3727.36),)
        self.assertEqual(str(e.exception), "Augmentation table entries should have 2 values")

    def test_aug_table_incorrect_value3(self):
        # create power storage
        result = LithiumPowerStorage(num_of_years=25, connection_size=5000)
        # check for error in setter
        with self.assertRaises(ValueError) as e:
            result.aug_table = ((-1, 10), (60, 20), (180, 10))
        self.assertEqual(str(e.exception), "Augmentation table entries should have non negative and positive values")

    def test_aug_table_incorrect_value4(self):
        # create power storage
        result = LithiumPowerStorage(num_of_years=25, connection_size=5000)
        # check for error in setter
        with self.assertRaises(ValueError) as e:
            result.aug_table = ((0, 20), (50, 10), (100, 0))
        self.assertEqual(str(e.exception), "Augmentation table entries should have non negative and positive values")

    def test_aug_table_incorrect_value5(self):
        # create power storage
        result = LithiumPowerStorage(num_of_years=25, connection_size=5000)
        # check for error in setter
        with self.assertRaises(ValueError) as e:
            result.aug_table = ((0, 20), (20, 10), (60, 10))
        self.assertEqual(str(e.exception), "Augmentation table entries should have at least a 3 year gap")

    def test_set_aug_table_no_month_diff_check(self):
        # create power storage
        result = LithiumPowerStorage(num_of_years=25, connection_size=5000, block_size=10)
        # call function
        result.set_aug_table(((0, 20), (20, 10), (60, 10)), False)
        # check output
        nptesting.assert_array_equal(result.aug_table, ((0, 20, 200), (20, 10, 100), (60, 10, 100)))

    def test_set_aug_table_too_large(self):
        # create power storage
        result = LithiumPowerStorage(num_of_years=25, connection_size=5000, block_size=10)
        # check warn called
        with patch("optibess_algorithm.power_storage.LithiumPowerStorage.check_battery_size", return_value=False):
            with patch("optibess_algorithm.power_storage.warnings.warn") as mock_warn:
                result.set_aug_table(((0, 100), (96, 50), (192, 50)))
                mock_warn.assert_called_once()

    def test_check_battery_size_valid(self):
        # create storage system and call function
        storage_system = LithiumPowerStorage(num_of_years=25, connection_size=5000)
        result = storage_system.check_battery_size(((0, 135), (96, 22), (192, 19)))
        # check output
        self.assertTrue(result)

    def test_check_battery_size_invalid(self):
        # create storage system and call function
        storage_system = LithiumPowerStorage(num_of_years=25, connection_size=5000)
        result = storage_system.check_battery_size(((0, 136), (96, 22), (192, 20)))
        # check output
        self.assertFalse(result)
