import unittest
from unittest.mock import Mock, call, PropertyMock
import numpy as np

from Optibess_algorithm.financial_calculator import FinancialCalculator
from Optibess_algorithm.output_calculator import OutputCalculator
from Optibess_algorithm.power_storage import PowerStorage
from Optibess_algorithm.power_system_optimizer import NevergradOptimizer


class MockAugTableProperty:

    def __get__(self, instance, owner=None):
        return np.array([[0, 70, 7000], [96, 20, 2000], [192, 10, 1000]])

    def __set__(self, instance, value):
        raise ValueError


class TestPowerSystemOptimizer(unittest.TestCase):

    def setUp(self) -> None:
        self.finance = Mock(spec=FinancialCalculator)
        type(self.finance).num_of_years = 25
        self.output = Mock(spec=OutputCalculator)
        type(self.output).aug_table = np.array([[0, 70, 7000], [96, 20, 2000], [192, 10, 1000]])
        type(self.finance).output_calculator = self.output
        self.power_storage = Mock(spec=PowerStorage)
        self.power_storage.check_battery_size = Mock(return_value=True)
        type(self.output).power_storage = self.power_storage

    def test_creation_calculator_only(self):
        # set inputs
        opt = NevergradOptimizer(self.finance)
        # check outputs
        self.assertEqual(opt._max_aug_num, 6)
        self.assertEqual(opt._initial_aug_num, 3)
        self.assertEqual(opt._def_aug_diff, 5)

    def test_creation_no_args(self):
        # set inputs
        opt = NevergradOptimizer()
        # check outputs
        self.assertEqual(type(opt._financial_calculator), FinancialCalculator)
        self.assertEqual(opt._max_aug_num, 6)
        self.assertEqual(opt._initial_aug_num, 3)
        self.assertEqual(opt._def_aug_diff, 5)

    def test_creation_max_aug_num_big(self):
        # set inputs
        opt = NevergradOptimizer(self.finance, max_aug_num=15)
        # check outputs
        self.assertEqual(opt._max_aug_num, 9)
        self.assertEqual(opt._initial_aug_num, 4)
        self.assertEqual(opt._def_aug_diff, 3)

    def test_creation_initial_aug_num_big(self):
        # set inputs
        opt = NevergradOptimizer(self.finance, initial_aug_num=10)
        # check outputs
        self.assertEqual(opt._max_aug_num, 6)
        self.assertEqual(opt._initial_aug_num, 6)
        self.assertEqual(opt._def_aug_diff, 5)

    def test_creation_incorrect_max_aug_num(self):
        with self.assertRaises(ValueError) as e:
            NevergradOptimizer(self.finance, max_aug_num=0)
        self.assertEqual(str(e.exception), "Number of maximum augmentations should be at least 1")

    def test_creation_incorrect_initial_aug_num(self):
        with self.assertRaises(ValueError) as e:
            NevergradOptimizer(self.finance, initial_aug_num=-3)
        self.assertEqual(str(e.exception), "Number of maximum augmentations should be at least 1")

    def test_creation_incorrect_budget(self):
        with self.assertRaises(ValueError) as e:
            NevergradOptimizer(self.finance, budget=0)
        self.assertEqual(str(e.exception), "Optimization budget should be at least 1")

    def test_maximize_objective(self):
        # set inputs
        type(self.finance).get_irr = Mock(return_value=14.5)
        type(self.output).run = Mock()
        opt = NevergradOptimizer(self.finance)
        # check outputs
        result = opt.maximize_objective((0, 5, 8, 0, 0, 0, 100, 20, 20, 0, 0, 0, 98))
        self.assertEqual(result, 14.5)
        self.assertEqual(opt._first_aug_of_size, {1: False, 2: False, 3: True, 4: False, 5: False, 6: False})
        self.assertEqual(opt._memory, {(((0, 100), (60, 20), (96, 20)), 98): 14.5})
        self.assertEqual(opt._output.aug_table, ((0, 100), (60, 20), (96, 20)))
        self.assertEqual(opt._output.producer_factor, 0.98)
        self.output.run.assert_called_once()

    def test_maximize_objective_result_from_memory(self):
        # set inputs
        opt = NevergradOptimizer(self.finance)
        opt._memory = {(((0, 100), (60, 20), (96, 20)), 98): 15.5}
        # check outputs
        result = opt.maximize_objective((0, 5, 8, 0, 0, 0, 100, 20, 20, 0, 0, 0, 98))
        self.assertEqual(result, 15.5)
        self.assertEqual(opt._first_aug_of_size, {1: False, 2: False, 3: True, 4: False, 5: False, 6: False})
        self.assertEqual(opt._memory, {(((0, 100), (60, 20), (96, 20)), 98): 15.5})
        np.testing.assert_equal(np.any(np.not_equal(opt._output.aug_table,
                                                    ((0, 100, 10000), (60, 20, 2000), (96, 20, 2000)))), True)
        self.assertNotEqual(opt._output.producer_factor, 0.98)
        self.output.run.assert_not_called()

    def test_maximize_objective_invalid_aug_table(self):
        # setup inputs and mocks
        type(self.output).aug_table = MockAugTableProperty()
        opt = NevergradOptimizer(self.finance)

        # check outputs
        result = opt.maximize_objective((0, 5, 8, 0, 0, 0, 100, 20, 20, 0, 0, 0, 98))
        self.assertEqual(result, -100)
        self.assertEqual(opt._first_aug_of_size, {1: False, 2: False, 3: True, 4: False, 5: False, 6: False})
        self.assertEqual(opt._memory, {(((0, 100), (60, 20), (96, 20)), 98): -100})
        np.testing.assert_equal(np.any(np.not_equal(opt._output.aug_table,
                                                    ((0, 100, 10000), (60, 20, 2000), (96, 20, 2000)))), True)
        self.assertNotEqual(opt._output.producer_factor, 0.98)
        self.output.run.assert_not_called()

    def test_candid_creation(self):
        # set inputs
        opt = NevergradOptimizer(self.finance)
        # check outputs
        result = opt.get_candid((0, 5, 8, 0, 0, 0, 100, 20, 20, 0, 0, 0, 98))
        self.assertEqual(result, (((0, 100), (60, 20), (96, 20)), 98))

    def test_aug_table_creation(self):
        # set inputs
        opt = NevergradOptimizer(self.finance)
        # check outputs
        result = opt.get_aug_table((0, 5, 8, 0, 0, 0, 100, 20, 20, 0, 0, 0, 98))
        self.assertEqual(result, ((0, 100), (60, 20), (96, 20)))

    def test_run(self):
        # set inputs
        type(self.finance).get_irr = Mock(return_value=14.5)
        type(self.output).run = Mock()
        type(self.output).num_of_years = 25
        progress_recorder = Mock()
        progress_recorder.set_progress = Mock()
        opt = NevergradOptimizer(self.finance, budget=4)
        # check outputs
        result = opt.run(progress_recorder)
        self.assertEqual(type(result[0]), tuple)
        self.assertEqual(type(result[1]), float)
        progress_recorder.set_progress.assert_has_calls([call(i, 4) for i in range(1, 5)])
        self.assertEqual(progress_recorder.set_progress.call_count, 4)
