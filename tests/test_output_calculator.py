import unittest
from unittest.mock import Mock
import pandas as pd
import numpy as np
import numpy.testing as nptesting
import os

from optibess_algorithm.output_calculator import OutputCalculator, Coupling
from optibess_algorithm.producers import Producer
from optibess_algorithm.power_storage import PowerStorage
import optibess_algorithm.constants as constants

test_folder = os.path.dirname(os.path.abspath(__file__))


class TestOutputCalculator(unittest.TestCase):

    def setUp(self) -> None:
        # create mock producer
        self.producer = Mock(spec=Producer)
        type(self.producer).power_output = pd.DataFrame(np.zeros((8760,)),
                                                        pd.date_range(start='2023-1-1 00:00',
                                                                      end='2023-12-31 23:00', freq='h'), ['pv_output'])
        type(self.producer).annual_deg = 0.0035
        # create mock power storage
        self.power_storage = Mock(spec=PowerStorage)
        type(self.power_storage).num_of_years = 25
        type(self.power_storage).aug_table = np.array([[0, 70, 7000], [96, 20, 2000], [192, 10, 1000]])
        type(self.power_storage).rte_table = constants.DEFAULT_RTE_TABLE
        type(self.power_storage).degradation_table = constants.DEFAULT_DEG_TABLE
        type(self.power_storage).dod_table = constants.DEFAULT_DOD_TABLE
        type(self.power_storage).active_self_consumption = constants.DEFAULT_ACTIVE_SELF_CONSUMPTION
        type(self.power_storage).idle_self_consumption = constants.DEFAULT_IDLE_SELF_CONSUMPTION

        # create default output calculator
        self.output = OutputCalculator(num_of_years=25, grid_size=5000, producer=self.producer,
                                       power_storage=self.power_storage)

    def test_creation_regular(self):
        # create output calculator
        result = OutputCalculator(num_of_years=25, grid_size=5000, producer=self.producer,
                                  power_storage=self.power_storage)
        # check pcs value
        self.assertAlmostEqual(result.pcs_power, 5532, 0)

    def test_creation_incorrect_num_of_years(self):
        # check for error in creation
        with self.assertRaises(ValueError) as e:
            OutputCalculator(num_of_years=0, grid_size=5000, producer=self.producer, power_storage=self.power_storage)
        self.assertEqual(str(e.exception), "Number of years should be positive")

    def test_set_grid_size_pcs_power_changed(self):
        # change value
        self.output.grid_size = 4000
        # check pcs value
        self.assertAlmostEqual(self.output.pcs_power, 4441, 0)

    def test_grid_size_incorrect_value(self):
        # check for error in setter
        with self.assertRaises(ValueError) as e:
            self.output.grid_size = -400
        self.assertEqual(str(e.exception), "Grid size should be positive")

    def test_producer_incorrect_type(self):
        # check for error in setter
        with self.assertRaises(ValueError) as e:
            self.output.producer = np.empty(8670)
        self.assertEqual(str(e.exception), "Producer should be an instance of class Producer")

    def test_set_power_storage_pcs_power_changed(self):
        # setup mock
        power_storage = Mock(spec=PowerStorage)
        type(power_storage).num_of_years = 25
        type(power_storage).aug_table = np.array([[0, 70, 7000], [96, 20, 2000], [192, 10, 1000]])
        type(power_storage).rte_table = constants.DEFAULT_RTE_TABLE
        type(power_storage).active_self_consumption = 0.005
        # changed value
        self.output.power_storage = power_storage
        # check pcs value
        self.assertAlmostEqual(self.output.pcs_power, 5552, 0)

    def test_power_storage_incorrect_type(self):
        # check for error in setter
        with self.assertRaises(ValueError) as e:
            self.output.power_storage = [0 for _ in range(8760)]
        self.assertEqual(str(e.exception), "Power storage should be an instance of class PowerStorage")

    def test_power_storage_too_few_years(self):
        # setup mock
        power_storage = Mock(spec=PowerStorage)
        type(power_storage).num_of_years = 20
        # check for error in setter
        with self.assertRaises(ValueError) as e:
            self.output.power_storage = power_storage
        self.assertEqual(str(e.exception), "Power Storage number of years should be at least as many as the system")

    def test_set_coupling_values_change(self):
        # change value
        self.output.coupling = Coupling.DC
        # check values
        self.assertAlmostEqual(self.output.prod_trans_loss, 0.02485, 4)
        self.assertEqual(self.output.charge_loss, 0.015)
        self.assertAlmostEqual(self.output.grid_bess_loss, 0.03947725, 4)
        self.assertAlmostEqual(self.output.pcs_power, 5559, 0)

    def test_set_coupling_same_value(self):
        # change value
        self.output.coupling = Coupling.AC
        # check no value change
        self.assertAlmostEqual(self.output.prod_trans_loss, 0.0199, 4)

    def test_set_mvpv_loss_values_changed_ac_coupling(self):
        # change value
        self.output.mvpv_loss = 0.005
        # check values
        self.assertAlmostEqual(self.output.charge_loss, 0.02972575, 4)
        self.assertAlmostEqual(self.output.prod_trans_loss, 0.01495, 4)
        self.assertAlmostEqual(self.output.pcs_power, 5532, 0)

    def test_set_mvpv_loss_values_changed_dc_coupling(self):
        # change to dc coupling
        self.output.coupling = Coupling.DC

        # change loss value
        self.output.mvpv_loss = 0.005
        # check values
        self.assertEqual(self.output.charge_loss, 0.015)
        self.assertAlmostEqual(self.output.prod_trans_loss, 0.02485, 4)
        self.assertAlmostEqual(self.output.pcs_power, 5559, 0)

    def test_charge_loss_incorrect_value(self):
        # check for error in setter
        with self.assertRaises(ValueError) as e:
            self.output.mvpv_loss = 1.5
        self.assertEqual(str(e.exception), "MV-PV loss should be between 0 and 1 (exclusive)")

    def test_set_trans_loss_values_changed_ac_coupling(self):
        # change value
        self.output.trans_loss = 0.012
        # check values
        self.assertAlmostEqual(self.output.grid_bess_loss, 0.0365518, 4)
        self.assertAlmostEqual(self.output.prod_trans_loss, 0.02188, 4)
        self.assertAlmostEqual(self.output.pcs_power, 5543, 0)

    def test_set_trans_loss_values_changed_dc_coupling(self):
        # change to dc coupling
        self.output.coupling = Coupling.DC
        # change loss value
        self.output.trans_loss = 0.012
        # check values
        self.assertAlmostEqual(self.output.grid_bess_loss, 0.0414177, 4)
        self.assertAlmostEqual(self.output.prod_trans_loss, 0.02682, 4)
        self.assertAlmostEqual(self.output.pcs_power, 5571, 0)

    def test_trans_loss_incorrect_value(self):
        # check for error in setter
        with self.assertRaises(ValueError) as e:
            self.output.trans_loss = 0
        self.assertEqual(str(e.exception), "Trans loss should be between 0 and 1 (exclusive)")

    def test_set_mvbat_loss_values_changed_ac_coupling(self):
        # change value
        self.output.mvbat_loss = 0.013
        # check values
        self.assertAlmostEqual(self.output.grid_bess_loss, 0.03752695, 4)
        self.assertAlmostEqual(self.output.charge_loss, 0.03752695, 4)
        self.assertAlmostEqual(self.output.pcs_power, 5548, 0)

    def test_set_mvbat_loss_values_changed_dc_coupling(self):
        # change to dc coupling
        self.output.coupling = Coupling.DC
        # change loss value
        self.output.mvbat_loss = 0.013
        # check values
        self.assertAlmostEqual(self.output.grid_bess_loss, 0.03947725, 4)
        self.assertEqual(self.output.charge_loss, 0.015)
        self.assertAlmostEqual(self.output.pcs_power, 5559, 0)

    def test_mvbat_loss_incorrect_value(self):
        # check for error in setter
        with self.assertRaises(ValueError) as e:
            self.output.mvbat_loss = 0
        self.assertEqual(str(e.exception), "MV-BAT loss should be between 0 and 1 (exclusive)")

    def test_set_pcs_loss_values_changed_ac_coupling(self):
        # change value
        self.output.pcs_loss = 0.011
        # check values
        self.assertAlmostEqual(self.output.prod_trans_loss, 0.0199, 4)
        self.assertAlmostEqual(self.output.grid_bess_loss, 0.0306811, 4)
        self.assertAlmostEqual(self.output.charge_loss, 0.0306811, 4)
        self.assertAlmostEqual(self.output.pcs_power, 5510, 0)

    def test_set_pcs_loss_values_changed_dc_coupling(self):
        # change to dc coupling
        self.output.coupling = Coupling.DC
        # change loss value
        self.output.pcs_loss = 0.011
        # check values
        self.assertAlmostEqual(self.output.prod_trans_loss, 0.02089, 4)
        self.assertAlmostEqual(self.output.grid_bess_loss, 0.03557665, 4)
        self.assertEqual(self.output.charge_loss, 0.015)
        self.assertAlmostEqual(self.output.pcs_power, 5537, 0)

    def test_pcs_loss_incorrect_value(self):
        # check for error in setter
        with self.assertRaises(ValueError) as e:
            self.output.pcs_loss = 0
        self.assertEqual(str(e.exception), "PCS loss should be between 0 and 1 (exclusive)")

    def test_set_dc_dc_loss_values_changed_ac_coupling(self):
        # change value
        self.output.dc_dc_loss = 0.01
        # check values
        self.assertAlmostEqual(self.output.grid_bess_loss, 0.0346015, 4)
        self.assertAlmostEqual(self.output.charge_loss, 0.0346015, 4)
        self.assertAlmostEqual(self.output.pcs_power, 5532, 0)

    def test_set_dc_dc_loss_values_changed_dc_coupling(self):
        # change to dc coupling
        self.output.coupling = Coupling.DC
        # change loss value
        self.output.dc_dc_loss = 0.01
        # check values
        self.assertAlmostEqual(self.output.grid_bess_loss, 0.0346015, 4)
        self.assertEqual(self.output.charge_loss, 0.01)
        self.assertAlmostEqual(self.output.pcs_power, 5532, 0)

    def test_dc_dc_loss_incorrect_value(self):
        # check for error in setter
        with self.assertRaises(ValueError) as e:
            self.output.dc_dc_loss = 1.1
        self.assertEqual(str(e.exception), "DC-DC loss should be between 0 and 1 (exclusive)")

    def test_set_aug_table_pcs_power_changed(self):
        # setup mock for power storage set aug table
        def mock_set_aug_table(x, _):
            type(self.power_storage).aug_table = x

        type(self.power_storage).set_aug_table = Mock(side_effect=mock_set_aug_table)
        # change value
        self.output.aug_table = np.array([[0, 70, 7000], [96, 20, 1000], [192, 10, 1000]])
        # check pcs value
        self.assertAlmostEqual(self.output.pcs_power, 5524, 0)

    def test_discharge_hour_incorrect_value(self):
        # check for error in setter
        with self.assertRaises(ValueError) as e:
            self.output.bess_discharge_start_hour = 25
        self.assertEqual(str(e.exception), "Battery discharge start hour should be between 0 and 23 (inclusive)")

    def test_producer_factor_incorrect_value(self):
        # check for error in setter
        with self.assertRaises(ValueError) as e:
            self.output.producer_factor = 0
        self.assertEqual(str(e.exception), "Producer factor should be between 0 (Exclusive) and 1 (inclusive)")

    def test_get_data_shape(self):
        # check first year values
        self.output._get_data(0)
        self.assertEqual(self.output._df["pv_output"].shape[0], 8760)
        nptesting.assert_array_equal(self.output._df.index, pd.date_range(start='2023-1-1 00:00',
                                                                          end='2023-12-31 23:00', freq='h'))
        # check second year values
        self.output._get_data(1)
        self.assertEqual(self.output._df["pv_output"].shape[0], 8784)
        nptesting.assert_array_equal(self.output._df.index, pd.date_range(start='2024-1-1 00:00',
                                                                          end='2024-12-31 23:00', freq='h'))

    def test_calc_overflow_values(self):
        # set inputs
        self.output.grid_size = 2000
        self.output._pcs_power = 0
        self.output._df = pd.read_csv(os.path.join(test_folder, "output_calculator/overflow_data1.csv"))
        self.output._df["acc_losses"] = 0
        # check outputs
        self.output._calc_overflow()
        nptesting.assert_array_almost_equal(self.output._df["overflow"],
                                            pd.read_csv(os.path.join(test_folder,
                                                                     "output_calculator/overflow_data1_results.csv"))
                                            ["overflow"], 0)

    def test_get_day_deg_regular(self):
        # call function
        result = self.output._get_day_deg(pd.Timestamp("2024-1-15"), pd.Timestamp("2023-1-1"))
        # check output
        self.assertAlmostEqual(result, 0.92336438356, 4)

    def test_det_day_deg_earlier_month(self):
        # call function
        result = self.output._get_day_deg(pd.Timestamp("2027-2-1"), pd.Timestamp("2025-5-1"))
        # check output
        self.assertAlmostEqual(result, 0.90415, 4)

    def additional_setup(self):
        """
        additional setup for tests after creation
        """
        type(self.power_storage).aug_table = np.array([[0, 20, 2000], [96, 10, 1000]])
        self.output.grid_size = 1000
        self.output._pcs_power = 1140
        self.output._df = pd.DataFrame(index=pd.date_range(start='2023-1-1 00:00',
                                                           end='2023-1-1 23:00', freq='h'))
        self.output._daily_index = pd.date_range(start='2023-1-1 17:00', end='2023-1-1 17:00', freq='d')
        self.output._df["battery_capacity"] = 2700
        self.output._df["battery_nameplate"] = 3000
        self.output._df["acc_losses"] = 0
        self.output._active_hourly_self_cons = np.full(24, 12)
        self.output._indices = np.arange(24)
        self.output._charge_loss = 0.015
        self.output._grid_bess_loss = 1 - .985 * .98

    def test_calc_augmentations_battery_capacity_values(self):
        # set inputs
        self.additional_setup()
        self.output._initial_date = pd.Timestamp("2023-1-1")
        self.output._next_aug = 1
        self.output._first_aug_entries = [pd.Timestamp("2023-1-1")]
        self.output._df = pd.DataFrame(index=pd.date_range(start='2031-1-1 00:00',
                                                           end='2031-1-31 23:00', freq='h'))
        self.output._daily_index = pd.date_range(start='2031-1-1 17:00', end='2031-1-31 17:00', freq='d')
        self.output._calc_augmentations()
        # check outputs
        self.assertEqual(self.output._df["aug_1"].iloc[0], 900)
        for x in range(17, 30 * 24 + 17, 24):
            self.assertGreater(self.output._df["aug_1"].iloc[x], self.output._df["aug_1"].iloc[x + 24])
        self.assertAlmostEqual(self.output._df["aug_1"].iloc[30 * 24 + 17], 894.39, 2)
        nptesting.assert_array_equal(self.output._df["battery_capacity"], self.output._df["aug_0"] +
                                     self.output._df["aug_1"])

    def test_pv2bess_battery_overflow_scenario(self):
        # set inputs
        self.additional_setup()
        self.output._df["pv_output"] = pd.read_csv(os.path.join(test_folder, "output_calculator/pv2bess_data1.csv"))[
            "pv_output"].to_numpy()
        # check outputs
        self.output._calc_pv_to_bess()
        nptesting.assert_array_almost_equal(self.output._df.loc[:, ["pv2bess", "battery_overflow"]],
                                            pd.read_csv(os.path.join(test_folder,
                                                                     "output_calculator/pv2bess_data1_results.csv")), 0)

    def test_pv2bess_battery_underflow_scenario(self):
        # set inputs
        self.additional_setup()
        self.output._df["pv_output"] = pd.read_csv(os.path.join(test_folder, "output_calculator/pv2bess_data2.csv"))[
            "pv_output"].to_numpy()
        # check outputs
        self.output._calc_pv_to_bess()
        nptesting.assert_allclose(self.output._df.loc[:, ["pv2bess", "battery_overflow"]],
                                  pd.read_csv(os.path.join(test_folder,
                                                           "output_calculator/pv2bess_data2_results.csv")), atol=3)

    def test_daily_initial_soc_regular(self):
        # set inputs
        self.additional_setup()
        self.output._df["pv2bess"] = pd.read_csv(os.path.join(test_folder, "output_calculator/daily_soc_data1.csv"))[
            "pv2bess"].to_numpy()
        # check outputs
        self.output._calc_daily_initial_battery_soc(0)
        np.testing.assert_array_almost_equal(self.output._initial_battery_soc,
                                             np.loadtxt(os.path.join(test_folder,
                                                                     "output_calculator/daily_soc_data1_results.csv")),
                                             0)

    def test_daily_initial_soc_early_discharge_hour_2_days_second_year(self):
        # set inputs
        self.additional_setup()
        self.output._df = self.output._df.reindex(pd.date_range(start='2024-1-1 00:00', end='2024-1-2 23:00', freq='h'),
                                                  method='nearest')
        self.output._df["pv2bess"] = pd.read_csv(os.path.join(test_folder, "output_calculator/daily_soc_data2.csv"))[
            "pv2bess"].to_numpy()
        self.output.bess_discharge_start_hour = 1
        self.output._indices = np.arange(48)
        self.output._daily_index = pd.date_range(start='2024-1-1 1:00', end='2024-1-2 1:00', freq='d')
        self.output._active_hourly_self_cons = np.full(48, 12)
        # check outputs
        self.output._calc_daily_initial_battery_soc(1)
        np.testing.assert_array_almost_equal(self.output._initial_battery_soc,
                                             np.loadtxt(os.path.join(test_folder,
                                                                     "output_calculator/daily_soc_data2_results.csv")),
                                             0)

    def test_grid_to_bess_full_battery(self):
        # set inputs
        self.additional_setup()
        self.output._initial_battery_soc = np.full(24, 2700)
        self.output._df["pv2bess"] = pd.read_csv(os.path.join(test_folder, "output_calculator/grid2bess_data1.csv"))[
            "pv2bess"].to_numpy()
        # check outputs
        self.output._calc_grid_to_bess(0)
        nptesting.assert_array_equal(self.output._df["grid2bess"], 0)
        nptesting.assert_array_equal(self.output._initial_battery_soc, 2700)

    def test_grid_to_bess_missing_power(self):
        # set inputs
        self.additional_setup()
        self.output._initial_battery_soc = np.full(24, 1331)
        self.output._df["pv2bess"] = pd.read_csv(os.path.join(test_folder, "output_calculator/grid2bess_data2.csv"))[
            "pv2bess"].to_numpy()
        # check outputs
        self.output._calc_grid_to_bess(0)
        nptesting.assert_array_almost_equal(self.output._df["grid2bess"],
                                            pd.read_csv(os.path.join(test_folder,
                                                                     "output_calculator/grid2bess_data2_results.csv"))
                                            ["grid2bess"],
                                            0)
        nptesting.assert_array_almost_equal(self.output._initial_battery_soc, 2700, 0)

    def test_grid_to_bess_early_discharge_hour(self):
        # set inputs
        self.additional_setup()
        self.output.bess_discharge_start_hour = 1
        self.output._initial_battery_soc = np.insert(np.full(59, 1331, dtype=float), 0, np.full(13, 0))
        self.output._df = self.output._df.reindex(pd.date_range("2023-1-1 00:00", "2023-1-3 23:00", freq='h'),
                                                  method='nearest')
        self.output._df["pv2bess"] = pd.read_csv(os.path.join(test_folder, "output_calculator/grid2bess_data3.csv"))[
            "pv2bess"].to_numpy()
        self.output._active_hourly_self_cons = np.full(72, 12)
        self.output._indices = np.arange(72)
        # check outputs
        self.output._calc_grid_to_bess(0)
        expected = pd.read_csv(os.path.join(test_folder, "output_calculator/grid2bess_data3_results.csv"))
        nptesting.assert_array_almost_equal(self.output._df["grid2bess"], expected["grid2bess"], 0)
        nptesting.assert_array_almost_equal(self.output._initial_battery_soc, expected["initial_battery_soc"], 0)

    def additional_setup_power_to_grid(self):
        """
        additional setup (on top of the previous) for power to grid tests
        """
        self.additional_setup()
        self.output._calc_soc = Mock()
        self.output._initial_battery_soc = np.full(24, 2700)

    def test_power_to_grid_pv_only(self):
        # set inputs
        self.additional_setup_power_to_grid()
        self.output._df["grid2bess"] = 0
        inputs = pd.read_csv(os.path.join(test_folder, "output_calculator/power2grid_data1.csv"))
        self.output._df["pv_output"] = inputs["pv_output"].to_numpy()
        self.output._df["pv2bess"] = inputs["pv2bess"].to_numpy()
        # check outputs
        self.output._calc_power_to_grid(0)
        nptesting.assert_allclose(self.output._df.loc[:, ["pv2grid", "bess2grid", "output", "grid2bess",
                                                          "grid_from_pv", "grid_from_bess"]],
                                  pd.read_csv(os.path.join(test_folder,
                                                           "output_calculator/power2grid_data1_results.csv")), atol=3)
        nptesting.assert_array_equal(self.output._df["grid2pv"], 0)

    def test_power_to_grid_pv_and_grid(self):
        # set inputs
        self.additional_setup_power_to_grid()
        inputs = pd.read_csv(os.path.join(test_folder, "output_calculator/power2grid_data2.csv"))
        self.output._df["pv_output"] = inputs["pv_output"].to_numpy()
        self.output._df["pv2bess"] = inputs["pv2bess"].to_numpy()
        self.output._df["grid2bess"] = inputs["grid2bess"].to_numpy()
        # check outputs
        self.output._calc_power_to_grid(0)
        nptesting.assert_array_almost_equal(self.output._df.loc[:, ["pv2grid", "bess2grid", "output", "grid2bess",
                                                                    "grid_from_pv", "grid_from_bess"]],
                                            pd.read_csv(os.path.join(test_folder,
                                                                     "output_calculator/power2grid_data2_results.csv")),
                                            0)
        nptesting.assert_array_equal(self.output._df["grid2pv"], 0)

    def test_power_to_grid_negative_pv(self):
        # set inputs
        self.additional_setup_power_to_grid()
        inputs = pd.read_csv(os.path.join(test_folder, "output_calculator/power2grid_data3.csv"))
        self.output._df["pv_output"] = inputs["pv_output"].to_numpy()
        self.output._df["pv2bess"] = inputs["pv2bess"].to_numpy()
        self.output._df["grid2bess"] = inputs["grid2bess"].to_numpy()
        # check outputs
        self.output._calc_power_to_grid(0)
        nptesting.assert_array_almost_equal(self.output._df.loc[:, ["pv2grid", "bess2grid", "output", "grid2bess",
                                                                    "grid2pv", "grid_from_pv", "grid_from_bess"]],
                                            pd.read_csv(os.path.join(test_folder,
                                                                     "output_calculator/power2grid_data3_results.csv")),
                                            0)

    def test_power_to_grid_early_discharge_hour_3_days(self):
        # set inputs
        self.additional_setup_power_to_grid()
        self.output._df = self.output._df.reindex(pd.date_range("2023-1-1 00:00", "2023-1-3 23:00", freq='h'),
                                                  method='nearest')
        self.output.bess_discharge_start_hour = 2
        self.output._initial_battery_soc = np.full(72, 2700)
        self.output._active_hourly_self_cons = np.full(72, 12)
        self.output._indices = np.arange(72)
        self.output._df["grid2bess"] = 0
        inputs = pd.read_csv(os.path.join(test_folder, "output_calculator/power2grid_data4.csv"))
        self.output._df["pv_output"] = inputs["pv_output"].to_numpy()
        self.output._df["pv2bess"] = inputs["pv2bess"].to_numpy()
        # check outputs
        self.output._calc_power_to_grid(0)
        pd.set_option('display.max_rows', None)
        nptesting.assert_array_almost_equal(self.output._df.loc[:, ["pv2grid", "bess2grid", "output", "grid2bess",
                                                                    "grid_from_pv", "grid_from_bess"]],
                                            pd.read_csv(os.path.join(test_folder,
                                                                     "output_calculator/power2grid_data4_results.csv")),
                                            0)
        nptesting.assert_array_equal(self.output._df["grid2pv"], 0)

    def test_soc_only_pv(self):
        # set inputs
        self.additional_setup()
        self.output._idle_hourly_self_cons = np.full(24, 6)
        self.output._df["grid2bess"] = 0
        inputs = pd.read_csv(os.path.join(test_folder, "output_calculator/soc_data1.csv"))
        self.output._df["pv2bess"] = inputs["pv2bess"].to_numpy()
        self.output._df["bess2grid"] = inputs["bess2grid"].to_numpy()
        # check outputs
        self.output._calc_soc(0, 1)
        nptesting.assert_array_almost_equal(self.output._df["soc"],
                                            pd.read_csv(os.path.join(test_folder,
                                                                     "output_calculator/soc_data1_results.csv"))[
                                                "soc"].to_numpy(), 0)

    def test_soc_pv_and_grid(self):
        # set inputs
        self.additional_setup()
        self.output._idle_hourly_self_cons = np.full(24, 6)
        inputs = pd.read_csv(os.path.join(test_folder, "output_calculator/soc_data2.csv"))
        self.output._df["pv2bess"] = inputs["pv2bess"].to_numpy()
        self.output._df["bess2grid"] = inputs["bess2grid"].to_numpy()
        self.output._df["grid2bess"] = inputs["grid2bess"].to_numpy()
        # check outputs
        self.output._calc_soc(0, 1)
        nptesting.assert_array_almost_equal(self.output._df["soc"],
                                            pd.read_csv(os.path.join(test_folder,
                                                                     "output_calculator/soc_data2_results.csv"))[
                                                "soc"].to_numpy(), 0)

    def test_soc_late_discharge_hour(self):
        # set inputs
        self.additional_setup()
        self.output._idle_hourly_self_cons = np.full(48, 6)
        self.output._active_hourly_self_cons = pd.Series(np.full(48, 12))
        self.output._indices = np.arange(48)
        self.output.bess_discharge_start_hour = 22
        self.output._df = self.output._df.reindex(pd.date_range("2025-1-1 00:00", "2025-1-2 23:00", freq='h'),
                                                  method='nearest')
        inputs = pd.read_csv(os.path.join(test_folder, "output_calculator/soc_data3.csv"), dtype=float)
        self.output._df["pv2bess"] = inputs["pv2bess"].to_numpy()
        self.output._df["bess2grid"] = inputs["bess2grid"].to_numpy()
        self.output._df["grid2bess"] = inputs["grid2bess"].to_numpy()
        # check outputs
        self.output._calc_soc(3, 1)
        nptesting.assert_array_almost_equal(self.output._df["soc"],
                                            pd.read_csv(os.path.join(test_folder,
                                                                     "output_calculator/soc_data3_results.csv"))[
                                                "soc"].to_numpy(), 0)

    def additional_setup_run(self):
        """
        additional setup for testing run
        """
        type(self.power_storage).aug_table = np.array([[0, 30, 3000], [96, 10, 1000]])
        self.output.grid_size = 1000
        self.output._pcs_power = 1140
        type(self.power_storage).degradation_table = np.full(31, 1)
        self.output._initial_data = pd.read_csv(os.path.join(test_folder, "output_calculator/run_data1.csv"))
        self.output._initial_data.index = pd.date_range("2023-1-1 00:00", "2023-12-31 23:00", freq='h')
        self.output._charge_loss = 0.015
        self.output._grid_bess_loss = 1 - 0.985 * 0.98

    def test_run_deg_table_ones(self):
        # set inputs
        self.additional_setup_run()
        # wrap methods with mocks
        self.output._get_data = Mock(side_effect=self.output._get_data)
        self.output._calc_overflow = Mock(side_effect=self.output._calc_overflow)
        self.output._calc_augmentations = Mock(side_effect=self.output._calc_augmentations)
        self.output._calc_pv_to_bess = Mock(side_effect=self.output._calc_pv_to_bess)
        self.output._calc_daily_initial_battery_soc = Mock(side_effect=self.output._calc_daily_initial_battery_soc)
        self.output._calc_grid_to_bess = Mock(side_effect=self.output._calc_grid_to_bess)
        self.output._calc_power_to_grid = Mock(side_effect=self.output._calc_power_to_grid)
        self.output._calc_soc = Mock(side_effect=self.output._calc_soc)
        # check results
        self.output.run()
        nptesting.assert_allclose(self.output.results[0].loc[:, ["pv2bess", "grid2bess", "pv2grid",
                                                                 "bess2grid", "output"]],
                                  pd.read_csv(os.path.join(test_folder, "output_calculator/run_data1_results.csv")),
                                  atol=3)
        nptesting.assert_array_equal(self.output.results[0]["grid2pv"], 0)
        nptesting.assert_array_almost_equal(self.output.results[0]["battery_capacity"], 2700, 0)
        self.assertEqual(self.output._next_aug, 2)
        self.assertEqual(self.output._first_aug_entries, [pd.Timestamp("2023-1-1 00:00"),
                                                          pd.Timestamp("2031-1-1 00:00")])
        self.assertEqual(self.output._get_data.call_count, self.output.num_of_years)
        self.assertEqual(self.output._calc_overflow.call_count, self.output.num_of_years)
        self.assertEqual(self.output._calc_augmentations.call_count, self.output.num_of_years)
        self.assertEqual(self.output._calc_pv_to_bess.call_count, self.output.num_of_years)
        self.assertEqual(self.output._calc_daily_initial_battery_soc.call_count, self.output.num_of_years)
        self.assertEqual(self.output._calc_grid_to_bess.call_count, self.output.num_of_years)
        self.assertEqual(self.output._calc_power_to_grid.call_count, self.output.num_of_years)
        self.assertEqual(self.output._calc_soc.call_count, self.output.num_of_years)

    def test_run_no_save_results(self):
        # set inputs
        self.additional_setup_run()
        self.output.save_all_results = False
        # replace methods with mock
        self.output._calc_overflow = Mock()
        self.output._calc_soc = Mock()
        # check outputs
        self.output.run()
        self.assertIsNone(self.output.results)
        self.output._calc_overflow.assert_not_called()
        self.output._calc_soc.assert_not_called()

    def test_run_no_buy_from_grid(self):
        # set inputs
        self.additional_setup_run()
        self.output.fill_battery_from_grid = False
        # replace method with mock
        self.output._calc_grid_to_bess = Mock()
        # check outputs
        self.output.run()
        self.output._calc_grid_to_bess.assert_not_called()

    def test_monthly_averages_values(self):
        # set inputs
        year1 = pd.read_csv(os.path.join(test_folder, "output_calculator/monthly_averages_data1_1.csv"))
        year1.index = pd.date_range("2023-1-1 00:00", "2023-12-31 23:00", freq='h')
        year2 = pd.read_csv(os.path.join(test_folder, "output_calculator/monthly_averages_data1_2.csv"))
        year2.index = pd.date_range("2024-1-1 00:00", "2024-12-31 23:00", freq='h')
        self.output._results = [year1, year2]
        # check outputs
        result = self.output.monthly_averages(years=(0, 1), stat="test")
        nptesting.assert_array_equal(result, np.loadtxt(os.path.join(test_folder,
                                                                     "output_calculator/"
                                                                     "monthly_averages_data1_results.csv"),
                                                        delimiter=","))

    def test_monthly_averages_output_values(self):
        # set inputs
        year1 = pd.read_csv(os.path.join(test_folder, "output_calculator/monthly_averages_data1_1.csv"))
        year1.index = pd.date_range("2023-1-1 00:00", "2023-12-31 23:00", freq='h')
        year1 = year1.squeeze()
        year2 = pd.read_csv(os.path.join(test_folder, "output_calculator/monthly_averages_data1_2.csv"))
        year2.index = pd.date_range("2024-1-1 00:00", "2024-12-31 23:00", freq='h')
        year2 = year2.squeeze()
        self.output._output = [year1, year2]
        # check outputs
        result = self.output.monthly_averages(years=(0, 1), stat="output")
        nptesting.assert_array_equal(result, np.loadtxt(os.path.join(test_folder,
                                                                     "output_calculator/"
                                                                     "monthly_averages_data1_results.csv"),
                                                        delimiter=","))

    def test_monthly_averages_wrong_year(self):
        # set inputs
        self.output._output = [np.full(8760, 0)]
        # check error raised
        with self.assertRaises(ValueError) as e:
            self.output.monthly_averages(years=(30,))
        self.assertEqual(str(e.exception), "years provided not in range of calculation years")

    def test_monthly_averages_wrong_stat(self):
        # set inputs
        self.output._results = [pd.DataFrame(np.full(8760, 0), index=np.arange(8760), columns=["output"])]
        # check error raised
        with self.assertRaises(ValueError) as e:
            self.output.monthly_averages(years=(0, 1), stat="test")
        self.assertEqual(str(e.exception), "Stat 'test' is not present in results")

    def test_monthly_averages_no_results(self):
        # set inputs
        self.output._results = None
        # check error raised
        with self.assertRaises(ValueError) as e:
            self.output.monthly_averages(stat="test")
        self.assertEqual(str(e.exception), "The calculator full results are not available (ether you didn't use run, or"
                                           " used save_all_result=False)")

    def test_monthly_averages_no_output(self):
        # check for error when function called
        with self.assertRaises(ValueError) as e:
            self.output.monthly_averages()
        self.assertEqual(str(e.exception), "The calculator results are not available (use run to generate output)")

    def test_plot_stat_wrong_stat(self):
        # set inputs
        self.output._results = [pd.DataFrame(np.full(8760, 0), index=np.arange(8760), columns=["test"])]
        # check error raised
        with self.assertRaises(ValueError) as e:
            self.output.plot_stat(years=(0, 1), stat="test1")
        self.assertEqual(str(e.exception), "Stat 'test1' is not present in results")

    def test_plot_stat_wrong_year(self):
        # set inputs
        self.output._results = [pd.DataFrame(np.full(8760, 0), index=np.arange(8760), columns=["test"])]
        # check error raised
        with self.assertRaises(ValueError) as e:
            self.output.plot_stat(years=(30,), stat="test")
        self.assertEqual(str(e.exception), "years provided not in range of calculation years")

    def test_plot_stat_no_results(self):
        # set inputs
        self.output._results = None
        # check error raised
        with self.assertRaises(ValueError) as e:
            self.output.plot_stat()
        self.assertEqual(str(e.exception), "The calculator full results are not available (either you didn't use run, "
                                           "or used save_all_result=False)")
