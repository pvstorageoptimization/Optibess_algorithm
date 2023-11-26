import json
import os
import unittest
from unittest.mock import Mock, patch, DEFAULT
import numpy as np
import pandas as pd
import pvlib.iotools

from optibess_algorithm.pv_output_calculator import get_pvlib_output, get_pvgis_hourly, Tech

test_folder = os.path.dirname(os.path.abspath(__file__))


class TestPvOutputCalculator(unittest.TestCase):

    def setUp(self) -> None:
        # mocks calls for pvgis server
        pvlib.iotools = Mock()
        pvlib.iotools.get_pvgis_tmy.return_value = [pd.read_csv(os.path.join(test_folder,
                                                                             "output_calculator/tmy_example.csv"),
                                                                parse_dates=True, index_col=0), ]
        patch("optibess_algorithm.pv_output_calculator.pvfactors_timeseries",
              return_value=[pd.read_csv(os.path.join(test_folder,
                                                     "output_calculator/irrad_example.csv"),
                                        parse_dates=True, index_col=0)]).start()
        pvlib.iotools.get_pvgis_hourly.return_value = [pd.read_csv(os.path.join(test_folder,
                                                                                "output_calculator"
                                                                                "/pvgis_hourly_data_example.csv"),
                                                                   parse_dates=True, index_col=0), ]

        # mocks calls to pvlib library to test functions called
        self.mock_fixed_mount = patch("pvlib.pvsystem.FixedMount").start()
        self.mock_single_axis_tracker_mount = patch("pvlib.pvsystem.SingleAxisTrackerMount").start()
        self.mock_array = patch("pvlib.pvsystem.Array").start()
        self.mock_pvsystem = patch("pvlib.pvsystem.PVSystem").start()
        self.model_chain = patch('pvlib.modelchain.ModelChain').start()
        self.model_chain.return_value.results.ac = pd.read_csv(os.path.join(test_folder,
                                                                            "output_calculator"
                                                                            "/model_chain_ac_results_example.csv"),
                                                               parse_dates=True, index_col=0).squeeze()

    def tearDown(self) -> None:
        patch.stopall()

    def test_pvlib_output_fixed(self):
        result = get_pvlib_output(latitude=30, longitude=34, modules_per_string=10, number_of_inverters=100)
        # check data shape
        self.assertEqual(result.shape[0], 8760, "pvlib output is not in the right shape (Should have 8760 rows)")
        # check values are in reasonable range
        self.assertTrue(np.all(result.iloc[0] < 25))
        # check function calls
        self.mock_fixed_mount.assert_called_once()
        self.mock_array.assert_called_once()
        self.mock_pvsystem.assert_called_once()
        self.model_chain.assert_called_once()
        self.model_chain.return_value.run_model.assert_called_once()

    def test_pvlib_output_fixed_cec_module(self):
        # get module data
        with open(os.path.join(test_folder, "output_calculator/pv_module_data.json")) as module_data_file:
            json_content = json.load(module_data_file)
            module_data = json_content[0]
        # change model chain implementation
        self.model_chain.side_effect = [ValueError(), DEFAULT]
        # call function
        result = get_pvlib_output(latitude=30, longitude=34, modules_per_string=10, number_of_inverters=100,
                                  module=pd.Series(module_data))
        # check data shape
        self.assertEqual(result.shape[0], 8760, "pvlib output is not in the right shape (Should have 8760 rows)")
        # check values are in reasonable range
        self.assertTrue(np.all(result.iloc[0] < 25))
        # check function calls
        self.mock_fixed_mount.assert_called_once()
        self.mock_array.assert_called_once()
        self.mock_pvsystem.assert_called_once()
        self.assertEqual(self.model_chain.call_count, 2)
        self.assertEqual(self.model_chain.call_args[1]['aoi_model'], 'no_loss')
        self.assertEqual(self.model_chain.call_args[1]['spectral_model'], 'no_loss')
        self.model_chain.return_value.run_model.assert_called_once()

    def test_pvlib_output_tracker(self):
        result = get_pvlib_output(latitude=30, longitude=34, modules_per_string=10, number_of_inverters=100,
                                  tech=Tech.TRACKER)
        # check data shape
        self.assertEqual(result.shape[0], 8760, "pvlib output is not in the right shape (Should have 8760 rows)")
        # check values are not nan
        self.assertFalse(np.isnan(result).any())
        # check function calls
        self.mock_single_axis_tracker_mount.assert_called_once()
        self.mock_array.assert_called_once()
        self.mock_pvsystem.assert_called_once()
        self.model_chain.assert_called_once()
        self.model_chain.return_value.run_model.assert_called_once()

    def test_pvlib_output_east_west(self):
        result = get_pvlib_output(latitude=30, longitude=34, modules_per_string=10, number_of_inverters=100,
                                  tech=Tech.EAST_WEST)
        # check data shape
        self.assertEqual(result.shape[0], 8760, "pvlib output is not in the right shape (Should have 8760 rows)")
        # check values are in reasonable range
        self.assertTrue(np.all(result.iloc[0] < 25))
        # check function calls
        self.assertEqual(self.mock_fixed_mount.call_count, 2)
        self.assertEqual(self.mock_array.call_count, 2)
        self.mock_pvsystem.assert_called_once()
        self.model_chain.assert_called_once()
        self.model_chain.return_value.run_model.assert_called_once()

    def test_pvlib_output_fixed_cec_module_bifacial(self):
        # get module data for bifacial
        with open(os.path.join(test_folder, "output_calculator/pv_module_data.json")) as module_data_file:
            json_content = json.load(module_data_file)
            module_data = json_content[0]
        # mock location
        pvlib.location.Location = patch("pvlib.location.Location").start()
        pvlib.location.Location.return_value.get_solarposition.return_value = \
            pd.read_csv(os.path.join(test_folder, "output_calculator/solar_position_example.csv"),
                        parse_dates=True, index_col=0)
        # call function
        result = get_pvlib_output(latitude=30, longitude=34, modules_per_string=10, number_of_inverters=100,
                                  use_bifacial=True, module=pd.Series(module_data))
        # check data shape
        self.assertEqual(result.shape[0], 8760, "pvlib output is not in the right shape (Should have 8760 rows)")
        # check function calls
        self.mock_fixed_mount.assert_called_once()
        self.mock_array.assert_called_once()
        self.mock_pvsystem.assert_called_once()
        self.model_chain.assert_called_once()
        self.model_chain.return_value.run_model_from_effective_irradiance.assert_called_once()

    def test_pvlib_output_fixed_non_bifacial_cec_module_bifacial(self):
        # get module data for non bifacial module
        with open(os.path.join(test_folder, "output_calculator/pv_module_data.json")) as module_data_file:
            json_content = json.load(module_data_file)
            module_data = json_content[2]
        # mock location
        pvlib.location.Location = patch("pvlib.location.Location").start()
        pvlib.location.Location.return_value.get_solarposition.return_value = \
            pd.read_csv(os.path.join(test_folder, "output_calculator/solar_position_example.csv"),
                        parse_dates=True, index_col=0)
        # call function
        result = get_pvlib_output(latitude=30, longitude=34, modules_per_string=10, number_of_inverters=100,
                                  use_bifacial=True, module=pd.Series(module_data))
        # check data shape
        self.assertEqual(result.shape[0], 8760, "pvlib output is not in the right shape (Should have 8760 rows)")
        # check function calls
        self.mock_fixed_mount.assert_called_once()
        self.mock_array.assert_called_once()
        self.mock_pvsystem.assert_called_once()
        self.model_chain.assert_called_once()
        self.model_chain.return_value.run_model_from_effective_irradiance.assert_not_called()

    def test_pvlib_output_fixed_sandia_module_bifacial(self):
        # mock location
        pvlib.location.Location = patch("pvlib.location.Location").start()
        pvlib.location.Location.return_value.get_solarposition.return_value = \
            pd.read_csv(os.path.join(test_folder, "output_calculator/solar_position_example.csv"),
                        parse_dates=True, index_col=0)
        # call function
        result = get_pvlib_output(latitude=30, longitude=34, modules_per_string=10, number_of_inverters=100,
                                  use_bifacial=True)
        # check data shape
        self.assertEqual(result.shape[0], 8760, "pvlib output is not in the right shape (Should have 8760 rows)")
        # check function calls
        self.mock_fixed_mount.assert_called_once()
        self.mock_array.assert_called_once()
        self.mock_pvsystem.assert_called_once()
        self.model_chain.assert_called_once()
        self.model_chain.return_value.run_model_from_effective_irradiance.assert_not_called()

    def test_pvlib_output_incorrect_arg(self):
        # check error is raised when number of inverters is 0
        with self.assertRaises(ValueError) as e:
            get_pvlib_output(latitude=30, longitude=34, number_of_inverters=0)
        self.assertEqual(str(e.exception), "Number of units should be positive")

    def test_pvgis_output_fixed(self):
        result = get_pvgis_hourly(latitude=30, longitude=34)
        # check data shape
        self.assertEqual(result.shape[0], 8760, "pvgis output is not in the right shape (Should have 8760 rows)")
        # check values are in reasonable range
        self.assertTrue(np.all(result.iloc[0] < 225))

    def test_pvgis_output_tracker(self):
        result = get_pvgis_hourly(latitude=30, longitude=34, tech=Tech.TRACKER)
        # check data shape
        self.assertEqual(result.shape[0], 8760, "pvgis output is not in the right shape (Should have 8760 rows)")
        # check values are in reasonable range
        self.assertTrue(np.all(result.iloc[0] < 225))

    def test_pvgis_output_east_west(self):
        result = get_pvgis_hourly(latitude=30, longitude=34, tech=Tech.EAST_WEST)
        # check data shape
        self.assertEqual(result.shape[0], 8760, "pvgis output is not in the right shape (Should have 8760 rows)")
        # check values are in reasonable range
        self.assertTrue(np.all(result.iloc[0] < 225))
