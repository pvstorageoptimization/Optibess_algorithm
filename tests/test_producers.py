import json
import unittest
from unittest.mock import patch

import pandas as pd
import pytz

from Optibess_algorithm.producers import PvProducer
from Optibess_algorithm.constants import *

test_folder = os.path.dirname(os.path.abspath(__file__))


class TestProducer(unittest.TestCase):

    def test_creation_with_file(self):
        result = PvProducer(pv_output_file=os.path.join(test_folder, "test.csv"), pv_peak_power=12000)
        # check output size
        self.assertEqual(result.power_output.shape[0], 8760, "pv output should have 8760 entries")
        # check selected values
        self.assertEqual(result.power_output.columns, ['pv_output'])
        self.assertEqual(result.power_output.index[0].day, 1)
        self.assertEqual(result.power_output.index[0].month, 1)
        self.assertEqual(result.power_output.index[0].hour, 0)
        self.assertEqual(result.power_output['pv_output'][9], 3648.1)

    @patch('Optibess_algorithm.producers.get_pvlib_output')
    def test_creation_with_pvlib(self, pvlib_output_calc):
        # mock call for function to generated data from pvlib
        pvlib_output_calc.return_value = pd.read_csv(os.path.join(test_folder,
                                                                  "output_calculator/pvlib_data_example.csv"),
                                                     index_col=0,
                                                     parse_dates=True).squeeze()
        result = PvProducer(latitude=30, longitude=34, number_of_inverters=1000)
        # check output size
        self.assertEqual(result.power_output.shape[0], 8760, "pv output should have 8760 entries")
        # check selected values
        self.assertEqual(result.power_output.index[2].day, 1)
        self.assertEqual(result.power_output.index[2].month, 1)
        self.assertEqual(result.power_output.index[2].hour, 2)
        self.assertGreater(result.power_output['pv_output'][10], 0)
        self.assertEqual(round(result.rated_power), 720)

    @patch('Optibess_algorithm.producers.get_pvlib_output')
    def test_creation_with_pvlib_cec_module(self, pvlib_output_calc):
        # mock call for function to generated data from pvlib
        pvlib_output_calc.return_value = pd.read_csv(os.path.join(test_folder,
                                                                  "output_calculator/pvlib_data_example.csv"),
                                                     index_col=0,
                                                     parse_dates=True).squeeze()
        with open(os.path.join(test_folder, "output_calculator/pv_module_data.json")) as module_data_file:
            json_content = json.load(module_data_file)
            module_data = json_content[0]
        result = PvProducer(latitude=30, longitude=34, number_of_inverters=1000, module=pd.Series(module_data))
        # check rated power
        self.assertEqual(result.rated_power, 720)

    @patch('Optibess_algorithm.producers.get_pvgis_hourly')
    def test_creation_with_pvgis(self, pvgis_output_calc):
        # mock call for function to generated data from pvgis
        pvgis_output_calc.return_value = pd.read_csv(os.path.join(test_folder,
                                                                  "output_calculator/pvgis_hourly_data_example2.csv"),
                                                     index_col=0, parse_dates=True).squeeze()

        result = PvProducer(latitude=30, longitude=34, pv_peak_power=10000)
        # check output size
        self.assertEqual(result.power_output.shape[0], 8760, "pv output should have 8760 entries")
        # check selected values
        self.assertEqual(result.power_output.index[25].day, 2)
        self.assertEqual(result.power_output.index[2].month, 1)
        self.assertEqual(result.power_output.index[2].hour, 2)
        self.assertGreater(result.power_output['pv_output'][11], 0)

    def test_creation_no_args(self):
        with self.assertRaises(ValueError) as e:
            PvProducer()
        self.assertEqual(str(e.exception), "Missing values for parameters for pvgis option")

    def test_creation_no_latitude(self):
        with self.assertRaises(ValueError) as e:
            PvProducer(longitude=34, pv_peak_power=10000)
        self.assertEqual(str(e.exception), "Missing values for parameters for pvgis option")

    def test_creation_pvlib_no_lat(self):
        with self.assertRaises(ValueError) as e:
            PvProducer(longitude=34, number_of_inverters=1000)
        self.assertEqual(str(e.exception), "Missing values for parameters for pvlib option")

    def test_creation_file_no_pv_peak_power(self):
        with self.assertRaises(ValueError) as e:
            PvProducer(pv_output_file=os.path.join(test_folder, "test.csv"))
        self.assertEqual(str(e.exception), "PV peak power should have value for options other than pvlib")

    def test_creation_file_negative_pv_peak_power(self):
        with self.assertRaises(ValueError) as e:
            PvProducer(pv_output_file=os.path.join(test_folder, "test.csv"), pv_peak_power=-3)
        self.assertEqual(str(e.exception), "PV peak power should be non negative")

    def test_creation_file_incorrect_time_zone(self):
        with self.assertRaises(pytz.exceptions.UnknownTimeZoneError) as e:
            PvProducer(pv_output_file=os.path.join(test_folder, "test.csv"), pv_peak_power=12000, time_zone="Asia/Jeru")

    def test_creation_incorrect_file_extension(self):
        with self.assertRaises(ValueError) as e:
            PvProducer(pv_output_file=os.path.join(test_folder, "test.xsl"), pv_peak_power=10000)
        self.assertEqual(str(e.exception), "PV file should be of type csv")

    def test_creation_incorrect_file_length(self):
        with self.assertRaises(ValueError) as e:
            PvProducer(pv_output_file=os.path.join(test_folder, "test_tranc.csv"), pv_peak_power=10000)
        self.assertEqual(str(e.exception), "Number of lines in file should be dividable by number of hours in a year "
                                           "(8670)")

    def test_creation_non_numeric_file(self):
        with self.assertRaises(ValueError) as e:
            PvProducer(pv_output_file=os.path.join(test_folder, "test_non_numeric.csv"), pv_peak_power=10000)
        self.assertTrue(str(e.exception).startswith("could not convert string to float"))

    @patch('Optibess_algorithm.producers.get_pvlib_output')
    def test_creation_pvlib_germany_time_zone(self, pvlib_output_calc):
        # mock call for function to generated data from pvlib
        pvlib_output_calc.return_value = pd.read_csv(os.path.join(test_folder,
                                                                  "output_calculator/pvlib_germany_output.csv"),
                                                     index_col=0,
                                                     parse_dates=True).squeeze()

        result = PvProducer(latitude=52.5, longitude=13, number_of_inverters=1000)
        self.assertEqual(result._time_zone, "Europe/Berlin")

    def test_creation_incorrect_latitude(self):
        with self.assertRaises(ValueError) as e:
            PvProducer(latitude=100)
        self.assertEqual(str(e.exception), f"Latitude value should be between {LATITUDE.min} and "
                                           f"{LATITUDE.max}")

    def test_creation_incorrect_longitude(self):
        with self.assertRaises(ValueError) as e:
            PvProducer(longitude=200)
        self.assertEqual(str(e.exception), f"Longitude value should be between {LONGITUDE.min} and "
                                           f"{LONGITUDE.max}")

    def test_creation_incorrect_number_of_modules(self):
        with self.assertRaises(ValueError) as e:
            PvProducer(modules_per_string=0)
        self.assertEqual(str(e.exception), "Number of modules per string should be positive")

    def test_creation_incorrect_string_per_inverter(self):
        with self.assertRaises(ValueError) as e:
            PvProducer(strings_per_inverter=-2)
        self.assertEqual(str(e.exception), "Number of strings per inverters should be positive")

    def test_creation_incorrect_number_of_inverters(self):
        with self.assertRaises(ValueError) as e:
            PvProducer(number_of_inverters=-5)
        self.assertEqual(str(e.exception), "Number of inverters should be positive")

    def test_creation_incorrect_tilt(self):
        with self.assertRaises(ValueError) as e:
            PvProducer(tilt=-5)
        self.assertEqual(str(e.exception), f"Tilt should be between {TILT.min} and {TILT.max}")

    def test_creation_incorrect_azimuth(self):
        with self.assertRaises(ValueError) as e:
            PvProducer(azimuth=367)
        self.assertEqual(str(e.exception), f"Azimuth should be between {AZIMUTH.min} and {AZIMUTH.max}")

    def test_creation_incorrect_losses(self):
        with self.assertRaises(ValueError) as e:
            PvProducer(losses=-15)
        self.assertEqual(str(e.exception), "Losses percentage should be non negative")

    def test_creation_incorrect_pv_peak_power(self):
        with self.assertRaises(ValueError) as e:
            PvProducer(latitude=30, longitude=34, pv_peak_power=-1)
        self.assertEqual(str(e.exception), "PV peak power should be non negative")

    def test_set_incorrect_annual_deg(self):
        result = PvProducer(pv_output_file=os.path.join(test_folder, "test.csv"), pv_peak_power=12000)
        with self.assertRaises(ValueError) as e:
            result.annual_deg = -1
        self.assertEqual(str(e.exception), "Annual degradation should be between 0 (inclusive) and 1 (exclusive)")
