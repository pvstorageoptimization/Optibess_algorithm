import unittest
from unittest.mock import Mock, patch
from numpy import testing as nptesting
import pandas as pd

from Optibess_algorithm.financial_calculator import FinancialCalculator
from Optibess_algorithm.output_calculator import OutputCalculator
from Optibess_algorithm.power_storage import PowerStorage
from Optibess_algorithm.producers import Producer
from Optibess_algorithm.constants import *

test_folder = os.path.dirname(os.path.abspath(__file__))


class TestFinancialCalculator(unittest.TestCase):

    def setUp(self) -> None:
        # create mock producer
        producer = Mock(spec=Producer)
        type(producer).power_output = pd.DataFrame(np.zeros((8760,)),
                                                   pd.date_range(start='2023-1-1 00:00',
                                                                 end='2023-12-31 23:00', freq='h'), ['pv_output'])
        type(producer)._producer_factor = 1
        # create mock power storage
        power_storage = Mock(spec=PowerStorage)
        type(power_storage).num_of_years = 25
        type(power_storage).aug_table = np.array([[0, 70, 7000], [96, 20, 2000], [192, 10, 1000]])

        # create default output calculator
        self.output = Mock(spec=OutputCalculator)
        type(self.output).num_of_years = 25
        type(self.output).grid_size = 5000
        type(self.output).pcs_power = 6265
        type(self.output).producer = producer
        type(self.output).power_storage = power_storage
        type(self.output).rated_power = 12000
        type(self.output).aug_table = np.array([[0, 70, 7000], [96, 20, 2000], [192, 10, 1000]])

    def test_creation_regular(self):
        # created financial calculator
        result = FinancialCalculator(output_calculator=self.output, land_size=100, capex_per_land_unit=100,
                                     opex_per_land_unit=1, capex_per_kwp=100, opex_per_kwp=1, battery_capex_per_kwh=100,
                                     battery_opex_per_kwh=1, battery_connection_capex_per_kw=100,
                                     battery_connection_opex_per_kw=1, usd_to_ils=3.5)
        # check outputs
        self.assertEqual(result.land_capex, 10000)
        self.assertEqual(result.land_opex, 100)
        self.assertEqual(result.total_producer_capex, 4200000)
        self.assertEqual(result.total_producer_opex, 42000)
        battery_data = np.loadtxt(os.path.join(test_folder, "financial_calculator/creation_battery_data1.csv"),
                                  delimiter=",")
        nptesting.assert_array_equal(result._battery_size, battery_data[0])
        nptesting.assert_array_equal(result.battery_cost, battery_data[1])
        nptesting.assert_array_equal(result.battery_opex, battery_data[2])
        self.assertEqual(result.battery_connection_capex, 2192750)
        self.assertEqual(result.battery_connection_opex, 21927.5)

    def test_creation_with_tariff_table(self):
        # created financial calculator
        result = FinancialCalculator(output_calculator=self.output, land_size=100, capex_per_land_unit=100,
                                     opex_per_land_unit=1, capex_per_kwp=100, opex_per_kwp=1, battery_capex_per_kwh=100,
                                     battery_opex_per_kwh=1, battery_connection_capex_per_kw=100,
                                     battery_connection_opex_per_kw=1, usd_to_ils=3.5, tariff_table=np.ones((12, 24)))
        # check outputs
        nptesting.assert_array_equal(result.tariff_table, np.ones((7, 12, 24)))

    def test_set_usd_to_ils_after_creation(self):
        # create financial calculator
        finance = FinancialCalculator(output_calculator=self.output)
        # setup mocks
        finance._set_rated_power = Mock(side_effect=finance._set_rated_power)
        finance._set_battery_size = Mock(side_effect=finance._set_battery_size)
        finance._set_battery_connection_size = Mock(side_effect=finance._set_battery_connection_size)

        # set usd_to_ils
        finance.usd_to_ils = 3.1

        # check calls
        finance._set_rated_power.assert_called_once_with(finance._pv_peak_power)
        finance._set_battery_size.assert_called_once_with(finance._output_calculator.aug_table)
        finance._set_battery_connection_size.assert_called_once_with(finance._battery_connection_size)

    def test_creation_incorrect_land_size(self):
        with self.assertRaises(ValueError) as e:
            FinancialCalculator(output_calculator=self.output, land_size=0)
        self.assertEqual(str(e.exception), "Land size should be positive")

    def test_creation_incorrect_land_capex(self):
        with self.assertRaises(ValueError) as e:
            FinancialCalculator(output_calculator=self.output, capex_per_land_unit=-1)
        self.assertEqual(str(e.exception), "Capex per land unit should be non negative")

    def test_creation_incorrect_land_opex(self):
        with self.assertRaises(ValueError) as e:
            FinancialCalculator(output_calculator=self.output, opex_per_land_unit=-2)
        self.assertEqual(str(e.exception), "Opex per land unit should be non negative")

    def test_set_usd_to_ils_incorrect_value(self):
        result = FinancialCalculator(output_calculator=self.output)
        with self.assertRaises(ValueError) as e:
            result.usd_to_ils = -1
        self.assertEqual(str(e.exception), "Dollar to shekel exchange should be non negative")

    def test_creation_incorrect_power_capex(self):
        with self.assertRaises(ValueError) as e:
            FinancialCalculator(output_calculator=self.output, capex_per_kwp=-4)
        self.assertEqual(str(e.exception), "Capex per kWp should be non negative")

    def test_creation_incorrect_power_opex(self):
        with self.assertRaises(ValueError) as e:
            FinancialCalculator(output_calculator=self.output, opex_per_kwp=-67)
        self.assertEqual(str(e.exception), "Opex per kWp should be non negative")

    def test_creation_incorrect_battery_capex(self):
        with self.assertRaises(ValueError) as e:
            FinancialCalculator(output_calculator=self.output, battery_capex_per_kwh=-3)
        self.assertEqual(str(e.exception), "Capex per battery kWh should be non negative")

    def test_creation_incorrect_battery_opex(self):
        with self.assertRaises(ValueError) as e:
            FinancialCalculator(output_calculator=self.output, battery_opex_per_kwh=-10)
        self.assertEqual(str(e.exception), "Opex per battery kWh should be non negative")

    def test_creation_incorrect_battery_connection_capex(self):
        with self.assertRaises(ValueError) as e:
            FinancialCalculator(output_calculator=self.output, battery_connection_capex_per_kw=-5)
        self.assertEqual(str(e.exception), "Capex per kw of battery connection should be non negative")

    def test_creation_incorrect_battery_connection_opex(self):
        with self.assertRaises(ValueError) as e:
            FinancialCalculator(output_calculator=self.output, battery_connection_opex_per_kw=-36)
        self.assertEqual(str(e.exception), "Opex per kw of battery connection should be non negative")

    def test_creation_incorrect_tariff_table(self):
        with self.assertRaises(ValueError) as e:
            FinancialCalculator(output_calculator=self.output, tariff_table=np.ones((15, 24)))
        self.assertEqual(str(e.exception), "Tariff table should be of shape (12, 24)")

    def test_set_fixed_capex_incorrect_value(self):
        result = FinancialCalculator(output_calculator=self.output)
        with self.assertRaises(ValueError) as e:
            result.fixed_capex = -2
        self.assertEqual(str(e.exception), "Fixed capex should be non negative")

    def test_set_fixed_opex_incorrect_value(self):
        result = FinancialCalculator(output_calculator=self.output)
        with self.assertRaises(ValueError) as e:
            result.fixed_opex = -0.3
        self.assertEqual(str(e.exception), "Fixed opex should be non negative")

    def test_set_interest_rate_incorrect_value(self):
        result = FinancialCalculator(output_calculator=self.output)
        with self.assertRaises(ValueError) as e:
            result.interest_rate = -0.5
        self.assertEqual(str(e.exception), "Interest rate should be non negative")

    def test_set_cpi_incorrect_value(self):
        result = FinancialCalculator(output_calculator=self.output)
        with self.assertRaises(ValueError) as e:
            result.cpi = -3
        self.assertEqual(str(e.exception), "CPI should be non negative")

    def test_set_battery_cost_deg_incorrect_value(self):
        result = FinancialCalculator(output_calculator=self.output)
        with self.assertRaises(ValueError) as e:
            result.battery_cost_deg = 1
        self.assertEqual(str(e.exception), "Battery cost degradation should be between 0 (inclusive) and 1 (exclusive)")

    def test_set_buy_from_grid_factor_incorrect_value(self):
        result = FinancialCalculator(output_calculator=self.output)
        with self.assertRaises(ValueError) as e:
            result.buy_from_grid_factor = -0.7
        self.assertEqual(str(e.exception), "Factor for buying from grid should be non negative")

    def test_set_output_calculator_incorrect_type(self):
        result = FinancialCalculator(output_calculator=self.output)
        with self.assertRaises(ValueError) as e:
            result.output_calculator = np.zeros((4, 5))
        self.assertEqual(str(e.exception), "Output calculator should be an instance of OutputCalculator")

    def test_creation_build_tariff_table_check(self):
        result = FinancialCalculator(output_calculator=self.output)
        data = np.loadtxt(os.path.join(test_folder, "financial_calculator/build_tariff_table_result.csv"),
                          delimiter=",")
        nptesting.assert_array_almost_equal(result._tariff_table, data.reshape((7, 12, 24)), 2)

    def test_get_hourly_tariff_values(self):
        finance = FinancialCalculator(output_calculator=self.output)
        result = finance.get_hourly_tariff(2023)
        nptesting.assert_allclose(result, np.loadtxt(os.path.join(test_folder,
                                                                  "financial_calculator/hourly_tariff_output.csv")),
                                  atol=0.01)

    @patch.object(pd, 'date_range', side_effect=pd.date_range)
    def test_get_hourly_tariff_call_again(self, mock_date_range):
        finance = FinancialCalculator(output_calculator=self.output)
        finance.get_hourly_tariff(2023)
        finance.get_hourly_tariff(2023)

        # check only created once
        mock_date_range.assert_called_once()

    def addition_setup_power_sales(self):
        """
        do additional setup for the power sales tests
        """
        type(self.output).num_of_years = 2
        data = np.loadtxt(os.path.join(test_folder, "financial_calculator/power_sales_data1.csv"), delimiter=",")
        power_output = [pd.Series(data[:, 0], pd.date_range(start='2023-1-1 00:00',
                                                            end='2023-12-31 23:00', freq='h')),
                        pd.Series(data[:, 1], pd.date_range(start='2023-1-1 00:00',
                                                            end='2023-12-31 23:00', freq='h'))
                        ]
        finance = FinancialCalculator(output_calculator=self.output, cpi=0.025)
        finance.get_hourly_tariff = Mock(return_value=np.loadtxt(os.path.join(test_folder,
                                                                              "financial_calculator/hourly_tariff_output.csv")))
        return finance, power_output

    def test_get_power_sales_no_purchase(self):
        # set inputs
        purchased_from_grid = [np.zeros(8760), np.zeros(8760)]
        finance, power_output = self.addition_setup_power_sales()
        # check outputs
        result = finance.get_power_sales(power_output=power_output, purchased_from_grid=purchased_from_grid)
        nptesting.assert_array_almost_equal(result, [156309.9, 320435.3], 2)
        self.assertEqual(finance.get_hourly_tariff.call_count, self.output.num_of_years)

    def test_get_power_sales_with_purchase(self):
        # set inputs
        finance, power_output = self.addition_setup_power_sales()
        purchase_data = np.loadtxt(os.path.join(test_folder, "financial_calculator/purchased_power_data1.csv"),
                                   delimiter=",")
        purchased_from_grid = [pd.Series(purchase_data[:, 0]), pd.Series(purchase_data[:, 1])]
        # check outputs
        result = finance.get_power_sales(power_output=power_output, purchased_from_grid=purchased_from_grid)
        nptesting.assert_array_almost_equal(result, [85782.9, 248145.12], 2)
        self.assertEqual(finance.get_hourly_tariff.call_count, self.output.num_of_years)

    def test_power_sales_calculator_data_no_purchase(self):
        # set inputs
        finance, power_output = self.addition_setup_power_sales()
        type(self.output).output = power_output
        purchased_from_grid = [np.zeros(8760), np.zeros(8760)]
        # check outputs
        result = finance.get_power_sales(purchased_from_grid=purchased_from_grid)
        nptesting.assert_array_almost_equal(result, [156309.9, 320435.3], 2)
        self.assertEqual(finance.get_hourly_tariff.call_count, self.output.num_of_years)

    def test_power_sales_calculator_data_with_purchase(self):
        # set inputs
        finance, power_output = self.addition_setup_power_sales()
        type(self.output).output = power_output
        purchase_data = np.loadtxt(os.path.join(test_folder, "financial_calculator/purchased_power_data1.csv"),
                                   delimiter=",")
        purchased_from_grid = [pd.Series(purchase_data[:, 0]), pd.Series(purchase_data[:, 1])]
        # check outputs
        result = finance.get_power_sales(purchased_from_grid=purchased_from_grid)
        nptesting.assert_array_almost_equal(result, [85782.9, 248145.12], 2)
        self.assertEqual(finance.get_hourly_tariff.call_count, self.output.num_of_years)

    def test_power_sales_calculator_data_no_purchase_flag(self):
        finance, power_output = self.addition_setup_power_sales()
        type(self.output).output = power_output
        purchase_data = np.loadtxt(os.path.join(test_folder, "financial_calculator/purchased_power_data1.csv"),
                                   delimiter=",")
        purchased_from_grid = [pd.Series(purchase_data[:, 0]), pd.Series(purchase_data[:, 1])]
        # check outputs
        result = finance.get_power_sales(purchased_from_grid=purchased_from_grid, no_purchase=True)
        nptesting.assert_array_almost_equal(result, [156309.9, 320435.3], 2)
        self.assertEqual(finance.get_hourly_tariff.call_count, self.output.num_of_years)

    def test_power_sales_calculator_data_no_data_created(self):
        finance, power_output = self.addition_setup_power_sales()
        type(self.output).output = None
        type(self.output).purchased_from_grid = None

        def run_mock():
            type(self.output).output = power_output
            purchase_data = np.loadtxt(os.path.join(test_folder, "financial_calculator/purchased_power_data1.csv"),
                                       delimiter=",")
            purchased_from_grid = [pd.Series(purchase_data[:, 0]), pd.Series(purchase_data[:, 1])]
            type(self.output).purchased_from_grid = purchased_from_grid

        self.output.run = Mock(side_effect=run_mock)
        # check outputs
        result = finance.get_power_sales()
        nptesting.assert_array_almost_equal(result, [85782.9, 248145.12], 2)
        self.assertEqual(finance.get_hourly_tariff.call_count, self.output.num_of_years)
        self.output.run.assert_called_once()

    def test_power_sales_calculator_data_no_purchase_data_created(self):
        finance, power_output = self.addition_setup_power_sales()
        type(self.output).output = power_output
        type(self.output).purchased_from_grid = None

        def run_mock():
            purchase_data = np.loadtxt(os.path.join(test_folder, "financial_calculator/purchased_power_data1.csv"),
                                       delimiter=",")
            purchased_from_grid = [pd.Series(purchase_data[:, 0]), pd.Series(purchase_data[:, 1])]
            type(self.output).purchased_from_grid = purchased_from_grid

        self.output.run = Mock(side_effect=run_mock)
        # check outputs
        result = finance.get_power_sales()
        nptesting.assert_array_almost_equal(result, [85782.9, 248145.12], 2)
        self.assertEqual(finance.get_hourly_tariff.call_count, self.output.num_of_years)
        self.output.run.assert_called_once()

    def test_get_power_purchased_cost_regular(self):
        # set inputs
        finance, _ = self.addition_setup_power_sales()
        purchase_data = np.loadtxt(os.path.join(test_folder, "financial_calculator/purchased_power_data1.csv"),
                                   delimiter=",")
        purchased_from_grid = [pd.Series(purchase_data[:, 0], pd.date_range(start='2023-1-1 00:00',
                                                                            end='2023-12-31 23:00', freq='h')),
                               pd.Series(purchase_data[:, 1], pd.date_range(start='2023-1-1 00:00',
                                                                            end='2023-12-31 23:00', freq='h'))
                               ]
        # check outputs
        result = finance.get_power_purchases_cost(purchased_from_grid)
        nptesting.assert_array_almost_equal(result, [70527, 72290.17], 2)
        self.assertEqual(finance.get_hourly_tariff.call_count, self.output.num_of_years)

    def test_power_purchased_cost_calculation_data(self):
        # set inputs
        finance, _ = self.addition_setup_power_sales()
        purchase_data = np.loadtxt(os.path.join(test_folder, "financial_calculator/purchased_power_data1.csv"),
                                   delimiter=",")
        purchased_from_grid = [pd.Series(purchase_data[:, 0], pd.date_range(start='2023-1-1 00:00',
                                                                            end='2023-12-31 23:00', freq='h')),
                               pd.Series(purchase_data[:, 1], pd.date_range(start='2023-1-1 00:00',
                                                                            end='2023-12-31 23:00', freq='h'))
                               ]
        type(self.output).purchased_from_grid = purchased_from_grid
        # check outputs
        result = finance.get_power_purchases_cost()
        nptesting.assert_array_almost_equal(result, [70527, 72290.17], 2)
        self.assertEqual(finance.get_hourly_tariff.call_count, self.output.num_of_years)

    def test_power_purchased_cost_calculation_data_no_data_created(self):
        # set inputs
        finance, _ = self.addition_setup_power_sales()
        type(self.output).purchased_from_grid = None

        def run_mock():
            purchase_data = np.loadtxt(os.path.join(test_folder, "financial_calculator/purchased_power_data1.csv"),
                                       delimiter=",")
            purchased_from_grid = [pd.Series(purchase_data[:, 0], pd.date_range(start='2023-1-1 00:00',
                                                                                end='2023-12-31 23:00', freq='h')),
                                   pd.Series(purchase_data[:, 1], pd.date_range(start='2023-1-1 00:00',
                                                                                end='2023-12-31 23:00', freq='h'))
                                   ]
            type(self.output).purchased_from_grid = purchased_from_grid

        self.output.run = Mock(side_effect=run_mock)
        # check outputs
        result = finance.get_power_purchases_cost()
        nptesting.assert_array_almost_equal(result, [70527, 72290.17], 2)
        self.assertEqual(finance.get_hourly_tariff.call_count, self.output.num_of_years)
        self.output.run.assert_called_once()

    def test_get_expenses(self):
        # set inputs
        finance = FinancialCalculator(self.output)
        finance._land_capex = 10000
        finance._land_opex = 100
        finance._total_producer_capex = 4200000
        finance._total_producer_opex = 42000
        battery_data = np.loadtxt(os.path.join(test_folder, "financial_calculator/creation_battery_data1.csv"),
                                  delimiter=",")
        finance._battery_cost = battery_data[1]
        finance._battery_opex = battery_data[2]
        finance._battery_connection_capex = 2192750
        finance._battery_connection_opex = 21927.5
        finance._fixed_capex = 10000000
        finance._fixed_opex = 10000
        finance._cpi = 0.025
        # check outputs
        result = finance.get_expenses()
        nptesting.assert_array_almost_equal(result, np.loadtxt(os.path.join(test_folder,
                                                                            "financial_calculator/expenses_results1.csv")), 2)

    def test_get_producer_expenses(self):
        # set inputs
        finance = FinancialCalculator(self.output)
        finance._total_producer_capex = 4200000
        finance._total_producer_opex = 42000
        finance._cpi = 0.025
        # check outputs
        result = finance.get_producer_expenses()
        nptesting.assert_array_almost_equal(result, np.loadtxt(os.path.join(test_folder,
                                                                            "financial_calculator/expenses_results3.csv")), 2)

    def test_get_bess_expenses(self):
        # set inputs
        finance = FinancialCalculator(self.output)
        finance._land_capex = 10000
        finance._land_opex = 100
        finance._total_producer_capex = 4200000
        finance._total_producer_opex = 42000
        battery_data = np.loadtxt(os.path.join(test_folder, "financial_calculator/creation_battery_data1.csv"),
                                  delimiter=",")
        finance._battery_cost = battery_data[1]
        finance._battery_opex = battery_data[2]
        finance._battery_connection_capex = 2192750
        finance._battery_connection_opex = 21927.5
        finance._fixed_capex = 10000000
        finance._fixed_opex = 10000
        finance._cpi = 0.025
        finance._battery_cost_deg = 0.05
        # check outputs
        result = finance.get_bess_expenses()
        nptesting.assert_array_almost_equal(result, np.loadtxt(os.path.join(test_folder,
                                                                            "financial_calculator/expenses_results2.csv")), 2)

    def test_get_cash_flow(self):
        # set inputs
        finance = FinancialCalculator(self.output)
        cash_flow_data = np.loadtxt(os.path.join(test_folder, "financial_calculator/cash_flow_data1.csv"),
                                    delimiter=",")
        income = cash_flow_data[:, 0]
        costs = cash_flow_data[:, 1]
        finance.get_power_sales = Mock(return_value=income)
        finance.get_expenses = Mock(return_value=costs)
        # check outputs
        result = finance.get_cash_flow()
        expected_outputs = np.loadtxt(os.path.join(test_folder, "financial_calculator/cash_flow_data1_result.csv"),
                                      delimiter=",")
        nptesting.assert_array_almost_equal(result[0], expected_outputs[:, 0], 2)
        nptesting.assert_array_almost_equal(result[1], expected_outputs[:, 1], 2)
        nptesting.assert_array_almost_equal(result[2], expected_outputs[:, 2], 2)
        finance.get_power_sales.assert_called_once()
        finance.get_expenses.assert_called_once()

    def test_get_irr(self):
        # set inputs
        finance = FinancialCalculator(self.output)
        cash_flow_data = np.loadtxt(os.path.join(test_folder, "financial_calculator/cash_flow_data1_result.csv"),
                                    delimiter=",")
        finance.get_cash_flow = Mock(return_value=(cash_flow_data[:, 0], cash_flow_data[:, 1], cash_flow_data[:, 2]))
        # check outputs
        result = finance.get_irr()
        self.assertAlmostEqual(result, 0.0996, 4)
        finance.get_cash_flow.assert_called_once()

    def test_get_npv(self):
        # set inputs
        finance = FinancialCalculator(self.output)
        cash_flow_data = np.loadtxt(os.path.join(test_folder, "financial_calculator/cash_flow_data1_result.csv"),
                                    delimiter=",")
        finance.get_cash_flow = Mock(return_value=(cash_flow_data[:, 0], cash_flow_data[:, 1], cash_flow_data[:, 2]))
        # check outputs
        result = finance.get_npv()
        self.assertAlmostEqual(result, -56925.75, 2)
        finance.get_cash_flow.assert_called_once()

    def test_get_lcoe(self):
        # set inputs
        power_output = np.loadtxt(os.path.join(test_folder, "financial_calculator/power_output_data1.csv"))
        power_output = [np.array([x / 2, x / 2]) for x in power_output]
        purchased_from_grid = 0
        finance = FinancialCalculator(self.output)
        finance.get_power_purchases_cost = Mock(return_value=
                                                np.loadtxt(os.path.join(test_folder,
                                                                        "financial_calculator/power_purchased_data2.csv")))
        finance.get_producer_expenses = Mock(return_value=np.loadtxt(os.path.join(test_folder,
                                                                                  "financial_calculator/expenses_results1.csv")))
        # check outputs
        result = finance.get_lcoe(power_output, purchased_from_grid)
        self.assertAlmostEqual(result, 0.053114, 4)
        self.output.run.assert_not_called()
        finance.get_producer_expenses.assert_called_once()
        finance.get_power_purchases_cost.assert_called_once()

    def test_get_lcoe_calculator_data(self):
        # set inputs
        power_output = np.loadtxt(os.path.join(test_folder, "financial_calculator/power_output_data1.csv"))
        calc_results = [pd.DataFrame({"grid_from_pv": [x / 2, x / 2], "bess_from_grid": [0, 0]}) for x in power_output]
        type(self.output).results = calc_results
        finance = FinancialCalculator(self.output)
        finance.get_power_purchases_cost = Mock(return_value=
                                                np.loadtxt(os.path.join(test_folder,
                                                                        "financial_calculator/power_purchased_data2.csv")))
        finance.get_producer_expenses = Mock(return_value=np.loadtxt(os.path.join(test_folder,
                                                                                  "financial_calculator/expenses_results1.csv")))
        # check outputs
        result = finance.get_lcoe()
        self.assertAlmostEqual(result, 0.053114, 4)
        self.output.run.assert_not_called()
        finance.get_producer_expenses.assert_called_once()
        finance.get_power_purchases_cost.assert_called_once()

    def test_get_lcoe_no_output_data(self):
        # set inputs
        type(self.output).results = None

        def run_mock():
            power_output = np.loadtxt(os.path.join(test_folder, "financial_calculator/power_output_data1.csv"))
            calc_results = [pd.DataFrame({"grid_from_pv": [x / 2, x / 2], "bess_from_grid": [0, 0]}) for x in
                            power_output]
            type(self.output).results = calc_results
            purchase_data = np.loadtxt(os.path.join(test_folder, "financial_calculator/purchased_power_data1.csv"),
                                       delimiter=",")
            purchased_from_grid = [pd.Series(purchase_data[:, 0]), pd.Series(purchase_data[:, 1])]
            type(self.output).purchased_from_grid = purchased_from_grid

        self.output.run = Mock(side_effect=run_mock)
        finance = FinancialCalculator(self.output)
        finance.get_power_purchases_cost = Mock(return_value=
                                                np.loadtxt(os.path.join(test_folder,
                                                                        "financial_calculator/power_purchased_data2.csv")))
        finance.get_producer_expenses = Mock(return_value=np.loadtxt(os.path.join(test_folder,
                                                                                  "financial_calculator/expenses_results1.csv")))
        # check outputs
        result = finance.get_lcoe()
        self.assertAlmostEqual(result, 0.053114, 4)
        self.output.run.assert_called_once()
        finance.get_producer_expenses.assert_called_once()
        finance.get_power_purchases_cost.assert_called_once()

    def test_get_lcoe_no_purchase_data(self):
        # set inputs
        power_output = np.loadtxt(os.path.join(test_folder, "financial_calculator/power_output_data1.csv"))
        type(self.output).purchased_from_grid = None

        def run_mock():
            purchase_data = np.loadtxt(os.path.join(test_folder, "financial_calculator/purchased_power_data1.csv"),
                                       delimiter=",")
            purchased_from_grid = [pd.Series(purchase_data[:, 0]), pd.Series(purchase_data[:, 1])]
            type(self.output).purchased_from_grid = purchased_from_grid

        self.output.run = Mock(side_effect=run_mock)
        finance = FinancialCalculator(self.output)
        finance.get_power_purchases_cost = Mock(return_value=
                                                np.loadtxt(os.path.join(test_folder,
                                                                        "financial_calculator/power_purchased_data2.csv")))
        finance.get_producer_expenses = Mock(return_value=np.loadtxt(os.path.join(test_folder,
                                                                                  "financial_calculator/expenses_results1.csv")))
        # check outputs
        result = finance.get_lcoe(power_output)
        self.assertAlmostEqual(result, 0.053114, 4)
        self.output.run.assert_called_once()
        finance.get_producer_expenses.assert_called_once()
        finance.get_power_purchases_cost.assert_called_once()

    def test_get_lcoe_calculator_data_no_save_results(self):
        # set inputs
        type(self.output).save_all_results = False

        def run_mock():
            power_output = np.loadtxt(os.path.join(test_folder, "financial_calculator/power_output_data1.csv"))
            calc_results = [pd.DataFrame({"grid_from_pv": [x / 2, x / 2], "bess_from_grid": [0, 0]}) for x in
                            power_output]
            type(self.output).results = calc_results
            purchase_data = np.loadtxt(os.path.join(test_folder, "financial_calculator/purchased_power_data1.csv"),
                                       delimiter=",")
            purchased_from_grid = [pd.Series(purchase_data[:, 0]), pd.Series(purchase_data[:, 1])]
            type(self.output).purchased_from_grid = purchased_from_grid

        self.output.run = Mock(side_effect=run_mock)
        finance = FinancialCalculator(self.output)
        finance.get_power_purchases_cost = Mock(return_value=
                                                np.loadtxt(os.path.join(test_folder,
                                                                        "financial_calculator/power_purchased_data2.csv")))
        finance.get_producer_expenses = Mock(return_value=np.loadtxt(os.path.join(test_folder,
                                                                                  "financial_calculator/expenses_results1.csv")))
        # check outputs
        result = finance.get_lcoe()
        self.assertAlmostEqual(result, 0.053114, 4)
        self.output.run.assert_called_once()
        self.assertFalse(self.output.save_all_results)
        finance.get_producer_expenses.assert_called_once()
        finance.get_power_purchases_cost.assert_called_once()

    def test_get_lcoe_no_power_costs(self):
        # set inputs
        power_output = np.loadtxt(os.path.join(test_folder, "financial_calculator/power_output_data1.csv"))
        power_output = [np.array([x / 2, x / 2]) for x in power_output]
        finance = FinancialCalculator(self.output)
        finance.get_producer_expenses = Mock(return_value=np.loadtxt(os.path.join(test_folder,
                                                                                  "financial_calculator/expenses_results1.csv")))
        # check outputs
        result = finance.get_lcoe_no_power_costs(power_output)
        self.assertAlmostEqual(result, 0.049409, 4)
        self.output.run.assert_not_called()
        finance.get_producer_expenses.assert_called_once()

    def test_get_lcoe_no_power_costs_claculator_data(self):
        # set inputs
        power_output = np.loadtxt(os.path.join(test_folder, "financial_calculator/power_output_data1.csv"))
        calc_results = [pd.DataFrame({"grid_from_pv": [x / 2, x / 2], "bess_from_grid": [0, 0]}) for x in power_output]
        type(self.output).results = calc_results
        finance = FinancialCalculator(self.output)
        finance.get_producer_expenses = Mock(return_value=np.loadtxt(os.path.join(test_folder,
                                                                                  "financial_calculator/expenses_results1.csv")))
        # check outputs
        result = finance.get_lcoe_no_power_costs()
        self.assertAlmostEqual(result, 0.049409, 4)
        self.output.run.assert_not_called()
        finance.get_producer_expenses.assert_called_once()

    def test_get_lcoe_no_power_costs_no_output_data(self):
        # set inputs
        type(self.output).results = None

        def run_mock():
            power_output = np.loadtxt(os.path.join(test_folder, "financial_calculator/power_output_data1.csv"))
            calc_results = [pd.DataFrame({"grid_from_pv": [x / 2, x / 2], "bess_from_grid": [0, 0]}) for x in
                            power_output]
            type(self.output).results = calc_results

        self.output.run = Mock(side_effect=run_mock)
        finance = FinancialCalculator(self.output)
        finance.get_producer_expenses = Mock(return_value=np.loadtxt(os.path.join(test_folder,
                                                                                  "financial_calculator/expenses_results1.csv")))
        # check outputs
        result = finance.get_lcoe_no_power_costs()
        self.assertAlmostEqual(result, 0.049409, 4)
        self.output.run.assert_called_once()
        finance.get_producer_expenses.assert_called_once()

    def test_get_lcoe_no_power_costs_calculator_data_no_save_results(self):
        # set inputs
        type(self.output).save_all_results = False

        def run_mock():
            power_output = np.loadtxt(os.path.join(test_folder, "financial_calculator/power_output_data1.csv"))
            calc_results = [pd.DataFrame({"grid_from_pv": [x / 2, x / 2], "bess_from_grid": [0, 0]}) for x in
                            power_output]
            type(self.output).results = calc_results

        self.output.run = Mock(side_effect=run_mock)
        finance = FinancialCalculator(self.output)
        finance.get_producer_expenses = Mock(return_value=np.loadtxt(os.path.join(test_folder,
                                                                                  "financial_calculator/expenses_results1.csv")))
        # check outputs
        result = finance.get_lcoe_no_power_costs()
        self.assertAlmostEqual(result, 0.049409, 4)
        self.output.run.assert_called_once()
        self.assertFalse(self.output.save_all_results)
        finance.get_producer_expenses.assert_called_once()

    def test_get_lcos(self):
        # set inputs
        output_results = np.loadtxt(os.path.join(test_folder, "financial_calculator/grid_from_bess_data1.csv"),
                                    delimiter=",")
        type(self.output).results = [pd.DataFrame([x], columns=["grid_from_bess"]) for x in output_results]
        finance = FinancialCalculator(self.output)
        finance.get_bess_expenses = Mock(return_value=np.loadtxt(os.path.join(test_folder,
                                                                              "financial_calculator/expenses_results2.csv")))
        # check outputs
        result = finance.get_lcos()
        self.assertAlmostEqual(result, 0.014232998, 4)
        self.output.run.assert_not_called()
        finance.get_bess_expenses.assert_called_once()

    def test_get_lcos_no_results_data(self):
        # set inputs
        output_results = np.loadtxt(os.path.join(test_folder, "financial_calculator/grid_from_bess_data1.csv"),
                                    delimiter=",")
        type(self.output).results = None

        def run_mock():
            type(self.output).results = [pd.DataFrame([output_results[i]], columns=["grid_from_bess"]) for i in
                                         range(output_results.shape[0])]

        self.output.run = Mock(side_effect=run_mock)
        finance = FinancialCalculator(self.output)
        finance.get_power_purchases_cost = Mock(return_value=
                                                np.loadtxt(os.path.join(test_folder,
                                                                        "financial_calculator/power_purchased_data2.csv")))
        finance.get_bess_expenses = Mock(return_value=np.loadtxt(os.path.join(test_folder,
                                                                              "financial_calculator/expenses_results2.csv")))
        # check outputs
        result = finance.get_lcos()
        self.assertAlmostEqual(result, 0.014232998, 4)
        self.output.run.assert_called_once()
        finance.get_bess_expenses.assert_called_once()

    def test_get_lcos_no_save_results(self):
        # set inputs
        output_results = np.loadtxt(os.path.join(test_folder, "financial_calculator/grid_from_bess_data1.csv"),
                                    delimiter=",")
        type(self.output).save_all_results = False

        def run_mock():
            type(self.output).results = [pd.DataFrame([output_results[i]], columns=["grid_from_bess"]) for i in
                                         range(output_results.shape[0])]

        self.output.run = Mock(side_effect=run_mock)
        finance = FinancialCalculator(self.output)
        finance.get_power_purchases_cost = Mock(return_value=
                                                np.loadtxt(os.path.join(test_folder,
                                                                        "financial_calculator/power_purchased_data2.csv")))
        finance.get_bess_expenses = Mock(return_value=np.loadtxt(os.path.join(test_folder,
                                                                              "financial_calculator/expenses_results2.csv")))
        # check outputs
        result = finance.get_lcos()
        self.assertAlmostEqual(result, 0.014232998, 4)
        self.output.run.assert_called_once()
        self.assertFalse(self.output.save_all_results)
        finance.get_bess_expenses.assert_called_once()

