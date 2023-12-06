from typing import Any

import numpy_financial as npf
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from .output_calculator import OutputCalculator
from . import constants
from .producers import PvProducer
from .power_storage import LithiumPowerStorage


class FinancialCalculator:
    """
    calculates finance for a pv system with storage
    """

    def __init__(self,
                 output_calculator: OutputCalculator,
                 land_size: float = constants.DEFAULT_LAND_SIZE,
                 capex_per_land_unit: float = constants.DEFAULT_CAPEX_PER_LAND_UNIT,
                 opex_per_land_unit: float = constants.DEFAULT_OPEX_PER_LAND_UNIT,
                 capex_per_kwp: float = constants.DEFAULT_CAPEX_PER_KWP,
                 opex_per_kwp: float = constants.DEFAULT_OPEX_PER_KWP,
                 battery_capex_per_kwh: float = constants.DEFAULT_BATTERY_CAPEX_PER_KWH,
                 battery_opex_per_kwh: float = constants.DEFAULT_BATTERY_OPEX_PER_KWH,
                 battery_connection_capex_per_kw: float = constants.DEFAULT_BATTERY_CONNECTION_CAPEX_PER_KW,
                 battery_connection_opex_per_kw: float = constants.DEFAULT_BATTERY_CONNECTION_OPEX_PER_KW,
                 fixed_capex: float = constants.DEFAULT_FIXED_CAPEX,
                 fixed_opex: float = constants.DEFAULT_FIXED_OPEX,
                 usd_to_ils: float = 3.73,
                 interest_rate: float = constants.DEFAULT_INTEREST_RATE / 100,
                 cpi: float = constants.DEFAULT_CPI / 100,
                 battery_cost_deg: float = constants.DEFAULT_BATTERY_COST_DEG / 100,
                 tariff_table: np.ndarray[Any, np.dtype[np.float64]] | None = None,
                 base_tariff: float = constants.DEFAULT_BASE_TARIFF,
                 winter_low_factor: float = constants.DEFAULT_WINTER_LOW_FACTOR,
                 winter_high_factor: float = constants.DEFAULT_WINTER_HIGH_FACTOR,
                 transition_low_factor: float = constants.DEFAULT_TRANSITION_LOW_FACTOR,
                 transition_high_factor: float = constants.DEFAULT_TRANSITION_HIGH_FACTOR,
                 summer_low_factor: float = constants.DEFAULT_SUMMER_LOW_FACTOR,
                 summer_high_factor: float = constants.DEFAULT_SUMMER_HIGH_FACTOR,
                 buy_from_grid_factor: float = 1.0):
        """
        initialize the calculator with info on the system

        :param capex_per_land_unit: capital cost per unit of land (shekel)
        :param opex_per_land_unit: operational cost per unit of land (shekel)
        :param capex_per_kwp: capital cost per kW of peak power ($)
        :param opex_per_kwp: operational cost per kW of peak power ($)
        :param battery_capex_per_kwh: capital cost per kWh in the battery capacity ($)
        :param battery_opex_per_kwh: operational cost per kWh in the battery capacity ($)
        :param battery_connection_capex_per_kw: capital cost per kWh of the battery connection ($)
        :param battery_connection_opex_per_kw: operational cost per kWh of the battery connection ($)
        :param fixed_capex: variable capital cost (shekel)
        :param fixed_opex: variable operational cost (shekel)
        :param usd_to_ils: exchange rate of us dollar to shekel
        :param interest_rate: the market interest rate (as fraction)
        :param cpi: the market consumer price index (as fraction)
        :param battery_cost_deg yearly degradation in price of batteries (as fraction)
        :param tariff_table: a numpy array with tariff for every hour in every month (or none)
        :param base_tariff: basic tariff for power (multiplied by seasonal factor to get seasonal rate, shekel. if
            tariff table is supplied, this and the factor below are ignored)
        :param winter_low_factor: winter low factor
        :param winter_high_factor: winter high factor
        :param transition_low_factor: transition season low factor
        :param transition_high_factor: transition season high factor
        :param summer_low_factor: summer low factor
        :param summer_high_factor: summer high factor
        :param buy_from_grid_factor: the factor for the cost of buying power from grid compare to tariffs
        """
        # expenses variables
        self._set_capex_per_land_unit(capex_per_land_unit)
        self._set_opex_per_land_unit(opex_per_land_unit)
        self.land_size = land_size
        self.usd_to_ils = usd_to_ils
        self._set_capex_per_kwp(capex_per_kwp)
        self._set_opex_per_kwp(opex_per_kwp)
        self._set_battery_capex_per_kwh(battery_capex_per_kwh)
        self._set_battery_opex_per_kwh(battery_opex_per_kwh)
        self._set_battery_connection_capex_per_kw(battery_connection_capex_per_kw)
        self._set_battery_connection_opex_per_kw(battery_connection_opex_per_kw)
        self.fixed_capex = fixed_capex
        self.fixed_opex = fixed_opex
        self.interest_rate = interest_rate
        self.cpi = cpi
        self.battery_cost_deg = battery_cost_deg

        self.output_calculator = output_calculator

        if tariff_table is not None:
            self._set_tariff_table(tariff_table)
        else:
            # revenues variables
            self._winter_months = [0, 1, 11]
            self._transition_months = [2, 3, 4, 9, 10]
            self._summer_months = [5, 6, 7, 8]
            # day of the week by datetime notation (sunday is 0)
            self._week_days = [0, 1, 2, 3, 4]
            self._weekend_days = [5, 6]
            self._winter_low_hours = list(range(1, 17)) + [22, 23, 0]
            self._winter_high_hours = list(range(17, 22))
            self._transition_low_hours = self._winter_low_hours
            self._transition_high_hours = self._winter_high_hours
            self._summer_low_hours = list(range(1, 17)) + [23, 0]
            self._summer_high_hours = list(range(17, 23))
            self._winter_low = base_tariff * winter_low_factor
            self._winter_high_week = base_tariff * winter_high_factor
            self._transition_low = base_tariff * transition_low_factor
            self._transition_high_week = base_tariff * transition_high_factor
            self._summer_low = base_tariff * summer_low_factor
            self._summer_high_week = base_tariff * summer_high_factor
            self._winter_high_weekend = self._winter_high_week
            self._transition_high_weekend = self._transition_low
            self._summer_high_weekend = self._summer_low

            self._build_tariff_table()

        self.buy_from_grid_factor = buy_from_grid_factor
        self._reset_variables()

    def _reset_variables(self):
        self._income_details = None
        # save the last tariff matrix calculated
        self._hourly_tariff = None
        self._hourly_matrix_year = None

    # region Properties
    @property
    def output_calculator(self) -> OutputCalculator:
        """
        an OutputCalculator object used for simulations
        """
        return self._output_calculator

    @output_calculator.setter
    def output_calculator(self, value: OutputCalculator):
        if not isinstance(value, OutputCalculator):
            raise ValueError("Output calculator should be an instance of OutputCalculator")
        self._output_calculator = value
        self._num_of_years = value.num_of_years
        self._set_rated_power(value.rated_power)
        self._set_battery_size(value.aug_table)
        self._set_battery_connection_size(value.pcs_power)

    @property
    def num_of_years(self):
        """
        the number of years for which to simulate and calculate data
        """
        return self._num_of_years

    @property
    def land_size(self):
        """
        the size of the land used for the system
        """
        return self._land_size

    @land_size.setter
    def land_size(self, value: int):
        if value <= 0:
            raise ValueError("Land size should be positive")
        self._land_size = value
        self._land_capex = value * self._capex_per_land_unit
        self._land_opex = value * self._opex_per_land_unit
        self._reset_variables()

    @property
    def capex_per_land_unit(self):
        """
        capital expense for each land unit
        """
        return self._capex_per_land_unit

    def _set_capex_per_land_unit(self, value):
        if value < 0:
            raise ValueError("Capex per land unit should be non negative")
        self._capex_per_land_unit = value

    @property
    def opex_per_land_unit(self):
        """
        operational expense for each land unit
        """
        return self._opex_per_land_unit

    def _set_opex_per_land_unit(self, value):
        if value < 0:
            raise ValueError("Opex per land unit should be non negative")
        self._opex_per_land_unit = value

    @property
    def land_capex(self):
        """
        total capital expense for land
        """
        return self._land_capex

    @property
    def land_opex(self):
        """
        total operational expense for land
        """
        return self._land_opex

    @property
    def usd_to_ils(self):
        """
        conversion rate from us dollars to israeli new shekels
        """
        return self._usd_to_ils

    @usd_to_ils.setter
    def usd_to_ils(self, value):
        if value < 0:
            raise ValueError("Dollar to shekel exchange should be non negative")
        self._usd_to_ils = value
        # recalculate costs
        if hasattr(self, '_pv_peak_power'):
            self._set_rated_power(self._pv_peak_power)
            self._set_battery_size(self._output_calculator.aug_table)
            self._set_battery_connection_size(self._battery_connection_size)

    def _set_rated_power(self, value: int):
        self._pv_peak_power = value
        self._total_producer_capex = value * self._capex_per_kwp * self._usd_to_ils
        self._total_producer_opex = value * self._opex_per_kwp * self._usd_to_ils
        self._reset_variables()

    @property
    def capex_per_kwp(self):
        """
        capital expense for each kWp from the rated producer power
        """
        return self._capex_per_kwp

    def _set_capex_per_kwp(self, value):
        if value < 0:
            raise ValueError("Capex per kWp should be non negative")
        self._capex_per_kwp = value

    @property
    def opex_per_kwp(self):
        """
        operational expense for each kWp from the rated producer power
        """
        return self._opex_per_kwp

    def _set_opex_per_kwp(self, value):
        if value < 0:
            raise ValueError("Opex per kWp should be non negative")
        self._opex_per_kwp = value

    @property
    def total_producer_capex(self):
        """
        total capital expense for the producer
        """
        return self._total_producer_capex

    @property
    def total_producer_opex(self):
        """
        total operational expense for the producer
        """
        return self._total_producer_opex

    def _set_battery_size(self, value: tuple[tuple[int, int, float]]):
        """
        calculate the battery size for each year

        :param value: an augmentation table
        """
        self._battery_size = []
        self._battery_cost = []
        self._battery_opex = []
        current_aug = 0
        # calculate capex and opex for each year (before cpi)
        for year in range(self._num_of_years):
            # calculate opex for current battery size
            if self._battery_size:
                self._battery_opex.append(self._battery_size[-1] * self._battery_opex_per_kwh * self._usd_to_ils)
            else:
                self._battery_opex.append(0)
            # if the current augmentation is this year, add to size, capex and opex
            if current_aug < len(value) and value[current_aug][0] // 12 == year:
                if self._battery_size:
                    self._battery_size.append(self._battery_size[-1] + value[current_aug][2])
                else:
                    self._battery_size.append(value[current_aug][2])
                self._battery_cost.append(value[current_aug][2] * self._battery_capex_per_kwh * self._usd_to_ils)
                # add opex for the new blocks, based on the month of the augmentation
                self._battery_opex[-1] += value[current_aug][2] * ((12 - (value[current_aug][0] % 12)) / 12) * \
                    self._battery_opex_per_kwh * self._usd_to_ils
                current_aug += 1
            else:
                if self._battery_size:
                    self._battery_size.append(self._battery_size[-1])
                else:
                    self._battery_size.append(0)
                self._battery_cost.append(0)
        self._reset_variables()

    @property
    def battery_capex_per_kwh(self):
        """
        capital expense for each kWh in the bess
        """
        return self._battery_capex_per_kwh

    def _set_battery_capex_per_kwh(self, value):
        if value < 0:
            raise ValueError("Capex per battery kWh should be non negative")
        self._battery_capex_per_kwh = value

    @property
    def battery_opex_per_kwh(self):
        """
        operational expense for each kWh in the bess
        """
        return self._battery_opex_per_kwh

    def _set_battery_opex_per_kwh(self, value):
        if value < 0:
            raise ValueError("Opex per battery kWh should be non negative")
        self._battery_opex_per_kwh = value

    @property
    def battery_cost(self):
        """
        total cost of the bess in each year
        """
        return self._battery_cost

    @property
    def battery_opex(self):
        """
        total operational expense of the bess in each year
        """
        return self._battery_opex

    def _set_battery_connection_size(self, value):
        self._battery_connection_size = value
        self._battery_connection_capex = value * self._battery_connection_capex_per_kw * self._usd_to_ils
        self._battery_connection_opex = value * self._battery_connection_opex_per_kw * self._usd_to_ils
        self._reset_variables()

    @property
    def battery_connection_capex_per_kw(self):
        """
        capital expense for each kW of the bess connection
        """
        return self._battery_connection_capex_per_kw

    def _set_battery_connection_capex_per_kw(self, value):
        if value < 0:
            raise ValueError("Capex per kw of battery connection should be non negative")
        self._battery_connection_capex_per_kw = value

    @property
    def battery_connection_opex_per_kw(self):
        """
        operational expense for each kW of the bess connection
        """
        return self._battery_connection_opex_per_kw

    def _set_battery_connection_opex_per_kw(self, value):
        if value < 0:
            raise ValueError("Opex per kw of battery connection should be non negative")
        self._battery_connection_opex_per_kw = value

    @property
    def battery_connection_capex(self):
        """
        total capital expense for the bess connection
        """
        return self._battery_connection_capex

    @property
    def battery_connection_opex(self):
        """
        total operational expense for the bess connection
        """
        return self._battery_connection_opex

    @property
    def fixed_capex(self):
        """
        capital expense for miscellaneous not depending on land, producer or bess
        """
        return self._fixed_capex

    @fixed_capex.setter
    def fixed_capex(self, value: int):
        if value < 0:
            raise ValueError("Fixed capex should be non negative")
        self._fixed_capex = value
        self._reset_variables()

    @property
    def fixed_opex(self):
        """
        operational expense for miscellaneous not depending on land, producer or bess
        """
        return self._fixed_opex

    @fixed_opex.setter
    def fixed_opex(self, value: int):
        if value < 0:
            raise ValueError("Fixed opex should be non negative")
        self._fixed_opex = value
        self._reset_variables()

    @property
    def interest_rate(self):
        """
        the market interest rate
        """
        return self._interest_rate

    @interest_rate.setter
    def interest_rate(self, value):
        if value < 0:
            raise ValueError("Interest rate should be non negative")
        self._interest_rate = value
        self._reset_variables()

    @property
    def cpi(self):
        """
        the consumer price index
        """
        return self._cpi

    @cpi.setter
    def cpi(self, value):
        if value < 0:
            raise ValueError("CPI should be non negative")
        self._cpi = value
        self._reset_variables()

    @property
    def battery_cost_deg(self):
        """
        the annual reduction in bess cost
        """
        return self._battery_cost_deg

    @battery_cost_deg.setter
    def battery_cost_deg(self, value):
        if not 0 <= value < 1:
            raise ValueError("Battery cost degradation should be between 0 (inclusive) and 1 (exclusive)")
        self._battery_cost_deg = value
        self._reset_variables()

    @property
    def buy_from_grid_factor(self):
        """
        a factor by which the tariffs are multiplied to get the price of buying power from the grid at each hour
        """
        return self._buy_from_grid_factor

    @buy_from_grid_factor.setter
    def buy_from_grid_factor(self, value: float):
        if value < 0:
            raise ValueError("Factor for buying from grid should be non negative")
        self._buy_from_grid_factor = value
        self._reset_variables()

    @property
    def income_details(self):
        """
        a list of annual income for the system
        """
        return self._income_details

    @property
    def tariff_table(self):
        """
        a table with power tariffs for each hour of the day in each month
        """
        return self._tariff_table

    def _set_tariff_table(self, value: np.ndarray[Any, np.dtype[np.float64]]):
        if value is not None:
            if value.shape != (12, 24):
                raise ValueError("Tariff table should be of shape (12, 24)")
            else:
                # repeats the tariff table 7 times (for each day of the week)
                temp = np.zeros(7)
                self._tariff_table = value[None, ...] + temp[:, None, None]

    # endregion

    def _build_tariff_table(self):
        """
        create a tariff table containing the tariff in each hour for each month
        """
        self._tariff_table = np.zeros((7, 12, 24))
        # winter tariffs
        self._tariff_table[np.ix_(self._week_days, self._winter_months, self._winter_low_hours)] = self._winter_low
        self._tariff_table[np.ix_(self._weekend_days, self._winter_months, self._winter_low_hours)] = self._winter_low
        self._tariff_table[np.ix_(self._week_days, self._winter_months, self._winter_high_hours)] = \
            self._winter_high_week
        self._tariff_table[np.ix_(self._weekend_days, self._winter_months, self._winter_high_hours)] = \
            self._winter_high_weekend
        # transition tariffs
        self._tariff_table[np.ix_(self._week_days, self._transition_months, self._transition_low_hours)] = \
            self._transition_low
        self._tariff_table[np.ix_(self._weekend_days, self._transition_months, self._transition_low_hours)] = \
            self._transition_low
        self._tariff_table[np.ix_(self._week_days, self._transition_months, self._transition_high_hours)] = \
            self._transition_high_week
        self._tariff_table[np.ix_(self._weekend_days, self._transition_months, self._transition_high_hours)] = \
            self._transition_high_weekend
        # summer tariffs
        self._tariff_table[np.ix_(self._week_days, self._summer_months, self._summer_low_hours)] = self._summer_low
        self._tariff_table[np.ix_(self._weekend_days, self._summer_months, self._summer_low_hours)] = self._summer_low
        self._tariff_table[np.ix_(self._week_days, self._summer_months, self._summer_high_hours)] = \
            self._summer_high_week
        self._tariff_table[np.ix_(self._weekend_days, self._summer_months, self._summer_high_hours)] = \
            self._summer_high_weekend

    def get_hourly_tariff(self, year):
        """
        create a matrix of hourly tariff in each day of the given year

        :param year: the year to calculate for (in 4 digits)
        :return: a numpy array with the tariffs
        """
        if self._hourly_tariff is not None and self._hourly_matrix_year == year:
            return self._hourly_tariff
        times = pd.date_range(start=f'{year}-01-01 00:00', end=f'{year}-12-31 23:00', freq='h', tz='Asia/Jerusalem')
        def f(x): return self._tariff_table[(x.day_of_week + 1) % 7, x.month - 1, x.hour]
        self._hourly_tariff = f(times)
        self._hourly_matrix_year = year
        return self._hourly_tariff

    def get_power_sales(self, power_output=None, cpi: float | None = None, purchased_from_grid=None,
                        no_purchase: bool = False):
        """
        calculate the yearly income according to the given power_output

        :param power_output: list of hourly output of the system for each year(list of pandas series, with datetime
            indices). if None takes info from output_calculator
        :param cpi: the consumer price index in the market in decimal (if none uses the calculator's cpi)
        :param purchased_from_grid: list of hourly amount purchased from grid to fill battery (list of pandas series).
            if None takes info from output_calculator
        :param no_purchase: whether to not use purchases from grid in the calculation

        :return: list of income per year
        """
        # get default values from output
        if power_output is None:
            if self._output_calculator.output is None:
                self._output_calculator.run()
            power_output = self._output_calculator.output
        if purchased_from_grid is None and not no_purchase:
            if self._output_calculator.purchased_from_grid is None:
                self._output_calculator.run()
            purchased_from_grid = self._output_calculator.purchased_from_grid
        if cpi is None:
            cpi = self._cpi

        sales = []
        cpi_multi = 1
        self._income_details = []
        for year in range(self._num_of_years):
            hourly_tariff = self.get_hourly_tariff(power_output[year].index[0].year)
            temp = power_output[year] * hourly_tariff * cpi_multi
            temp = np.where(temp >= 0, temp, temp * self._buy_from_grid_factor)
            # add the payments for power from grid to battery
            if not no_purchase:
                temp -= purchased_from_grid[year] * hourly_tariff * cpi_multi * self._buy_from_grid_factor
            sales.append(temp.sum())
            # save the matrices for each year
            self._income_details.append(temp)
            cpi_multi *= 1 + cpi
        return sales

    def get_power_purchases_cost(self, purchased_from_grid=None, cpi: float | None = None):
        """
        calculate the cost of purchases from grid

        :param purchased_from_grid: list of hourly amount purchased from grid to fill battery (list of pandas series,
            with datetime indices). if None takes info from output_calculator
        :param cpi: the consumer price index in the market in decimal (if none uses the calculator's cpi)
        """
        if purchased_from_grid is None:
            if self._output_calculator.purchased_from_grid is None:
                self._output_calculator.run()
            purchased_from_grid = self._output_calculator.purchased_from_grid
        if cpi is None:
            cpi = self._cpi

        purchases = []
        cpi_multi = 1
        for year in range(self._num_of_years):
            hourly_matrix = self.get_hourly_tariff(purchased_from_grid[year].index[0].year)
            temp = purchased_from_grid[year] * hourly_matrix * cpi_multi * self._buy_from_grid_factor
            purchases.append(temp.sum())
            cpi_multi *= 1 + cpi
        return purchases

    def get_expenses(self, cpi: float | None = None):
        """
        calculate yearly expenses (capex+opex)

        :param cpi: the yearly cpi in the market in decimal (if none uses the calculator's cpi)

        :return: list of expenses per year
        """
        if cpi is None:
            cpi = self._cpi

        expenses = []
        cpi_multi = 1
        battery_cost_rate = 1
        for year in range(self._num_of_years):
            yearly_expenses = 0
            # calculate first year capex
            if year == 0:
                yearly_expenses = self._land_capex + self._total_producer_capex + self._battery_connection_capex + \
                                  self._fixed_capex
            # calculate battery cost
            yearly_expenses += self._battery_cost[year] * battery_cost_rate
            # calculate opex
            yearly_expenses += self._land_opex + self._total_producer_opex + self._battery_opex[year] + \
                self._battery_connection_opex + self._fixed_opex

            # multiple expenses by cpi multiplier
            expenses.append(yearly_expenses * cpi_multi)

            cpi_multi *= 1 + cpi
            battery_cost_rate *= 1 - self._battery_cost_deg
        return expenses

    def get_producer_expenses(self, cpi: float | None = None):
        """
        calculate yearly expenses of power production (capex+opex)

        :param cpi: the yearly cpi in the market in decimal (if none uses the calculator's cpi)

        :return: list of expenses per year
        """
        if cpi is None:
            cpi = self._cpi

        expenses = []
        cpi_multi = 1
        for year in range(self._num_of_years):
            yearly_expenses = 0
            # calculate first year capex
            if year == 0:
                yearly_expenses = self._total_producer_capex + self._land_capex + self._fixed_capex
            # calculate opex
            yearly_expenses += self._total_producer_opex + self._land_opex + self._fixed_opex

            # multiple expenses by cpi multiplier
            expenses.append(yearly_expenses * cpi_multi)

            cpi_multi *= 1 + cpi
        return expenses

    def get_bess_expenses(self, cpi: float | None = None):
        """
        calculate yearly expenses on bess (capex+opex)

        :param cpi: the yearly cpi in the market in decimal (if none uses the calculator's cpi)

        :return: list of expenses per year
        """
        if cpi is None:
            cpi = self._cpi

        expenses = []
        cpi_multi = 1
        battery_cost_rate = 1
        for year in range(self._num_of_years):
            yearly_expenses = 0
            # calculate first year capex
            if year == 0:
                yearly_expenses = self._battery_connection_capex
            # calculate battery cost
            yearly_expenses += self._battery_cost[year] * battery_cost_rate
            # calculate opex
            yearly_expenses += self._battery_opex[year] + self._battery_connection_opex

            # multiple expenses by cpi multiplier
            expenses.append(yearly_expenses * cpi_multi)

            cpi_multi *= 1 + cpi
            battery_cost_rate *= 1 - self._battery_cost_deg
        return expenses

    def get_cash_flow(self, power_output=None, purchased_from_grid=None):
        """
        calculate the cash flow of the system

        :param power_output: list of hourly output of the system for each year(list of pandas series, with datetime
            indices). if None takes info from output_calculator
        :param purchased_from_grid: list of hourly amount purchased from grid to fill battery (list of pandas series).
            if None takes info from output_calculator

        :returns income, expenses and revenues of the system
        """
        income = self.get_power_sales(power_output=power_output, purchased_from_grid=purchased_from_grid)
        costs = self.get_expenses()
        revenues = [x - y for x, y in zip(income, costs)]
        return income, costs, revenues

    def get_irr(self, power_output=None, purchased_from_grid=None):
        """
        calculates the irr for the system

        :param power_output: list of hourly output of the system for each year(list of pandas series). if None takes
            info from output_calculator
        :param purchased_from_grid: list of hourly amount purchased from grid to fill battery (list of pandas series).
            if None takes info from output_calculator
        """
        _, _, revenues = self.get_cash_flow(power_output, purchased_from_grid)
        irr = npf.irr(revenues)
        return irr if not np.isnan(irr) else 0

    def get_npv(self, rate: float = 10, power_output=None, purchased_from_grid=None):
        """
        calculates the npv for the system

        :param rate: rate for the npv (in percent)
        :param power_output: list of hourly output of the system for each year(list of pandas series). if None takes
            info from output_calculator
        :param purchased_from_grid: list of hourly amount purchased from grid to fill battery (list of pandas series).
            if None takes info from output_calculator
        """
        _, _, revenues = self.get_cash_flow(power_output, purchased_from_grid)
        return npf.npv(rate / 100, revenues)

    def get_lcoe(self, power_output=None, purchased_from_grid=None):
        """
        calculate the lcoe for the system

        :param power_output: list of hourly output of the system for each year(list of pandas series). if None takes
            info from output_calculator
        :param purchased_from_grid: list of hourly amount purchased from grid to fill battery (list of pandas series,
            with datetime indices).
            if None takes info from output_calculator

        :return: lcoe for the system
        """
        # get default values from output
        if power_output is None:
            if not self._output_calculator.save_all_results:
                self._output_calculator.save_all_results = True
                self._output_calculator.run()
                self._output_calculator.save_all_results = False
            if self._output_calculator.results is None:
                self._output_calculator.run()
            power_output = [self._output_calculator.results[i]["grid_from_pv"] +
                            self._output_calculator.results[i]["bess_from_grid"] for i in range(self._num_of_years)]
        if purchased_from_grid is None:
            if self._output_calculator.purchased_from_grid is None:
                self._output_calculator.run()
            purchased_from_grid = self._output_calculator.purchased_from_grid

        # get capex and opex
        costs = self.get_producer_expenses()
        # calculate the costs of the power purchased from grid
        power_costs = self.get_power_purchases_cost(purchased_from_grid)

        lcoe = sum([(costs[i] + power_costs[i]) / (1 + self._interest_rate) ** i for i in range(self._num_of_years)]) /\
            sum([power_output[i].sum() / (1 + self._interest_rate) ** i for i in range(self._num_of_years)])
        return lcoe

    def get_lcoe_no_power_costs(self, power_output=None):
        """
        calculated the lcoe for the system without the costs of power purchased from the grid

        :param power_output: list of hourly output of the system for each year(list of pandas series, with datetime
            indices). if None takes info from output_calculator
        :return: lcoe for the system without purchases from grid
        """
        # get default values from output
        if power_output is None:
            if not self._output_calculator.save_all_results:
                self._output_calculator.save_all_results = True
                self._output_calculator.run()
                self._output_calculator.save_all_results = False
            if self._output_calculator.results is None:
                self._output_calculator.run()
            power_output = [self._output_calculator.results[i]["grid_from_pv"] for i in range(self._num_of_years)]

        # get capex and opex
        costs = self.get_producer_expenses()

        lcoe = sum([costs[i] / (1 + self._interest_rate) ** i for i in range(self._num_of_years)]) / \
            sum([power_output[i].sum() / (1 + self._interest_rate) ** i for i in range(self._num_of_years)])
        return lcoe

    def get_lcos(self):
        """
        calculate the lcos for the system
        """
        if not self._output_calculator.save_all_results:
            self._output_calculator.save_all_results = True
            self._output_calculator.run()
            self._output_calculator.save_all_results = False
        if self._output_calculator.results is None:
            self._output_calculator.run()
        results = self._output_calculator.results
        costs = self.get_bess_expenses()
        lcos = sum([costs[i] / (1 + self._interest_rate) ** i for i in range(self._num_of_years)]) / \
            sum([results[i]["grid_from_bess"].sum() /
                (1 + self._interest_rate) ** i for i in range(self._num_of_years)])
        return lcos

    def plot_cash_flow(self, power_output=None, purchased_from_grid=None):
        """
        plot the cash flow of the system

        :param power_output: list of hourly output of the system for each year(list of pandas series). if None takes
            info from output_calculator
        :param purchased_from_grid: list of hourly amount purchased from grid to fill battery (list of pandas series).
            if None takes info from output_calculator
        """
        _, _, revenues = self.get_cash_flow(power_output, purchased_from_grid)
        plt.plot(revenues)
        plt.show()
