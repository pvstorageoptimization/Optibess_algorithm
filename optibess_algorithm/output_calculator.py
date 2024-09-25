import datetime
import math
import os
from collections.abc import Iterable
from enum import Enum, auto
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import onnxruntime

from .producers import Producer
from .power_storage import PowerStorage
from .utils import (year_diff, month_diff, relu, clamp, is_leap_year, hour_num_in_range, get_yearly_prices,
                    is_real_numbers)
from .constants import YEAR_DAYS, YEAR_HOURS, DAY_LENGTH


# TODO: find a way to deal with case where discharge hour is 2 (this is problematic since daylight saving cause 1 day to
#  has 2 entry with the hour 2:00 and 1 day that has no hour 2:00)


class Coupling(Enum):
    """
    an enumeration of the options for the system layout
    """
    AC = auto()
    DC = auto()


class OutputCalculator:
    """
    Calculates the hourly output of the pv system and the storage system, and the hourly consumption from the grid
    """

    def __init__(self, num_of_years: int,
                 grid_size: int,
                 producer: Producer,
                 power_storage: PowerStorage,
                 coupling: Coupling = Coupling.AC,
                 mvpv_loss: float = 0.01,
                 trans_loss: float = 0.01,
                 mvbat_loss: float = 0.01,
                 pcs_loss: float = 0.015,
                 dc_dc_loss: float = 0.015,
                 bess_discharge_start_hour: int = 17,
                 fill_battery_from_grid: bool = True,
                 save_all_results: bool = True,
                 producer_factor: float = 1):
        """
        initialize the calculator with info on the system

        :param num_of_years: number of year to calculate the output for
        :param grid_size: the size of the grid connection (kwh)
        :param producer: the producer for the system
        :param power_storage: the battery for the system
        :param coupling: the coupling (AC/DC) of the system
        :param mvpv_loss: factor for loss for mv-pv (for ac coupling)
        :param trans_loss: factor for loss when transmitting power to/from grid
        :param mvbat_loss: factor for loss when mvbat (for ac coupling)
        :param pcs_loss: factor for loss for pcs unit
        :param dc_dc_loss: factor for loss between producer and bess (for dc coupling)
        :param bess_discharge_start_hour: the first hour at which to release power from the battery
        :param fill_battery_from_grid: whether to buy power from grid to fill battery if pv is not enough
        :param save_all_results: whether to save the all the data for every year
        :param producer_factor: a factor by which to reduce the producer output
        """

        # pv output variables
        self.grid_size = grid_size
        self.producer = producer
        self._initial_data = producer.power_output
        # get the first date in the data
        self._initial_date = self._initial_data.index[0]

        # times variables
        self._set_num_of_years(num_of_years)

        # storage variables
        self.power_storage = power_storage

        # loss variables
        self._coupling = coupling
        self.mvpv_loss = mvpv_loss
        self.trans_loss = trans_loss
        self.mvbat_loss = mvbat_loss
        self.pcs_loss = pcs_loss
        self.dc_dc_loss = dc_dc_loss
        self._set_prod_trans_loss(mvpv_loss, trans_loss, pcs_loss)
        self._set_charge_loss(mvpv_loss, mvbat_loss, pcs_loss, dc_dc_loss)
        self._set_grid_bess_loss(pcs_loss, mvbat_loss, trans_loss, dc_dc_loss)
        self._set_pcs_power()
        self.bess_discharge_start_hour = bess_discharge_start_hour
        self.fill_battery_from_grid = fill_battery_from_grid
        self._save_all_results = save_all_results
        self.producer_factor = producer_factor

        # augmentations variables
        self._next_aug = 0
        self._first_aug_entries = []

        # results lists
        self._results = None
        self._output = None
        self._purchased_from_grid = None

    # region Properties
    @property
    def num_of_years(self):
        """
        the number of year for which the system will be simulated
        """
        return self._num_of_years

    def _set_num_of_years(self, value: int):
        if value <= 0:
            raise ValueError("Number of years should be positive")
        self._num_of_years = value
        self._project_hour_num, self._yearly_hour_num = hour_num_in_range(self._producer.start_year, self._num_of_years)

    @property
    def grid_size(self):
        """
        the size of the connection to the grid (kW)
        """
        return self._grid_size

    @grid_size.setter
    def grid_size(self, value: int):
        if value <= 0:
            raise ValueError("Grid size should be positive")
        self._grid_size = value
        # reset pcs power if it was already initialized
        try:
            self._pcs_power
        except AttributeError:
            pass
        else:
            self._set_pcs_power()

    @property
    def producer(self):
        """
        the power producer of the system
        """
        return self._producer

    @producer.setter
    def producer(self, value: Producer):
        if not isinstance(value, Producer):
            raise ValueError("Producer should be an instance of class Producer")
        self._producer = value

    @property
    def power_storage(self):
        """
        the power storage of the system
        """
        return self._power_storage

    @power_storage.setter
    def power_storage(self, value: PowerStorage):
        if not isinstance(value, PowerStorage):
            raise ValueError("Power storage should be an instance of class PowerStorage")
        if value.num_of_years < self._num_of_years:
            raise ValueError("Power Storage number of years should be at least as many as the system")
        self._power_storage = value
        # reset pcs power if it was already initialized
        try:
            self._pcs_power
        except AttributeError:
            pass
        else:
            self._set_pcs_power()

    @property
    def rated_power(self):
        """
        the rated power of the system
        """
        return self._producer.rated_power * self._producer_factor

    @property
    def coupling(self):
        """
        The layout of the system depending on where the storage system is connected
        """
        return self._coupling

    @coupling.setter
    def coupling(self, value: Coupling):
        if value == self._coupling:
            return
        self._coupling = value
        # set losses again
        self._set_prod_trans_loss(self._mvpv_loss, self._trans_loss, self._pcs_loss)
        self._set_charge_loss(self._mvpv_loss, self._mvbat_loss, self._pcs_loss, self._dc_dc_loss)
        self._set_grid_bess_loss(self._pcs_loss, self._mvbat_loss, self._trans_loss, self._dc_dc_loss)

    @property
    def mvpv_loss(self):
        """
        the loss due to mvpv component
        """
        return self._mvpv_loss

    @mvpv_loss.setter
    def mvpv_loss(self, value: float):
        if not 0 < value < 1:
            raise ValueError("MV-PV loss should be between 0 and 1 (exclusive)")
        self._mvpv_loss = value
        # reset charge loss if it was already initialized
        try:
            self._charge_loss
        except AttributeError:
            pass
        else:
            self._set_charge_loss(value, self._mvbat_loss, self._pcs_loss, self._dc_dc_loss)
        # reset production trans loss if it was already initialized
        try:
            self._prod_trans_loss
        except AttributeError:
            pass
        else:
            self._set_prod_trans_loss(value, self._trans_loss, self._pcs_loss)

    @property
    def trans_loss(self):
        """
        the loss due to transformer component
        """
        return self._trans_loss

    @trans_loss.setter
    def trans_loss(self, value: float):
        if not 0 < value < 1:
            raise ValueError("Trans loss should be between 0 and 1 (exclusive)")
        self._trans_loss = value
        # reset grid bess loss if it was already initialized
        try:
            self._grid_bess_loss
        except AttributeError:
            pass
        else:
            self._set_grid_bess_loss(self._pcs_loss, self._mvbat_loss, value, self._dc_dc_loss)
        # reset production trans loss if it was already initialized
        try:
            self._prod_trans_loss
        except AttributeError:
            pass
        else:
            self._set_prod_trans_loss(self._mvpv_loss, value, self._pcs_loss)

    @property
    def mvbat_loss(self):
        """
        the loss due to mvbat component
        """
        return self._mvbat_loss

    @mvbat_loss.setter
    def mvbat_loss(self, value: float):
        if not 0 < value < 1:
            raise ValueError("MV-BAT loss should be between 0 and 1 (exclusive)")
        self._mvbat_loss = value
        # reset grid bess loss if it was already initialized
        try:
            self._grid_bess_loss
        except AttributeError:
            pass
        else:
            self._set_grid_bess_loss(self._pcs_loss, value, self._trans_loss, self._dc_dc_loss)
        # reset charge loss if it was already initialized
        try:
            self._charge_loss
        except AttributeError:
            pass
        else:
            self._set_charge_loss(self.mvpv_loss, value, self._pcs_loss, self._dc_dc_loss)

    @property
    def pcs_loss(self):
        """
        the loss due to pcs component
        """
        return self._pcs_loss

    @pcs_loss.setter
    def pcs_loss(self, value: float):
        if not 0 < value < 1:
            raise ValueError("PCS loss should be between 0 and 1 (exclusive)")
        self._pcs_loss = value
        # reset grid bess loss if it was already initialized
        try:
            self._grid_bess_loss
        except AttributeError:
            pass
        else:
            self._set_grid_bess_loss(value, self._mvbat_loss, self._trans_loss, self._dc_dc_loss)
        # reset charge loss if it was already initialized
        try:
            self._charge_loss
        except AttributeError:
            pass
        else:
            self._set_charge_loss(self.mvpv_loss, self._mvbat_loss, value, self._dc_dc_loss)
        # reset production trans loss if it was already initialized
        try:
            self._prod_trans_loss
        except AttributeError:
            pass
        else:
            self._set_prod_trans_loss(self._mvpv_loss, self._trans_loss, value)

    @property
    def dc_dc_loss(self):
        """
        the loss due to dc-dc component
        """
        return self._dc_dc_loss

    @dc_dc_loss.setter
    def dc_dc_loss(self, value: float):
        if not 0 < value < 1:
            raise ValueError("DC-DC loss should be between 0 and 1 (exclusive)")
        self._dc_dc_loss = value
        # reset grid bess loss if it was already initialized
        try:
            self._grid_bess_loss
        except AttributeError:
            pass
        else:
            self._set_grid_bess_loss(self._pcs_loss, self._mvbat_loss, self._trans_loss, value)
        # reset charge loss if it was already initialized
        try:
            self._charge_loss
        except AttributeError:
            pass
        else:
            self._set_charge_loss(self.mvpv_loss, self._mvbat_loss, self._pcs_loss, value)

    @property
    def prod_trans_loss(self):
        """
        the total loss where transmitting power from producer to grid
        """
        return self._prod_trans_loss

    def _set_prod_trans_loss(self, mvpv_loss, trans_loss, pcs_loss):
        if self._coupling == Coupling.AC:
            self._prod_trans_loss = 1 - ((1 - mvpv_loss) * (1 - trans_loss))
        else:
            self._prod_trans_loss = 1 - ((1 - pcs_loss) * (1 - trans_loss))

    @property
    def charge_loss(self):
        """
        the loss where charging storage from producer
        """
        return self._charge_loss

    def _set_charge_loss(self, mvpv_loss, mvbat_loss, pcs_loss, dc_dc_loss):
        if self._coupling == Coupling.AC:
            self._charge_loss = 1 - ((1 - mvpv_loss) * (1 - mvbat_loss) * (1 - pcs_loss))
        else:
            self._charge_loss = dc_dc_loss
        # reset pcs power if it was already initialized
        try:
            self._pcs_power
        except AttributeError:
            pass
        else:
            self._set_pcs_power()

    @property
    def grid_bess_loss(self):
        """
        the loss when transmitting power from grid to bess (or the other way around)
        """
        return self._grid_bess_loss

    def _set_grid_bess_loss(self, pcs_loss, mvbat_loss, trans_loss, dc_dc_loss):
        if self._coupling == Coupling.AC:
            self._grid_bess_loss = 1 - ((1 - pcs_loss) * (1 - mvbat_loss) * (1 - trans_loss))
        else:
            self._grid_bess_loss = 1 - ((1 - dc_dc_loss) * (1 - pcs_loss) * (1 - trans_loss))
        # reset pcs power if it was already initialized
        try:
            self._pcs_power
        except AttributeError:
            pass
        else:
            self._set_pcs_power()

    @property
    def pcs_power(self):
        """
        the size of the power control system responsible for regulating charge/discharge to the storage system
        """
        return self._pcs_power

    def _set_pcs_power(self):
        total_battery_nameplate = sum(self._power_storage.aug_table[:, 2])
        self._pcs_power = self._grid_size / (self._power_storage.rte_table[0] * (1 - self._grid_bess_loss)) + \
            total_battery_nameplate * self._power_storage.active_self_consumption
        # if total battery is small compare to grid size limit pcs by it size instead by grid size
        self._pcs_power = min(sum(self._power_storage.aug_table[:, 2]), self._pcs_power)

    @property
    def aug_table(self):
        """
        the table of augmentation for the storage system
        """
        return self._power_storage.aug_table

    @aug_table.setter
    def aug_table(self, value: tuple[tuple[int, int], ...]):
        self.set_aug_table(value)

    def set_aug_table(self, value: tuple[tuple[int, int], ...], month_diff_constraint: bool = True):
        self._power_storage.set_aug_table(value, month_diff_constraint)
        # reset the pcs power after aug table was changed
        self.power_storage = self._power_storage

    @property
    def bess_discharge_start_hour(self):
        """
        the hour at which the storage system starts to discharge
        """
        return self._bess_discharge_start_hour

    @bess_discharge_start_hour.setter
    def bess_discharge_start_hour(self, value: int):
        if not 0 <= value <= 23:
            raise ValueError("Battery discharge start hour should be between 0 and 23 (inclusive)")
        self._bess_discharge_start_hour = value

    @property
    def fill_battery_from_grid(self):
        """
        a boolean indicating if the system fills the storage system by buying power from the grid (if producer power
        wasn't sufficient)
        """
        return self._fill_battery_from_grid

    @fill_battery_from_grid.setter
    def fill_battery_from_grid(self, value: bool):
        self._fill_battery_from_grid = value

    @property
    def results(self):
        """
        a list of pandas dataframe with the hourly results of the simulation (see run method for details)
        """
        return self._results

    @property
    def output(self):
        """
        a list of pandas series with the hourly total output of the system
        """
        return self._output

    @property
    def purchased_from_grid(self):
        """
        a list of pandas series with the hourly power purchased from the grid
        """
        return self._purchased_from_grid

    @property
    def producer_factor(self):
        """
        a number between 0 and 1 which the producer output is multiplied by
        """
        return self._producer_factor

    @producer_factor.setter
    def producer_factor(self, value):
        if not 0 < value <= 2:
            raise ValueError("Producer factor should be between 0 (Exclusive) and 2 (inclusive)")
        self._producer_factor = value

    @property
    def save_all_results(self):
        """
        a boolean indicating if all of the results for each year should be saved (and also if optional results should
        be saved)
        """
        return self._save_all_results

    @save_all_results.setter
    def save_all_results(self, value: bool):
        self._save_all_results = value

    @property
    def project_hour_num(self):
        """
        return the number of hours in the project
        """
        return self._project_hour_num

    @property
    def yearly_hour_num(self):
        """
        return a list for number of hours in each year
        """
        return self._yearly_hour_num.copy()

    # endregion

    def _get_data(self, year: int):
        """
        create the basic dataframe for the year (indexed by the date, with values of the pv system output)

        :param year: the number of year in the simulation (first year is 0)
        """
        # in the first year Copy the power values for the initial source
        if year == 0:
            self._df = self._initial_data.copy(deep=True) * self._producer_factor
        else:
            # calculate PV for the year
            was_leap_year = is_leap_year(self._df.index[0].year)
            if was_leap_year:
                self._df = self._df.head(-DAY_LENGTH)
            # if last year was leap years add 366 days and not 365
            self._df.index += datetime.timedelta(days=(YEAR_DAYS + 1 if was_leap_year else YEAR_DAYS))
            # TODO: make degradation linear over the days
            self._df["pv_output"] = self._df["pv_output"] * (1 - self._producer.annual_deg)
        # checks if this is a leap year and add a day if so
        self._is_leap_year = is_leap_year(self._df.index[0].year)
        if self._is_leap_year:
            temp = self._df.tail(DAY_LENGTH)
            temp.index = self._df.index[-DAY_LENGTH:] + datetime.timedelta(days=1)
            self._df = pd.concat([self._df, temp])

        if self._save_all_results:
            # reset losses and soc
            self._df["acc_losses"] = 0
            self._df["soc"] = 0

        # create an index with 1 entry for each day for later calculations
        self._daily_index = pd.date_range(self._df.index[self._bess_discharge_start_hour], self._df.index[-1], freq='d')

    def _get_day_deg(self, date: pd.Timestamp, aug_initial_date: pd.Timestamp):
        """
        calculate the degradation of the battery for the given date, given the date of the start of the augmentation
        (using a value between the degradation values in the table, according to the number of months adn days passed)

        :param date: the date of the day
        :param aug_initial_date: the date of the start of the augmentation
        """
        aug_months_passed = month_diff(date, aug_initial_date)
        current_aug_year = year_diff(date, aug_initial_date)
        start_deg = self._power_storage.degradation_table[current_aug_year]
        end_deg = self._power_storage.degradation_table[current_aug_year + 1]
        # augmentation always happened in the start of the month, so we always reduce 1 from the day
        return start_deg - ((aug_months_passed % 12) / 12 + (date.day - 1) / 364) * (start_deg - end_deg)

    def _calc_augmentations(self):
        """
        calculate the current battery capacity for each augmentation (with respect to battery degradation) and the total
        battery capacity.
        """
        # calculate months passed since start
        months_passed = month_diff(self._df.index, self._initial_date)
        # check if the first entry of the next augmentation is this year
        if self._next_aug < self._power_storage.aug_table.shape[0]:
            # calculate total nameplate of battery
            self._df["battery_nameplate"] = np.where(months_passed >= int(self._power_storage.aug_table[
                                                                              self._next_aug][0]),
                                                     sum(self._power_storage.aug_table[:self._next_aug + 1, 2]),
                                                     sum(self._power_storage.aug_table[:self._next_aug, 2]))
            temp = np.where(months_passed == int(self._power_storage.aug_table[self._next_aug][0]))[0]
            if temp.size > 0:
                temp = temp[0]
            else:
                temp = -1
        else:
            # calculate total nameplate of battery
            self._df["battery_nameplate"] = sum(self._power_storage.aug_table[:, 2])
            temp = -1
        if temp != -1:
            # first_aug_entries is a list of the dates where the augmentations occurred
            self._first_aug_entries.append(self._df.index[temp])
            self._next_aug += 1
        # calculate augmentations
        for aug in range(self._next_aug):
            # get first day of augmentation
            aug_initial_date = self._first_aug_entries[aug]
            # calculate degradation for the first and last day of the year, and get the degradation values for each day
            # using interpolation
            # if the first aug day is in the middle of the year (the first day of the year is smaller) the degradation
            # is the first entry of the deg table
            if self._df.index[0] > aug_initial_date:
                first_day_deg = self._get_day_deg(self._df.index[0], aug_initial_date)
            else:
                first_day_deg = self._power_storage.degradation_table[0]
            last_day_deg = self._get_day_deg(self._df.index[-1], aug_initial_date)
            num_of_days = self._df.index.shape[0] // DAY_LENGTH
            bat_deg = [first_day_deg - (x / (num_of_days - 1)) * (first_day_deg - last_day_deg)
                       for x in range(num_of_days)]
            # fill the battery degradation for each hour according to the next discharge hour
            bat_deg = pd.Series(bat_deg, self._daily_index).reindex(self._df.index, method='nearest', copy=False)
            # get dod for each day and fill the battery degradation for each hour according to the next discharge hour
            current_dod = [self._power_storage.dod_table[x] for x in year_diff(self._daily_index, aug_initial_date)]
            current_dod = pd.Series(current_dod, self._daily_index).reindex(self._df.index, method='nearest',
                                                                            copy=False)
            # calculate the current battery capacity for the current augmentation (including degradation, and dod
            # factor)
            self._df[f"aug_{aug}"] = np.where(months_passed >= int(self._power_storage.aug_table[aug][0]),
                                              bat_deg * current_dod * self._power_storage.aug_table[aug][2],
                                              0)

        # sum all augmentations
        self._df["battery_capacity"] = self._df.loc[:, self._df.columns.str.startswith('aug')].sum(axis=1)

    def _calc_overflow(self):
        """
        calculate the hourly overflow of the pv output to bess and grid together
        """
        # calculate overflow of power from pv and bess to grid
        temp = (self._df["pv_output"] - self._grid_size / (1 - self._prod_trans_loss) -
                self._pcs_power / (1 - self._charge_loss) - self._df["battery_nameplate"] *
                self._power_storage.active_self_consumption)
        self._df["overflow"] = np.where(temp > 0, temp, 0)
        # add losses due to overflow
        self._df["acc_losses"] += self._df["overflow"]

    def _calc_pv_to_bess(self):
        """
        calculate the hourly transmission of power from pv to bess
        """
        # calculate excess power from pv (after reducing power that can be sent from pv to grid, and limited by pcs
        # connection size)
        self._active_hourly_self_cons = self._df["battery_nameplate"] * self._power_storage.active_self_consumption
        temp = np.minimum(self._df["pv_output"] - self._grid_size / (1 - self._prod_trans_loss),
                          self._pcs_power / (1 - self._charge_loss) + self._active_hourly_self_cons)
        prelim_pv2bess = np.where(temp > self._active_hourly_self_cons, temp, 0)
        # calculate battery self consumption when pv sends power to it
        hourly_battery_self_consumption = np.where(prelim_pv2bess > 0,
                                                   self._active_hourly_self_cons,
                                                   0)
        # calculate the daily sum of power from pv to bess until each hour
        self._indices = np.arange(self._df.shape[0])
        reductions = np.column_stack((np.maximum(self._indices - self._df.index.hour, 0), self._indices)).ravel()
        temp_sums = np.add.reduceat((prelim_pv2bess - hourly_battery_self_consumption) * (1 - self._charge_loss),
                                    reductions)[::2]
        # calculate the grid overflow that goes to the battery (for hours before the first discharge hour)
        # if hourly pv to bess is bigger than remaining bess capacity only add the remaining capacity to bess
        missing_power = self._df["battery_capacity"] - temp_sums
        # only add self consumption if there is power going from the pv to bess
        self_cons_addition = np.where(missing_power > 0,
                                      hourly_battery_self_consumption,
                                      0)
        pv2bess_pre = np.where(self._df.index.hour < self._bess_discharge_start_hour,
                               np.minimum(prelim_pv2bess, np.maximum(missing_power / (1 - self._charge_loss) +
                                                                     self_cons_addition, 0)),
                               0)
        # calculate loss due to battery size too small
        if self._save_all_results:
            self._df["battery_overflow"] = np.maximum(prelim_pv2bess - pv2bess_pre, 0)

        # calculate the daily amount missing from the battery (daily underflow, including trans loss). broadcast this
        # amount for each hour of the day
        daily_underflow = np.maximum(missing_power[self._df.index.hour == self._bess_discharge_start_hour], 0)

        daily_underflow = daily_underflow[~np.isnan(daily_underflow)]
        # use daily index for underflow value than spread it to each hour of that day (first use daily index, to have
        # the same value for the whole day, even days with different number of hour like daylight saving change days)
        daily_underflow = pd.Series(daily_underflow, self._daily_index)
        daily_underflow = daily_underflow.reindex(self._df.index, method='bfill', copy=False,
                                                  fill_value=daily_underflow.iloc[0])

        # calculate corrected hourly pv to bess (accounting for the underflow in the hours close to the discharge hour)
        # get the index of the hour when starting to discharge
        discharge_hour_indices = pd.Series(np.where(self._df.index.hour == self._bess_discharge_start_hour)[0],
                                           self._daily_index).reindex(self._df.index, method='bfill', copy=False,
                                                                      fill_value=self._indices[-1])
        # get the maximum extra pv that can be delivered to bess (accounting for available pv and remaining pcs
        # capacity)
        max_extra_pv2bess = np.array(np.minimum(np.maximum(self._df["pv_output"], 0),
                                                self._pcs_power / (1 - self._charge_loss) +
                                                self._active_hourly_self_cons) -
                                     pv2bess_pre)
        max_extra_pv2bess = np.where(max_extra_pv2bess + pv2bess_pre > self._active_hourly_self_cons,
                                     max_extra_pv2bess,
                                     0)
        # sum the extra pv to bess to be added in the hours after each hour (and before start of discharge) to
        # compensate for the underflow
        reductions = np.column_stack((np.minimum(self._indices + 1, self._indices[-1]), discharge_hour_indices)). \
            ravel()
        # calculate battery self consumption when pv sends power to it (if pv is already planned to be sent to bess,
        # don't add self consumption again)
        hourly_battery_self_consumption = np.where(pv2bess_pre <= 0,
                                                   self._active_hourly_self_cons,
                                                   0)
        temp_sums = np.where(self._indices != discharge_hour_indices - 1,
                             np.add.reduceat(np.maximum(max_extra_pv2bess - hourly_battery_self_consumption, 0) *
                                             (1 - self._charge_loss), reductions)[::2],
                             0)
        # only add self consumption if there is power going from the pv to bess
        self_cons_addition = np.where(daily_underflow - temp_sums > 0,
                                      hourly_battery_self_consumption,
                                      0)
        # calculate the added hourly pv to bess to compensate for underflow
        extra_pv2bess = np.where(self._df.index.hour < self._bess_discharge_start_hour,
                                 np.minimum(max_extra_pv2bess, np.maximum((daily_underflow - temp_sums) /
                                                                          (1 - self._charge_loss) +
                                                                          self_cons_addition, 0)),
                                 0)

        # final pv to bess (including extra power to account for charge and trans losses and battery self consumption)
        self._df["pv2bess"] = pv2bess_pre + extra_pv2bess

        if self._save_all_results:
            # add losses for power from pv to bess
            temp = self._df["pv2bess"] - np.where(self._df["pv2bess"] > 0, self._active_hourly_self_cons, 0)
            self._df["bess_from_pv"] = temp * (1 - self._charge_loss)
            self._df["acc_losses"] += temp * self._charge_loss

    def _calc_daily_initial_battery_soc(self, year: int):
        """
        calculate the battery soc before daily discharge (including power accounting for losses due to discharge), using
        only power from pv

        :param year: the number of year in the simulation (first year is 0)
        """
        reductions = np.column_stack((np.maximum(self._indices - 12, 0), self._indices)).ravel()
        # calculate battery self consumption in hours when pv charges it
        hourly_battery_self_consumption = np.where(self._df["pv2bess"] > 0,
                                                   self._active_hourly_self_cons,
                                                   0)
        temp_sums = np.where(self._df.index.hour == self._bess_discharge_start_hour,
                             np.add.reduceat(np.array(self._df["pv2bess"] - hourly_battery_self_consumption) *
                                             (1 - self._charge_loss), reductions)[::2],
                             np.nan)
        # fix the sum for the first day, when the discharge hour is early (<12)
        if year == 0:
            # for the first year we don't add power. if the discharge hour is 0 we need to 0 the sum, because of the way
            # reduceat works
            if self._bess_discharge_start_hour == 0:
                temp_sums[0] = 0
        # for other years we add the sum of the pv2bess for the last hours of the last day
        elif self._bess_discharge_start_hour < 12:
            temp = sum(self._df["pv2bess"][-(12 - self._bess_discharge_start_hour):] -
                       hourly_battery_self_consumption[-(12 - self._bess_discharge_start_hour):]) * \
                   (1 - self._charge_loss) / (1 - self._producer.annual_deg)
            if self._bess_discharge_start_hour == 0:
                temp_sums[self._bess_discharge_start_hour] = temp
            else:
                temp_sums[self._bess_discharge_start_hour] += temp

        initial_battery_soc = temp_sums[~np.isnan(temp_sums)]
        # copy value for each hour before discharge hour and fill the last hours of the last day according to the last
        # hours of the first day with degradation factor
        self._initial_battery_soc = pd.Series(initial_battery_soc, self._daily_index) \
            .reindex(self._df.index, method='nearest', copy=False)

    def _calc_grid_to_bess(self, year: int):
        """
        calculate the power needed from the grid to fill the battery

        :param year: the number of year in the simulation (first year is 0)
        """
        # calculate the available bandwidth of the connection to the battery for charge (the connection size minus the
        # power from pv)
        max_rate = np.array(np.minimum(self._pcs_power / (1 - self._grid_bess_loss) + self._active_hourly_self_cons -
                                       self._df["pv2bess"], self._grid_size))
        hours = self._df.index.hour
        # calculate the amount of power missing from the battery at the end of charge (including discharge losses)
        missing_bess_power = (self._df["battery_capacity"] - self._initial_battery_soc) / (1 - self._grid_bess_loss)

        # calculate the sum of power already inserted to battery. for the charge hours in the same day this is the
        # sum of hours until the first discharge hour. for the last hours of the previous day this is the sum for the
        # charge hours of the next day and the hours after this hour in this day
        # TODO: fix so it will work correctly for daylight saving day (where the difference between discharge hour of
        #  this day and the next are not 24)
        last_indices = self._indices + np.where(hours < self._bess_discharge_start_hour,
                                                (self._bess_discharge_start_hour - hours),
                                                self._bess_discharge_start_hour + DAY_LENGTH - hours)
        last_index = self._indices[-1]
        reductions = np.column_stack((np.minimum(self._indices + 1, last_index),
                                      np.minimum(last_indices, last_index))).ravel()
        # calculate battery self consumption for hours when only grid charges it
        hourly_battery_self_consumption = np.where(self._df["pv2bess"] <= 0,
                                                   self._active_hourly_self_cons,
                                                   0)
        # use modulo for case discharge start hour is 0
        max_rate_sums = np.where(hours != (self._bess_discharge_start_hour - 1 % DAY_LENGTH),
                                 np.add.reduceat((max_rate - hourly_battery_self_consumption), reductions)[::2],
                                 0)
        # the last index will get as the sum of its own max rate (because of how reduceat works) so we need to zero it
        # before adding the additional sum for the next day
        max_rate_sums[-1] = 0
        # for the last day, for hours past the discharge hour, add the power inserted to the battery in the charge
        # hours of the first day (to account for power inserted in the next day)
        max_rate_sums[-(DAY_LENGTH - self._bess_discharge_start_hour):] += \
            sum(max_rate[:self._bess_discharge_start_hour] -
                hourly_battery_self_consumption[:self._bess_discharge_start_hour]) * \
            self._power_storage.degradation_table[year + 1] / \
            self._power_storage.degradation_table[year]
        # only add self consumption if there is power from grid to bess
        hourly_battery_self_consumption = np.where(missing_bess_power - max_rate_sums > 0,
                                                   hourly_battery_self_consumption,
                                                   0)

        self._df["grid2bess"] = np.maximum(np.minimum(max_rate,
                                                      missing_bess_power - max_rate_sums +
                                                      hourly_battery_self_consumption),
                                           0)

        self._initial_battery_soc[:] = self._df["battery_capacity"]
        # for the first day of the first year, manually sum the power inserted to the battery (for cases where the power
        # inserted in this first day is not enough to fill the battery)
        if year == 0:
            self._initial_battery_soc[:self._bess_discharge_start_hour + 12] = \
                sum((self._df["pv2bess"][:self._bess_discharge_start_hour] -
                     np.where(self._df["pv2bess"][:self._bess_discharge_start_hour] > 0,
                              self._active_hourly_self_cons[:self._bess_discharge_start_hour], 0)) *
                    (1 - self._charge_loss) +
                    (self._df["grid2bess"][:self._bess_discharge_start_hour] -
                     np.where((self._df["pv2bess"][:self._bess_discharge_start_hour] <= 0) &
                              (self._df["grid2bess"][:self._bess_discharge_start_hour] > 0),
                              self._active_hourly_self_cons[:self._bess_discharge_start_hour], 0)) *
                    (1 - self._grid_bess_loss))

        if self._save_all_results:
            # add losses for power for grid to bess
            self._df["acc_losses"] += self._df["grid2bess"] * self._grid_bess_loss
            self._df["bess_from_grid"] = self._df["grid2bess"] * (1 - self._grid_bess_loss)

    def _calc_power_to_grid(self, year: int):
        """
        calculate the hourly power transmitted to the grid from both pv and bess

        :param year: the number of year in the simulation (first year is 0)
        """
        # calculate hourly pv to grid and gri
        self._df["pv2grid"] = np.minimum((self._df["pv_output"] - self._df["pv2bess"]),
                                         self._grid_size / (1 - self._prod_trans_loss))
        grid_from_pv = np.where(self._df["pv2grid"] > 0,
                                self._df["pv2grid"] * (1 - self._prod_trans_loss),
                                0)
        self._df["grid2pv"] = np.where(self._df["pv2grid"] > 0,
                                       0,
                                       -self._df["pv2grid"] / (1 - self._prod_trans_loss))
        if self._save_all_results:
            # add losses for power for pv to grid
            self._df["acc_losses"] += np.maximum(self._df["pv2grid"] * self._prod_trans_loss, 0)

        # calculate hourly bess to grid + self consumption
        # maximum hourly from bess to grid given the hourly pv to grid (including battery self consumption)
        total_discharge_factor = (1 - self._grid_bess_loss) * self._power_storage.rte_table[
            year]
        max_rate = np.array((self._grid_size - np.maximum(grid_from_pv, 0)) /
                            total_discharge_factor + self._active_hourly_self_cons)

        # calculate the sum of power already discharged from battery. for the discharge hours in the same day this is
        # the sum of hours since the first discharge hour. for the first hours of the next day this is the sum for the
        # discharge hours of the previous day and the hours before this hour in this day
        hours = self._df.index.hour
        first_indices = self._indices - np.where(self._bess_discharge_start_hour <= hours,
                                                 (hours - self._bess_discharge_start_hour),
                                                 DAY_LENGTH - self._bess_discharge_start_hour + hours)
        reductions = np.column_stack((np.maximum(first_indices, 0), self._indices)).ravel()
        max_rate_sums = np.where(hours == self._bess_discharge_start_hour,
                                 0,
                                 np.add.reduceat(max_rate, reductions)[::2])
        # fix first hour of the year sums itself instead of nothing
        max_rate_sums[0] = 0
        # for the first day add power discharged from battery in the discharge hours of the last day (to account for the
        # power discharged in the previous day)
        bat_deg_ratio = self._power_storage.degradation_table[year] / self._power_storage.degradation_table[year + 1]
        max_rate_sums[:self._bess_discharge_start_hour] += (max_rate_sums[-1] + max_rate[-1]) * bat_deg_ratio

        self._df["bess2grid"] = np.maximum(np.minimum(max_rate,
                                                      self._initial_battery_soc - max_rate_sums),
                                           0)

        # reduce power by the losses for discharge and self consumption in discharge hours
        grid_from_bess = np.where((self._df["bess2grid"] > self._active_hourly_self_cons),
                                  (self._df["bess2grid"] - self._active_hourly_self_cons) * total_discharge_factor,
                                  0)

        # reducing self consumption from pv to grid or add to grid to bess in charge and idle hours
        self._idle_hourly_self_cons = self._df["battery_nameplate"] * self._power_storage.idle_self_consumption
        active_hours = (self._df["bess2grid"] > 0) | (self._df["pv2bess"] > 0) | (self._df["grid2bess"] > 0)
        self._df["grid2bess"] += np.where(((grid_from_pv > self._idle_hourly_self_cons) & ~active_hours) |
                                          active_hours,
                                          0,
                                          self._idle_hourly_self_cons)
        grid_from_pv -= np.where((grid_from_pv > self._idle_hourly_self_cons) & ~active_hours,
                                 self._idle_hourly_self_cons,
                                 0)

        # save the power output of the system
        self._df["output"] = grid_from_pv + grid_from_bess

        if self._save_all_results:
            self._df["grid_from_pv"] = grid_from_pv
            self._df["grid_from_bess"] = grid_from_bess
            # calculate hourly soc (before adding idle self consumption from grid)
            self._calc_soc(year, bat_deg_ratio)
            # add losses for self consumption of the battery and losses for discharging battery
            bess_self_consumption = np.where(active_hours,
                                             self._active_hourly_self_cons,
                                             self._idle_hourly_self_cons)
            self._df["acc_losses"] += bess_self_consumption + self._df["bess2grid"] * (1 - total_discharge_factor)

    def _calc_soc(self, year, bat_deg_ratio):
        """
        calculate soc of the battery in every hour

        :param year: the number of year in the simulation (first year is 0)
        :param bat_deg_ratio: the ratio of the degradation for the battery in this year by to the one for the next year
        """
        # get the first hour of charge each day
        temp = pd.Series((self._df["pv2bess"] > 0) | (self._df["grid2bess"] > self._idle_hourly_self_cons),
                         self._df.index, name="temp")
        first_charge_hour = temp.groupby(pd.Grouper(freq='D')).idxmax()
        first_charge_hour = first_charge_hour.reindex(self._df.index, method="ffill", copy=False).dt.hour

        first_charge_hour_indices = pd.Series(np.where(self._df.index.hour == first_charge_hour)[0],
                                              pd.date_range(self._df.index[0],
                                                            self._df.index[-1], freq='d'))
        first_charge_hour_indices = first_charge_hour_indices.reindex(self._df.index, method='ffill', copy=False)
        # calculate the hourly self consumption of the battery (in the charge hours)
        pv_battery_self_cons = np.where(self._df["pv2bess"] > 0, self._active_hourly_self_cons, 0)
        grid_battery_self_cons = np.where((self._df["grid2bess"] > 0) & (self._df["pv2bess"] <= 0),
                                          self._active_hourly_self_cons,
                                          0)
        # sum charge and discharge until each hour
        reductions = np.column_stack((first_charge_hour_indices, self._indices)).ravel()
        temp_sums = np.add.reduceat(((self._df["pv2bess"] - pv_battery_self_cons) * (1 - self._charge_loss) +
                                     (self._df["grid2bess"] - grid_battery_self_cons) * (1 - self._grid_bess_loss) -
                                     self._df["bess2grid"]).to_numpy(),
                                    reductions)[::2]

        self._df["soc"] = np.where(self._df.index.hour > first_charge_hour,
                                   temp_sums,
                                   0)

        # calc soc for hour before charging start
        last_hour_soc = self._df.loc[self._df.index.hour == 23, "soc"]
        last_hour_soc = last_hour_soc.reindex(self._df.index, method="bfill", copy=False)
        last_hour_soc[:23] = 0 if year == 0 else last_hour_soc.iloc[-1] * bat_deg_ratio
        reductions = np.column_stack((np.maximum(self._indices - self._df.index.hour - 1, 0),
                                      self._indices)).ravel()
        temp_sums = np.add.reduceat(np.array(self._df["bess2grid"]), reductions)[::2]
        # fix first hours of the year value by adding the last hour bess to grid (in the first year there is no bess
        # to grid in the previous day)
        if year != 0:
            temp_sums[0] = 0
            temp_sums[:first_charge_hour.iloc[0]] += self._df["bess2grid"].iloc[-1] * bat_deg_ratio + \
                (self._active_hourly_self_cons.iloc[-1] if self._df["bess2grid"].iloc[-1] > 0 else 0)
        self._df["soc"] += np.where((self._df.index.hour <= first_charge_hour) &
                                    (self._df["bess2grid"] > 0),
                                    last_hour_soc - temp_sums,
                                    0)
        self._df["soc"] = np.where(self._df["battery_nameplate"] > 0,
                                   self._df["soc"] / self._df["battery_nameplate"] * 100,
                                   0)

    def _calc_power_flow(self, year):
        """
        Calculate the power flow in the system for the given year

        :param year: the year to calculate for (starting at 0 for the first year of simulation)
        """
        self._calc_pv_to_bess()
        self._calc_daily_initial_battery_soc(year)
        if self._fill_battery_from_grid:
            self._calc_grid_to_bess(year)
        else:
            self._df["grid2bess"] = 0
            self._df["bess_from_grid"] = 0
        self._calc_power_to_grid(year)

    def run(self):
        """
        run the calculation and save hourly results into 'results' (power is in kW):
            (pv_output - output of the pv system,
            overflow - pv overflow to grid and bess together (only when save_all_results is true),
            aug_{i} - battery capacity for augmentation number i (when use_augmentation is true),
            battery_capacity - total capacity of battery (including all augmentations) (when use_augmentation is true),
            battery_overflow - loss due to limit of battery capacity and grid size (only when save_all_results is true),
            pv2bess - power from pv to bess (after underflow),
            bess_from_pv - power from pv to bess after losses (only when save_all_results is true),
            grid2bess - power from grid to bess (to fill battery capacity),
            bess_from_grid - power from grid to bess after losses (only when save_all_results is true),
            grid2pv - power from grid to pv,
            pv2grid - power from pv to grid,
            bess2grid - power from bess to grid,
            grid_from_pv - power from pv to grid after losses (only when save_all_results is true),
            grid_from_bess - power from bess to grid after losses (only when save_all_results is true),
            output - power form pv+bess to grid (after losses),
            battery_nameplate - the nameplate value of the battery,
            acc_losses - the accumulated losses of the system (only when save_all_results is true),
            soc - the state of charge of the BESS (only when save_all_results is true))

        also save hourly output to 'output'
        """
        # reset augmentations variables
        self._next_aug = 0
        self._first_aug_entries = []

        # reset pcs power (in case power storage has changed)
        self._set_pcs_power()

        if self._save_all_results:
            self._results = []
        self._output = []
        self._purchased_from_grid = []
        for year in range(self._num_of_years):
            self._get_data(year)
            self._calc_augmentations()
            if self._save_all_results:
                self._calc_overflow()
            self._calc_power_flow(year)
            self._purchased_from_grid.append(self._df["grid2bess"] + self._df["grid2pv"])
            if self._save_all_results:
                self._results.append(self._df.copy(deep=True))
            self._output.append(self._df["output"].copy(deep=True))

    def monthly_averages(self, years: Iterable[int] = (0,), stat: str = 'output'):
        """
        calculate and print the average in each hour for each month across the given years

        :param years: iterable of years to calculate for (refer to years since year one, acceptable values are between
            0, inclusive, and num_of_years, exclusive)
        :param stat: the stat to calculate
        """
        averages = []
        for year in years:
            if not 0 <= year < self._num_of_years:
                raise ValueError("years provided not in range of calculation years")
            if stat != 'output':
                if self._results is None:
                    raise ValueError("The calculator full results are not available (ether you didn't use run, or used "
                                     "save_all_result=False)")
                if stat not in self._results[year]:
                    raise ValueError(f"Stat '{stat}' is not present in results")
                months_data = [self._results[year][stat][self._results[year].index.month == month] for month in
                               range(1, 13)]
            # divide the data to months
            elif self._output is None:
                raise ValueError("The calculator results are not available (use run to generate output)")
            else:
                months_data = [self._output[year][self._output[year].index.month == month] for month in range(1, 13)]
            # divide each month to hours (array for each month containing array for each hour, with value for every day
            # of the month)
            year_data = [[month[month.index.hour == hour] for hour in range(0, DAY_LENGTH)] for month in months_data]
            averages.append([[round(y.mean()) for y in x] for x in year_data])
        averages = np.mean(averages, axis=0, dtype=int)
        averages = np.vstack([list(range(DAY_LENGTH)), averages])
        return averages

    def plot_stat(self, years: Iterable[int] | None = None, stat: str = "output"):
        """
        plot a graph of the given stat over the years

        :param years: iterable of years to calculate for (refer to years since year one, acceptable values are between
            0, inclusive, and num_of_years, exclusive)
        :param stat: the stat to plot
        """
        # check stat is in results
        if self._results is None:
            raise ValueError("The calculator full results are not available (either you didn't use run, or used "
                             "save_all_result=False)")
        if stat not in self._results[0]:
            raise ValueError(f"Stat '{stat}' is not present in results")
        if years is None:
            years = range(self._num_of_years)
        # gather stat from all the years and plot
        stat_data = []
        for year in years:
            if not 0 <= year < self._num_of_years:
                raise ValueError("years provided not in range of calculation years")
            if stat in ["battery_capacity", "battery_nameplate"] or stat.startswith("aug"):
                stat_data.append(self._results[year][stat])
            else:
                stat_data.append(self._results[year][stat].groupby(self._results[year][stat].index.date).sum())
        all_data = pd.concat(stat_data)
        all_data.plot()
        plt.show()


class NNOutputCalculator(OutputCalculator):
    """
    Calculates the hourly output of the pv system and the storage system, using a neural network for daily scheduling
    of charge/discharge
    """

    POS_ENCODING_DIM = 24
    EPSILON = 0.00000001
    DISCRETE_SPLIT_COUNT = 10
    MAX_BAT_HOURS = 8

    def __init__(self, tariff_table: np.ndarray[Any, np.dtype[np.float64]] | None = None, sell_prices=None,
                 buy_prices=None, *args, **kwargs):
        """
        Initialize the calculator with info for the system

        :param tariff_table: a numpy array with tariff for every hour in every month (optional, ignored if sell_prices
            is provided)
        :param sell_prices: the price for selling power in each hour of a year (1d array with floats, must be provided
            if tariff table is not provided)
        :param buy_prices: the prices for buying power in each hour of the year (1d array with floats, if None uses sell
            prices)
        """
        if tariff_table is None and sell_prices is None:
            raise ValueError("NNOutputCalculator expects either sell prices or tariff table!")
        super().__init__(*args, **kwargs)
        self._fill_battery_from_grid = True
        self.tariff_table = tariff_table
        if sell_prices is not None:
            self.sell_prices = sell_prices
            self.buy_prices = buy_prices if buy_prices is not None else sell_prices
        else:
            self._sell_prices = self._buy_prices = None
        self._model_session = onnxruntime.InferenceSession(os.path.join(os.path.dirname(__file__),
                                                                        "schedule_model.onnx"),
                                                           providers=["CPUExecutionProvider"])
        self._pos_encodes = self._create_pos_encoding()

    def _create_pos_encoding(self):
        """
        create positional encoding for hours of a day
        """
        pos_encodes = [0] * DAY_LENGTH
        # create positional encoding for each hour
        for pos in range(DAY_LENGTH):
            encoded = [pos / 100 ** (2 * (i // 2) / self.POS_ENCODING_DIM) for i in range(self.POS_ENCODING_DIM)]
            encoded = [math.sin(encoded[i]) if i % 2 == 0 else math.cos(encoded[i]) for i in
                       range(self.POS_ENCODING_DIM)]
            pos_encodes[pos] = encoded
        return pos_encodes

    # disables the property as the class always assume it is true
    @property
    def fill_battery_from_grid(self):
        return True

    @fill_battery_from_grid.setter
    def fill_battery_from_grid(self, value):
        pass

    @property
    def sell_prices(self):
        return self._sell_prices

    @sell_prices.setter
    def sell_prices(self, value: np.ndarray[Any, np.dtype[np.float64]]):
        if value is not None and isinstance(value, np.ndarray) and is_real_numbers(value):
            if value.shape != (YEAR_HOURS,) and value.shape != (self._project_hour_num,):
                raise ValueError(f"Prices shape should be ({YEAR_HOURS},) or ({self._project_hour_num}, ), prices"
                                 f" for each hour in a year or for each hour of every year")
            # compare shapes if buy prices were set
            try:
                if value.shape != self._buy_prices.shape:
                    raise ValueError("Sell prices and buy prices should have the same shape")
            except AttributeError:
                pass
            self._sell_prices = value
        else:
            raise ValueError("Prices should be a numpy array of floats")

    @property
    def buy_prices(self):
        return self._buy_prices

    @buy_prices.setter
    def buy_prices(self, value: np.ndarray[Any, np.dtype[np.float64]]):
        if value is not None and isinstance(value, np.ndarray) and is_real_numbers(value):
            if value.shape != (YEAR_HOURS,) and value.shape != (self._project_hour_num, ):
                raise ValueError(f"Prices shape should be ({YEAR_HOURS},) or ({self._project_hour_num}, ), prices"
                                 f" for each hour in a year or for each hour of every year")
            try:
                if value.shape != self._sell_prices.shape:
                    raise ValueError("Sell prices and buy prices should have the same shape")
            except AttributeError:
                pass
            self._buy_prices = value
        else:
            raise ValueError("Prices should be a numpy array of floats")

    @property
    def tariff_table(self):
        """
        a table with power tariffs for each hour of the day in each month
        """
        return self._tariff_table

    @tariff_table.setter
    def tariff_table(self, value: np.ndarray[Any, np.dtype[np.float64]]):
        if value is not None:
            if value.shape != (7, 12, DAY_LENGTH):
                raise ValueError("Tariff table should be of shape (7, 12, 24)")
        self._tariff_table = value

    def _get_action_from_obs(self, obs: np.ndarray[Any, np.dtype[np.float64]]):
        """
        get action(s) from model given an observation

        :param obs: the given observation
        """
        # pass inputs through the model
        input_dict = {self._model_session.get_inputs()[0].name: obs, self._model_session.get_inputs()[1].name: []}
        outs = self._model_session.run(None, input_dict)[0]
        # get discrete actions from model outputs
        outs_split = np.split(outs, [2 * self.DISCRETE_SPLIT_COUNT + 1, (2 * self.DISCRETE_SPLIT_COUNT + 1) * 2],
                              axis=1)
        results = np.stack([np.argmax(out_, -1) for out_ in outs_split], axis=1)
        return results.squeeze().astype(np.float32)

    def _convert_action_to_range(self, actions):
        """
        converted action(s) from discrete value to a range (-1 to 1 or 0 to 1)

        :param actions: the actions to convert
        """
        actions[:, 0] = clamp((actions[:, 0] - self.DISCRETE_SPLIT_COUNT) / float(self.DISCRETE_SPLIT_COUNT))
        actions[:, 1] = (actions[:, 1] - self.DISCRETE_SPLIT_COUNT) / float(self.DISCRETE_SPLIT_COUNT)
        actions[:, 2] = np.where(actions[:, 2] == 0, -1, 1)
        return actions

    def _get_actions(self, year):
        """
        generate the actions from pytorch model for the system, which are use for power flow decisions

        :param year: the year to calculate for (starting at 0 for the first year of simulation)
        """
        # if leap year add day
        day_add = self._is_leap_year
        hour_add = self._is_leap_year * DAY_LENGTH
        # get prices according to init inputs
        sell_prices, buy_prices = get_yearly_prices(year, self._sell_prices, self._buy_prices, self.tariff_table,
                                                    self._producer.start_year, self._yearly_hour_num)
        # normalize hourly power
        normalization_factor = self.producer.rated_power * self.MAX_BAT_HOURS
        normalized_hourly_power = np.expand_dims(self._df["pv_output"].values / normalization_factor, 1)
        # create positional encoding for every day
        pos_encoding = np.tile(self._pos_encodes, (YEAR_DAYS + day_add, 1))

        ast = np.lib.stride_tricks.as_strided
        # create array with pv power, sell prices and buy prices for every day, repeated for each hour, normalized with
        # positional encoding added for each hour
        stride = self._df["pv_output"].values.strides[0]
        split_hourly_power = ast(normalized_hourly_power, (YEAR_DAYS + day_add, DAY_LENGTH),
                                 (DAY_LENGTH * stride, stride))
        split_hourly_power = np.repeat(split_hourly_power, DAY_LENGTH, axis=0) + pos_encoding
        split_prices = ast(sell_prices, (YEAR_DAYS + day_add, DAY_LENGTH), (DAY_LENGTH, sell_prices.strides[0]))
        max_sell_prices = np.repeat(np.max(np.abs(split_prices), axis=1) + self.EPSILON, DAY_LENGTH)
        daily_sell_prices = np.repeat(split_prices, DAY_LENGTH, axis=0) / max_sell_prices[:, None] + pos_encoding
        hourly_sell_prices = np.expand_dims(sell_prices / max_sell_prices, 1)
        split_prices = ast(buy_prices, (YEAR_DAYS + day_add, DAY_LENGTH), (DAY_LENGTH, buy_prices.strides[0]))
        max_buy_prices = np.repeat(np.max(np.abs(split_prices), axis=1) + self.EPSILON, DAY_LENGTH)
        daily_buy_prices = np.repeat(split_prices, DAY_LENGTH, axis=0) / max_buy_prices[:, None] + pos_encoding
        hourly_buy_prices = np.expand_dims(buy_prices / max_buy_prices, 1)
        # create repeated normalized project parameters
        grid_sizes = np.expand_dims(np.full(YEAR_HOURS + hour_add, self._grid_size / normalization_factor), 1)
        pcs_vec = np.expand_dims(np.full(YEAR_HOURS + hour_add, self._pcs_power / normalization_factor), 1)
        normalized_capacities = np.expand_dims(self._df["battery_capacity"].values / normalization_factor, 1)
        # create vectorized observation for a whole years
        obs = np.concatenate((grid_sizes, normalized_capacities, pcs_vec, normalized_hourly_power,
                              hourly_sell_prices, hourly_buy_prices, split_hourly_power, daily_sell_prices,
                              daily_buy_prices),
                             axis=1, dtype=np.float32)

        # get the actions
        a = self._get_action_from_obs(obs)
        # convert from discrete
        a = self._convert_action_to_range(a)
        self._actions = a

    def _calc_power_flow(self, year):
        # get actions from model
        self._get_actions(year)

        # if leap year add day
        day_add = self._is_leap_year
        hour_add = self._is_leap_year * DAY_LENGTH

        # initialize arrays to fill
        pv2grid = np.zeros(YEAR_HOURS + hour_add)
        pv2bess = np.zeros(YEAR_HOURS + hour_add)
        grid2bess = np.zeros(YEAR_HOURS + hour_add)
        bess2grid = np.zeros(YEAR_HOURS + hour_add)
        grid2pv = np.zeros(YEAR_HOURS + hour_add)
        output = np.zeros(YEAR_HOURS + hour_add)
        soc = np.zeros(YEAR_DAYS + day_add)

        if self._save_all_results:
            pv2bess_pre_acc = np.zeros(YEAR_HOURS + hour_add)
            bess_from_pv = np.zeros(YEAR_HOURS + hour_add)
            bess_from_grid = np.zeros(YEAR_HOURS + hour_add)
            grid_from_pv = np.zeros(YEAR_HOURS + hour_add)
            grid_from_bess = np.zeros(YEAR_HOURS + hour_add)
            acc_losses = np.zeros(YEAR_HOURS + hour_add)
            acc_soc = np.zeros(YEAR_HOURS + hour_add)

        # get parameters
        trans_ret = 1 - self._prod_trans_loss
        charge_ret = 1 - self._charge_loss
        grid_bess_ret = 1 - self._grid_bess_loss
        active_self_cons = self._df["battery_nameplate"].values * self._power_storage.active_self_consumption
        idle_self_cons = self._df["battery_nameplate"].values * self._power_storage.idle_self_consumption
        rte = self._power_storage.rte_table[year]

        for i in range(DAY_LENGTH):
            # get values for current hour
            hour_pv_output = self._df["pv_output"].values[i::DAY_LENGTH]
            hour_active_self_cons = active_self_cons[i::DAY_LENGTH]
            hour_idle_self_cons = idle_self_cons[i::DAY_LENGTH]
            hour_actions = self._actions[i::DAY_LENGTH]

            # calculate division of pv power to grid and bess
            # get excess pv power that can only go to bess
            available_cap = self._df["battery_capacity"].values[i::DAY_LENGTH] - soc
            added_self_cons = np.where(available_cap > 0, hour_active_self_cons, 0)
            pv2bess_pre = relu(np.minimum(np.minimum(relu(hour_pv_output - self._grid_size / trans_ret),
                                                     available_cap / charge_ret + added_self_cons),
                                          self._pcs_power / charge_ret + hour_active_self_cons))
            # get additional available power to bess from pv
            added_self_cons = np.where((pv2bess_pre < hour_active_self_cons) & (available_cap > 0),
                                       hour_active_self_cons, 0)
            max_extra_pv2bess = relu(np.minimum(np.minimum(relu(hour_pv_output),
                                                           self._pcs_power / charge_ret + hour_active_self_cons),
                                                available_cap / charge_ret + added_self_cons) - pv2bess_pre)
            charge_condition = hour_actions[:, 2] > 0
            # charge option calculations
            # calculate total power from pv to bess according to the actions
            # if there is excess power use it to charge even if not a charge option
            pv2bess[i::DAY_LENGTH] = np.where(charge_condition,
                                              np.minimum(pv2bess_pre + max_extra_pv2bess * hour_actions[:, 0],
                                                         self._pcs_power / charge_ret + hour_active_self_cons),
                                              np.minimum(pv2bess_pre, self._pcs_power / charge_ret +
                                                         hour_active_self_cons))
            net_pv2bess = np.where(charge_condition, relu(pv2bess[i::DAY_LENGTH] - hour_active_self_cons) * charge_ret,
                                   0)
            # remaining pv power goes to grid
            pv2grid[i::DAY_LENGTH] = np.minimum(hour_pv_output - pv2bess[i::DAY_LENGTH], self._grid_size / trans_ret)
            net_pv2grid = np.where(pv2grid[i::DAY_LENGTH] > 0, pv2grid[i::DAY_LENGTH] * trans_ret,
                                   -pv2grid[i::DAY_LENGTH] / trans_ret)
            # calculate power from grid to bess, first the max power and then the actual power according to the actions
            added_self_cons = np.where(pv2bess[i::DAY_LENGTH] < hour_active_self_cons, hour_active_self_cons, 0)
            option1 = (available_cap - net_pv2bess) / grid_bess_ret
            option1 = np.where(option1 > 0, option1 + added_self_cons, 0)
            option2 = (self._pcs_power - net_pv2bess) / grid_bess_ret
            option2 = np.where(option2 > 0, option2 + added_self_cons, 0)
            max_grid2bess = relu(np.minimum(np.minimum(self._grid_size - net_pv2grid, option1), option2))
            grid2bess[i::DAY_LENGTH] = np.where(charge_condition, max_grid2bess * relu(-hour_actions[:, 1]), 0)
            net_grid2bess = np.where(charge_condition, relu(grid2bess[i::DAY_LENGTH] - added_self_cons) * grid_bess_ret,
                                     0)
            # discharge option calculations
            # calculate power from bess to grid (pv power sold first), first the max power and then the actual power
            # according to the actions
            max_bess2grid = np.where(hour_pv_output > 0,
                                     relu(np.minimum(np.minimum((self._grid_size - hour_pv_output *
                                                                trans_ret) / rte / grid_bess_ret +
                                                                hour_active_self_cons, soc), self._pcs_power)),
                                     relu(np.minimum(soc, self._pcs_power + hour_pv_output /
                                                     charge_ret)))
            bess2grid[i::DAY_LENGTH] = np.where(charge_condition,
                                                0,
                                                relu(max_bess2grid - hour_active_self_cons) * hour_actions[:, 1])
            # use bess power for pv self consumption if available, and remove it from power to grid
            bess2pv = np.where(~charge_condition & (hour_pv_output < 0) &
                               (bess2grid[i::DAY_LENGTH] > -hour_pv_output / charge_ret),
                               -hour_pv_output / charge_ret, 0)
            bess2grid[i::DAY_LENGTH] += np.where(bess2grid[i::DAY_LENGTH] > 0, hour_active_self_cons, 0) - bess2pv
            # account for self consumption (not yet accounted for in previous calculations)
            condition1 = (bess2grid[i::DAY_LENGTH] == 0) & (pv2bess[i::DAY_LENGTH] + grid2bess[i::DAY_LENGTH] <
                                                            hour_active_self_cons)
            condition2 = pv2grid[i::DAY_LENGTH] > hour_idle_self_cons
            pv2bess[i::DAY_LENGTH] += np.where(condition1 & condition2, hour_idle_self_cons, 0)
            pv2grid[i::DAY_LENGTH] -= np.where(condition1 & condition2, hour_idle_self_cons, 0)
            grid2bess[i::DAY_LENGTH] = np.where(condition1 & ~condition2, hour_idle_self_cons, grid2bess[i::DAY_LENGTH])
            # update soc after power flow for hour i
            soc += net_pv2bess + net_grid2bess - bess2grid[i::DAY_LENGTH] - bess2pv
            # calculate the power purchased from for pv self consumption
            grid2pv[i::DAY_LENGTH] = np.where(pv2grid[i::DAY_LENGTH] / trans_ret + bess2pv < 0, -pv2grid[i::DAY_LENGTH]
                                              / trans_ret, 0)

            # calculate the total power output to grid
            net_bess2grid = relu((bess2grid[i::DAY_LENGTH] - hour_active_self_cons) * grid_bess_ret * rte)
            net_pv2grid = relu(pv2grid[i::DAY_LENGTH]) * trans_ret
            output[i::DAY_LENGTH] = net_pv2grid + net_bess2grid

            # save additional stats for analysis
            if self._save_all_results:
                pv2bess_pre_acc[i::DAY_LENGTH] = pv2bess_pre
                bess_from_pv[i::DAY_LENGTH] = net_pv2bess
                bess_from_grid[i::DAY_LENGTH] = net_grid2bess
                grid_from_pv[i::DAY_LENGTH] = net_pv2grid
                grid_from_bess[i::DAY_LENGTH] = net_bess2grid
                acc_losses[i::DAY_LENGTH] = (pv2bess[i::DAY_LENGTH] - net_pv2bess + grid2bess[i::DAY_LENGTH] -
                                             net_grid2bess + pv2grid[i::DAY_LENGTH] - net_pv2grid +
                                             bess2grid[i::DAY_LENGTH] - net_bess2grid)
                acc_soc[i::DAY_LENGTH] = soc

        # fix for when soc at last hour is not 0
        for i in range(DAY_LENGTH - 1, -1, -1):
            # stop when all soc are 0
            if np.all(soc < active_self_cons[i::DAY_LENGTH]):
                break

            can_discharge = pv2bess[i::DAY_LENGTH] + grid2bess[i::DAY_LENGTH] <= idle_self_cons[i::DAY_LENGTH]
            # find the amount that can be added
            added_amount = relu(np.minimum((self.grid_size - pv2grid[i::DAY_LENGTH] * trans_ret -
                                            bess2grid[i::DAY_LENGTH] * rte * grid_bess_ret) / rte / grid_bess_ret, soc))
            # only add discharge if it is more than self consumption
            added_amount = np.where(added_amount > active_self_cons[i::DAY_LENGTH], added_amount, 0)
            added_self_cons = np.where((added_amount > 0) & (bess2grid[i::DAY_LENGTH] <
                                                             active_self_cons[i::DAY_LENGTH]),
                                       active_self_cons[i::DAY_LENGTH],
                                       0)
            # correct amount if using all remaining soc and some is used for self consumption
            added_amount = np.where(added_amount + added_self_cons >= soc, soc - added_self_cons, added_amount)
            # increase the amount of energy from grid to bess and decrease the soc accordingly
            bess2grid[i::DAY_LENGTH] += np.where(can_discharge, added_amount + added_self_cons, 0)
            soc -= added_amount + added_self_cons
            # removed idle self consumption if added discharge
            net_added_amount = np.where(can_discharge, added_amount * rte * grid_bess_ret, 0)
            grid2bess[i::DAY_LENGTH] -= np.where((net_added_amount > 0) & (grid2bess[i::DAY_LENGTH] ==
                                                                           idle_self_cons[i::DAY_LENGTH]),
                                                 idle_self_cons[i::DAY_LENGTH],
                                                 0)
            pv2bess[i::DAY_LENGTH] -= np.where((net_added_amount > 0) & (pv2bess[i::DAY_LENGTH] ==
                                                                         idle_self_cons[i::DAY_LENGTH]),
                                               idle_self_cons[i::DAY_LENGTH],
                                               0)

            # update the saved data
            if self._save_all_results:
                grid_from_bess[i::DAY_LENGTH] = net_added_amount
                acc_losses[i::DAY_LENGTH] += np.where(can_discharge,
                                                      relu(added_self_cons + added_amount - net_added_amount -
                                                           idle_self_cons[i::DAY_LENGTH]),
                                                      0)
                for j in range(i, DAY_LENGTH):
                    acc_soc[j::DAY_LENGTH] -= added_amount + added_self_cons

        # save power flow to dataframe
        self._df["pv2bess"] = pv2bess
        self._df["grid2bess"] = grid2bess
        self._df["pv2grid"] = pv2grid
        self._df["grid2pv"] = grid2pv
        self._df["bess2grid"] = bess2grid
        self._df["output"] = output

        # save additional stats to dataframe
        if self._save_all_results:
            self._df["battery_overflow"] = relu(self._df["pv_output"] - pv2bess_pre_acc)
            self._df["bess_from_pv"] = bess_from_pv
            self._df["bess_from_grid"] = bess_from_grid
            self._df["grid_from_pv"] = grid_from_pv
            self._df["grid_from_bess"] = grid_from_bess
            self._df["acc_losses"] = acc_losses
            self._df["soc"] = acc_soc
