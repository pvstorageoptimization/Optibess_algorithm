import time

from numba import njit, float64

from Optibess_algorithm.output_calculator import OutputCalculator, Coupling
from Optibess_algorithm.financial_calculator import FinancialCalculator
from Optibess_algorithm.constants import *
from Optibess_algorithm.power_storage import LithiumPowerStorage, PowerStorage
from Optibess_algorithm.producers import PvProducer, Producer


# def discharge_algo(h, pre_pv2bess, pv_rest, soc, p_sell, p_buy):
#     def helper_algo(current_h, current_soc):
#         if current_h < 0:
#             if current_soc < 0:
#                 return -math.inf
#             else:
#                 return 0
#
#         # discharge option
#         if pv_rest[current_h] >= grid_size or (current_h > 0 and p_sell[current_h] < min(p_sell[:current_h])):
#             option1 = -math.inf
#         else:
#             x = min(grid_size, pcs - pv_rest[current_h])
#             option1 = (x + pv_rest[current_h]) * p_sell[current_h] + helper_algo(current_h - 1,
#                                                                                  current_soc - x)
#         # only overflow charge option
#         if pre_pv2bess[current_h] <= 0:
#             option2 = -math.inf
#         else:
#             x = min(pre_pv2bess[current_h], capacity - current_soc)
#             option2 = pv_rest[current_h] * p_sell[current_h] + helper_algo(current_h - 1,
#                                                                            min(current_soc + x, 0))
#         # if no charge needed, done consider the other options
#         if current_soc >= 0:
#             return max(option1, option2)
#         # only pv charge option
#         if pv_rest[current_h] <= 0:
#             option3 = -math.inf
#         else:
#             x = min(pcs, capacity - current_soc, pre_pv2bess[current_h] + pv_rest[current_h])
#             y = max(pv_rest[current_h] - max(0, min(pcs - pre_pv2bess[current_h],
#                                                     capacity - current_soc - pre_pv2bess[current_h])), 0)
#             option3 = y * p_sell[current_h] + helper_algo(current_h - 1, current_soc + x)
#         # charge without overlap option
#         y = max(min(grid_size - pv_rest[current_h], pcs - pre_pv2bess[current_h],
#                     capacity - current_soc - pre_pv2bess[current_h]), 0)
#         option4 = pv_rest[current_h] * p_sell[current_h] - y * p_buy[current_h] + helper_algo(current_h - 1,
#                                                                                               min(current_soc + y +
#                                                                                                   pre_pv2bess[
#                                                                                                       current_h], 0))
#         # charge with overlap option
#         if y == 0 or y == capacity - current_soc - pre_pv2bess[current_h]:
#             option5 = -math.inf
#         else:
#             z = (pcs - y - pre_pv2bess[current_h]) / 2
#             option5 = (pv_rest[current_h] - z) * p_sell[current_h] - (y + z) * p_buy[current_h] + helper_algo(
#                 current_h - 1,
#                 min(current_soc + y + 2 * z, 0))
#         return max(option1, option2, option3, option4, option5)
#
#     return helper_algo(h, soc)


@njit
def calc_power_to_bess(sell_less_price, buy_less_price, hour_ordering, pv_quan, grid_quan, p, self_cons, charge_loss,
                       grid_bess_loss):
    # concatenate prices together and sort
    length = len(pv_quan)
    # if no power needed just return zeros
    if p <= 0:
        return np.zeros(length), np.zeros(length)
    quantities = np.concatenate((pv_quan, grid_quan))
    # concatenate quantities together and sort by prices
    sorted_quantities = quantities[hour_ordering]
    sell_less_price_sorted = sell_less_price[hour_ordering]
    # sum quantities related to the lowest prices until reaching p
    loss_factor = np.where(hour_ordering < length, (1 - charge_loss), (1 - grid_bess_loss))
    # calculate the self consumption used when charging. if pv is used first reduce self consumption from pv charge,
    # otherwise reduce it from grid charge (if there is overflow pv that charged the bess, self consumption is
    # reduced from it before)
    self_cons = np.concatenate((self_cons, self_cons))[hour_ordering]
    self_cons_hourly = np.where(((hour_ordering < length) & sell_less_price_sorted) |
                                ((hour_ordering >= length) & buy_less_price[hour_ordering]),
                                self_cons,
                                0)
    temp = np.maximum(sorted_quantities - self_cons_hourly, 0) * loss_factor
    quantities_sum = np.zeros((2 * length,))
    for i in range(1, 2 * length):
        quantities_sum[i] = sum(temp[:i])
    # calculate p updated after every charge
    updated_p = np.maximum(p - quantities_sum, 0)
    # calculate charge when splitting between pv and grid
    split_charge = np.where(sell_less_price_sorted,
                            0,
                            np.minimum(updated_p, sorted_quantities) / (2 - charge_loss -
                                                                        grid_bess_loss))
    # when buy price is less than sell prices, split the power from pv to have half send to grid and half to
    # battery, the other half is filled using the grid (this way pv power is not wasted)
    # calculate preliminary charge from pv and grid (before adding split charge to grid to bess charge)
    add_self_cons = np.where(updated_p > 0, self_cons_hourly, 0)
    prelim_charge = np.where(hour_ordering < length,
                             np.where(sell_less_price_sorted,
                                      np.minimum(updated_p / (1 - charge_loss) + add_self_cons,
                                                 sorted_quantities),
                                      split_charge),
                             np.minimum(updated_p / (1 - grid_bess_loss) + add_self_cons, sorted_quantities))
    # restored the original order of the quantities and add split charge to grid to bess when charge is split
    unsort_indices = np.zeros(2 * length, dtype=np.int64)
    unsort_indices[hour_ordering] = np.arange(2 * length)
    temp = np.zeros(2 * length)
    temp[:length] = split_charge[unsort_indices][:length]
    unsorted_quantities = prelim_charge[unsort_indices] + temp
    # return the quantities for charge
    return unsorted_quantities[:length], unsorted_quantities[length:]


@njit
def calc_daily_bess_power_flow(pre_pv2bess, pv_rest, grid2bess_no_overlap, max_discharge, p_sell, p_buy,
                               day_capacity, charge_loss, grid_bess_loss, active_self_cons):
    length = len(pre_pv2bess)
    total_soc = 0
    total_pv_charge = np.zeros(length)
    total_grid_charge = np.zeros(length)
    total_grid_discharge = np.zeros(length)
    # if no battery return empty charge/discharge schedule
    if day_capacity <= 0:
        return total_pv_charge, total_grid_charge, total_grid_discharge
    # get order of hours by prices and by quantities (giving priority for pv power when prices are equal and hours
    # were excess pv is used)
    quantities = [-q for q in np.concatenate((pre_pv2bess + pv_rest, grid2bess_no_overlap))]
    temp = np.where(pre_pv2bess > 0, 0.0, 1.0)
    excess_pv_used = np.concatenate((temp, temp))
    prices = np.concatenate((p_sell / (1 - charge_loss), p_buy / (1 - grid_bess_loss)))
    temp = np.array(list(zip(prices, [0.0] * length + [1.0] * length, excess_pv_used, quantities)),
                    dtype=[('value', float64), ('type', float64), ('prior', float64), ('quan', float64)])
    sorted_hours = temp.argsort(kind='mergesort', order=['value', 'type', 'prior', 'quan'])

    self_cons = np.full(length, active_self_cons)
    while total_soc < day_capacity:
        # calculate hours were pv prices are less (or grid available quantity is 0), and hour were grid prices are
        # less (or pv available is 0)
        temp = np.less_equal(p_sell, p_buy)
        sell_price_less = temp | (grid2bess_no_overlap <= 0)
        buy_price_less = ~temp | (pv_rest <= 0)
        # for each hour assume we want to discharge it this hour and find the best charge schedule
        best_discharge_profits = 0
        best_discharge_hour = -1
        free_power = np.zeros(length)
        best_pv_charge, best_grid_charge = None, None
        for j in range(length - 1, 0, -1):
            if max_discharge[j] == 0:
                continue
            # if can charge only from excess pv, there is are no costs (and no need to schedule charge by prices)
            free_power[j] = sum(np.maximum(pre_pv2bess[:j] - self_cons[:j], 0)) * (1 - charge_loss)
            self_cons_needed = np.where(pre_pv2bess[:j] > 0, 0, self_cons[:j])
            if free_power[j] > max_discharge[j]:
                discharge_profits = p_sell[j]
                pv_charge = None
            else:
                # use only hour before j
                sorted_hours_j = sorted_hours[sorted_hours % length < j]
                sorted_hours_j[sorted_hours_j >= length] -= length - j
                # schedule charge for the remaining power
                pv_charge, grid_charge = calc_power_to_bess(np.concatenate((sell_price_less,
                                                                            np.full(j, True))),
                                                            np.concatenate((np.full(j, False),
                                                                            buy_price_less)),
                                                            sorted_hours_j,
                                                            pv_rest[:j],
                                                            grid2bess_no_overlap[:j],
                                                            min(max_discharge[j], day_capacity - total_soc) -
                                                            free_power[j],
                                                            self_cons_needed,
                                                            charge_loss,
                                                            grid_bess_loss)
                # if charge scheduled is not enough to discharge the whole hour, only discharge the charged power
                total_discharge = (sum(np.maximum(pv_charge * (1 - charge_loss) +
                                                  grid_charge * (1 - grid_bess_loss)
                                                  - self_cons_needed, 0)) + free_power[j]) * \
                                  (1 - grid_bess_loss)
                if total_discharge == 0:
                    discharge_profits = 0
                else:
                    discharge_profits = round((total_discharge * p_sell[j] -
                                               np.dot(grid_charge, p_buy[:j])) / total_discharge, 4)
            if discharge_profits > best_discharge_profits or (discharge_profits == best_discharge_profits and
                                                              pv_rest[j] > 0):
                best_discharge_profits = discharge_profits
                best_discharge_hour = j
                if pv_charge is not None:
                    best_pv_charge = pv_charge
                    best_grid_charge = grid_charge

        # if there is no hour were we can profit from discharge, stop charging
        if best_discharge_hour == -1:
            break

        total_grid_discharge[best_discharge_hour] += min(max_discharge[best_discharge_hour],
                                                         day_capacity - total_soc)
        # first uses free power for excess pv
        total_charge = 0
        j = best_discharge_hour - 1
        discharge_amount = min(max_discharge[best_discharge_hour],
                               day_capacity - total_soc)
        if free_power[best_discharge_hour] > 0:
            while round(total_charge * (1 - charge_loss), 4) < round(discharge_amount, 4) and j >= 0:
                charge_amount = min(discharge_amount / (1 - charge_loss) - total_charge +
                                    active_self_cons,
                                    pre_pv2bess[j])
                if charge_amount > 0:
                    total_pv_charge[j] += charge_amount
                    total_charge += max(charge_amount - self_cons[j], 0)
                    self_cons[j] = 0
                    pre_pv2bess[j] -= charge_amount
                j -= 1
        # charge the rest by the schedule and reduce the charge amount from the available
        if round(total_charge * (1 - charge_loss), 4) < round(discharge_amount, 4):
            pv_rest[:best_discharge_hour] -= best_pv_charge[:best_discharge_hour]
            total_pv_charge[:best_discharge_hour] += best_pv_charge[:best_discharge_hour]
            grid2bess_no_overlap[:best_discharge_hour] -= best_grid_charge[:best_discharge_hour]
            total_grid_charge[:best_discharge_hour] += best_grid_charge[:best_discharge_hour]
            self_cons[:best_discharge_hour] = np.where(total_pv_charge[:best_discharge_hour] +
                                                       total_grid_charge[:best_discharge_hour] > 0,
                                                       0,
                                                       self_cons[:best_discharge_hour])

        # prevent charge of discharge again in the discharge hour chosen
        max_discharge[best_discharge_hour] = 0
        pv_rest[best_discharge_hour] = 0
        grid2bess_no_overlap[best_discharge_hour] = 0
        pre_pv2bess[best_discharge_hour] = 0

        # prevent discharge in the hours chosen to charge
        max_discharge[:best_discharge_hour] = np.where(total_pv_charge[:best_discharge_hour] +
                                                       total_grid_charge[:best_discharge_hour] > 0,
                                                       0,
                                                       max_discharge[:best_discharge_hour])

        # update the total discharge from the bess in this day
        total_soc += total_grid_discharge[best_discharge_hour]

    return total_pv_charge, total_grid_charge, total_grid_discharge


def calc_bess_power_flow(pre_pv2bess, pv_rest, grid2bess_no_overlap, max_discharge, p_sell, p_buy,
                         bess_capacity, charge_loss, grid_bess_loss, active_self_cons, indices):
    pv2bess = np.zeros(len(pre_pv2bess))
    grid2bess = np.zeros(len(pre_pv2bess))
    bess2grid = np.zeros(len(pre_pv2bess))
    insert_indices = np.insert(indices, 0, 0)
    insert_indices = np.append(insert_indices, len(pre_pv2bess))
    # calculate power flow for each day
    for i in range(len(p_sell)):
        pv2bess[insert_indices[i]: insert_indices[i + 1]], grid2bess[insert_indices[i]: insert_indices[i + 1]], \
            bess2grid[insert_indices[i]: insert_indices[i + 1]] = \
            calc_daily_bess_power_flow(pre_pv2bess[i], pv_rest[i], grid2bess_no_overlap[i], max_discharge[i],
                                       p_sell[i], p_buy[i], bess_capacity[24 * i], charge_loss, grid_bess_loss,
                                       active_self_cons[24 * i])

    return pv2bess, grid2bess, bess2grid


def build_tariff_table():
    """
    create a tariff table containing the tariff in each hour for each month
    """
    winter_months = [0, 1, 11]
    transition_months = [2, 3, 4, 9, 10]
    summer_months = [5, 6, 7, 8]
    # day of the week by datetime notation (sunday is 0)
    week_days = [0, 1, 2, 3, 4]
    weekend_days = [5, 6]
    winter_low_hours = list(range(1, 17)) + [22, 23, 0]
    winter_high_hours = list(range(17, 22))
    transition_low_hours = winter_low_hours
    transition_high_hours = winter_high_hours
    summer_low_hours = list(range(1, 17)) + [23, 0]
    summer_high_hours = list(range(17, 23))
    tariff_table = np.zeros((7, 12, 24))
    winter_low = 0.14 * 1.04
    winter_high_week = winter_high_weekend = 0.14 * 3.91
    transition_low = transition_high_weekend = 0.14 * 1
    transition_high_week = 0.14 * 1.2
    summer_low = summer_high_weekend = 0.14 * 1.22
    summer_high_week = 0.14 * 6.28
    # winter tariffs
    tariff_table[np.ix_(week_days, winter_months, winter_low_hours)] = winter_low
    tariff_table[np.ix_(weekend_days, winter_months, winter_low_hours)] = winter_low
    tariff_table[np.ix_(week_days, winter_months, winter_high_hours)] = \
        winter_high_week
    tariff_table[np.ix_(weekend_days, winter_months, winter_high_hours)] = \
        winter_high_weekend
    # transition tariffs
    tariff_table[np.ix_(week_days, transition_months, transition_low_hours)] = \
        transition_low
    tariff_table[np.ix_(weekend_days, transition_months, transition_low_hours)] = \
        transition_low
    tariff_table[np.ix_(week_days, transition_months, transition_high_hours)] = \
        transition_high_week
    tariff_table[np.ix_(weekend_days, transition_months, transition_high_hours)] = \
        transition_high_weekend
    # summer tariffs
    tariff_table[np.ix_(week_days, summer_months, summer_low_hours)] = summer_low
    tariff_table[np.ix_(weekend_days, summer_months, summer_low_hours)] = summer_low
    tariff_table[np.ix_(week_days, summer_months, summer_high_hours)] = \
        summer_high_week
    tariff_table[np.ix_(weekend_days, summer_months, summer_high_hours)] = \
        summer_high_weekend

    return tariff_table


def get_hourly_tariff(times, tariff_table):
    f = lambda x: tariff_table[(x.day_of_week + 1) % 7, x.month - 1, x.hour]
    hourly_tariff = f(times)
    return hourly_tariff


class OutputCalculatorTariff(OutputCalculator):

    def __init__(self, num_of_years: int,
                 grid_size: int,
                 producer: Producer,
                 power_storage: PowerStorage,
                 coupling: Coupling = Coupling.AC,
                 mvpv_loss: float = 0.005,
                 trans_loss: float = 0.015,
                 mvbat_loss: float = 0.005,
                 pcs_loss: float = 0.015,
                 dc_dc_loss: float = 0.01,
                 bess_discharge_start_hour: int = 17,
                 fill_battery_from_grid: bool = True,
                 save_all_results: bool = True,
                 producer_factor: float = 1,
                 tariff_table: np.ndarray = None):
        super().__init__(num_of_years, grid_size, producer, power_storage, coupling, mvpv_loss, trans_loss, mvbat_loss,
                         pcs_loss, dc_dc_loss, bess_discharge_start_hour, fill_battery_from_grid, save_all_results,
                         producer_factor)
        if tariff_table is not None:
            temp = np.zeros(7)
            self._tariff_table = tariff_table[None, ...] + temp[:, None, None]
        else:
            self._tariff_table = build_tariff_table()

    def _calc_power_to_bess(self, sell_less_price, buy_less_price, hour_ordering, pv_quan, grid_quan, p, self_cons):
        # concatenate prices together and sort
        length = len(pv_quan)
        # if no power needed just return zeros
        if p <= 0:
            return np.zeros(length), np.zeros(length)
        quantities = np.concatenate((pv_quan, grid_quan))
        # concatenate quantities together and sort by prices
        sorted_quantities = quantities[hour_ordering]
        sell_less_price_sorted = sell_less_price[hour_ordering]
        # sum quantities related to the lowest prices until reaching p
        loss_factor = np.where(hour_ordering < length, (1 - self._charge_loss), (1 - self._grid_bess_loss))
        reductions = np.column_stack((np.full(2 * length, 0), np.arange(2 * length))).ravel()
        # calculate the self consumption used when charging. if pv is used first reduce self consumption from pv charge,
        # otherwise reduce it from grid charge (if there is overflow pv that charged the bess, self consumption is
        # reduced from it before)
        self_cons = np.concatenate((self_cons, self_cons))[hour_ordering]
        self_cons_hourly = np.where(((hour_ordering < length) & sell_less_price_sorted) |
                                    ((hour_ordering >= length) & buy_less_price[hour_ordering]),
                                    self_cons,
                                    0)
        quantities_sum = np.add.reduceat(np.maximum(sorted_quantities - self_cons_hourly, 0) * loss_factor,
                                         reductions)[::2]
        # fix sum between same index not being 0
        quantities_sum[0] = 0
        # calculate p updated after every charge
        updated_p = np.maximum(p - quantities_sum, 0)
        # calculate charge when splitting between pv and grid
        split_charge = np.where(sell_less_price_sorted,
                                0,
                                np.minimum(updated_p, sorted_quantities) / (2 - self._charge_loss -
                                                                            self._grid_bess_loss))
        # when buy price is less than sell prices, split the power from pv to have half send to grid and half to
        # battery, the other half is filled using the grid (this way pv power is not wasted)
        # calculate preliminary charge from pv and grid (before adding split charge to grid to bess charge)
        add_self_cons = np.where(updated_p > 0, self_cons_hourly, 0)
        prelim_charge = np.where(hour_ordering < length,
                                 np.where(sell_less_price_sorted,
                                          np.minimum(updated_p / (1 - self._charge_loss) + add_self_cons,
                                                     sorted_quantities),
                                          split_charge),
                                 np.minimum(updated_p / (1 - self._grid_bess_loss) + add_self_cons, sorted_quantities))
        # restored the original order of the quantities and add split charge to grid to bess when charge is split
        unsort_indices = np.empty(2 * length, dtype=int)
        unsort_indices[hour_ordering] = np.arange(2 * length)
        unsorted_quantities = prelim_charge[unsort_indices] + np.concatenate((np.zeros(length),
                                                                              split_charge[unsort_indices][:length]))
        # return the quantities for charge
        return unsorted_quantities[:length], unsorted_quantities[length:]

    def _calc_daily_bess_power_flow(self, pre_pv2bess, pv_rest, grid2bess_no_overlap, max_discharge, p_sell, p_buy, day,
                                    year):
        day_capacity = self._df["battery_capacity"][24 * day + self._bess_discharge_start_hour]
        length = len(pre_pv2bess)
        total_soc = 0
        total_pv_charge = np.zeros(length)
        total_grid_charge = np.zeros(length)
        total_grid_discharge = np.zeros(length)
        # if no battery return empty charge/discharge schedule
        if day_capacity <= 0:
            return total_pv_charge, total_grid_charge, total_grid_discharge
        # get order of hours by prices and by quantities (giving priority for pv power when prices are equal and hours
        # were excess pv is used)
        quantities = [-q for q in np.concatenate((pre_pv2bess + pv_rest, grid2bess_no_overlap))]
        excess_pv_used = np.tile(~(pre_pv2bess > 0), 2)
        prices = np.concatenate((p_sell / (1 - self._charge_loss), p_buy / (1 - self._grid_bess_loss)))
        temp = np.array(list(zip(prices, [0] * length + [1] * length, excess_pv_used, quantities)),
                        dtype=[('value', 'f8'), ('type', 'i4'), ('prior', 'i4'), ('quan', 'f8')])
        sorted_hours = temp.argsort(kind='mergesort', order=['value', 'type', 'prior', 'quan'])

        self_cons = np.full(length, self._active_hourly_self_cons[24 * day])
        while total_soc < day_capacity:
            # calculate hours were pv prices are less (or grid available quantity is 0), and hour were grid prices are
            # less (or pv available is 0)
            temp = np.less_equal(p_sell, p_buy)
            sell_price_less = temp | (grid2bess_no_overlap <= 0)
            buy_price_less = ~temp | (pv_rest <= 0)
            # for each hour assume we want to discharge it this hour and find the best charge schedule
            best_discharge_profits = 0
            best_discharge_hour = -1
            free_power = np.zeros(length)
            best_pv_charge, best_grid_charge = None, None
            for j in range(length - 1, 0, -1):
                if max_discharge[j] == 0:
                    continue
                # if can charge only from excess pv, there is are no costs (and no need to schedule charge by prices)
                free_power[j] = sum(np.maximum(pre_pv2bess[:j] - self_cons[:j], 0)) * (1 - self._charge_loss)
                self_cons_needed = np.where(pre_pv2bess[:j] > 0, 0, self_cons[:j])
                if free_power[j] > max_discharge[j]:
                    discharge_profits = p_sell[j]
                    pv_charge = None
                else:
                    # use only hour before j
                    sorted_hours_j = sorted_hours[sorted_hours % length < j]
                    sorted_hours_j[sorted_hours_j >= length] -= length - j
                    # schedule charge for the remaining power
                    pv_charge, grid_charge = calc_power_to_bess(np.concatenate((sell_price_less,
                                                                                np.full(j, True))),
                                                                np.concatenate((np.full(j, False),
                                                                                buy_price_less)),
                                                                sorted_hours_j,
                                                                pv_rest[:j],
                                                                grid2bess_no_overlap[:j],
                                                                min(max_discharge[j], day_capacity - total_soc) -
                                                                free_power[j],
                                                                self_cons_needed,
                                                                self.charge_loss,
                                                                self.grid_bess_loss)
                    # if charge scheduled is not enough to discharge the whole hour, only discharge the charged power
                    total_discharge = (sum(np.maximum(pv_charge * (1 - self._charge_loss) +
                                                      grid_charge * (1 - self._grid_bess_loss)
                                                      - self_cons_needed, 0)) + free_power[j]) * \
                                      (1 - self._grid_bess_loss)
                    if total_discharge == 0:
                        discharge_profits = 0
                    else:
                        discharge_profits = round((total_discharge * p_sell[j] -
                                                   np.dot(grid_charge, p_buy[:j])) / total_discharge, 4)
                if discharge_profits > best_discharge_profits or (discharge_profits == best_discharge_profits and
                                                                  pv_rest[j] > 0):
                    best_discharge_profits = discharge_profits
                    best_discharge_hour = j
                    if pv_charge is not None:
                        best_pv_charge = pv_charge
                        best_grid_charge = grid_charge

            # if there is no hour were we can profit from discharge, stop charging
            if best_discharge_hour == -1:
                break

            total_grid_discharge[best_discharge_hour] += min(max_discharge[best_discharge_hour],
                                                             day_capacity - total_soc)
            # first uses free power for excess pv
            total_charge = 0
            j = best_discharge_hour - 1
            discharge_amount = min(max_discharge[best_discharge_hour],
                                   day_capacity - total_soc)
            if free_power[best_discharge_hour] > 0:
                while round(total_charge * (1 - self._charge_loss), 4) < round(discharge_amount, 4) and j >= 0:
                    charge_amount = min(discharge_amount / (1 - self._charge_loss) - total_charge +
                                        self._active_hourly_self_cons[24 * day],
                                        pre_pv2bess[j])
                    if charge_amount > 0:
                        total_pv_charge[j] += charge_amount
                        total_charge += max(charge_amount - self_cons[j], 0)
                        self_cons[j] = 0
                        pre_pv2bess[j] -= charge_amount
                    j -= 1
            # charge the rest by the schedule and reduce the charge amount from the available
            if round(total_charge * (1 - self._charge_loss), 4) < round(discharge_amount, 4):
                pv_rest[:best_discharge_hour] -= best_pv_charge[:best_discharge_hour]
                total_pv_charge[:best_discharge_hour] += best_pv_charge[:best_discharge_hour]
                grid2bess_no_overlap[:best_discharge_hour] -= best_grid_charge[:best_discharge_hour]
                total_grid_charge[:best_discharge_hour] += best_grid_charge[:best_discharge_hour]
                self_cons[:best_discharge_hour] = np.where(total_pv_charge[:best_discharge_hour] +
                                                           total_grid_charge[:best_discharge_hour] > 0,
                                                           0,
                                                           self_cons[:best_discharge_hour])

            # prevent charge of discharge again in the discharge hour chosen
            max_discharge[best_discharge_hour] = 0
            pv_rest[best_discharge_hour] = 0
            grid2bess_no_overlap[best_discharge_hour] = 0
            pre_pv2bess[best_discharge_hour] = 0

            # prevent discharge in the hours chosen to charge
            max_discharge[:best_discharge_hour] = np.where(total_pv_charge[:best_discharge_hour] +
                                                           total_grid_charge[:best_discharge_hour] > 0,
                                                           0,
                                                           max_discharge[:best_discharge_hour])

            # update the total discharge from the bess in this day
            total_soc += total_grid_discharge[best_discharge_hour]

        return total_pv_charge, total_grid_charge, total_grid_discharge

    def _calc_daily_bess_power_flow2(self, pre_pv2bess, pv_rest, grid2bess_no_overlap, max_discharge, p_sell, p_buy,
                                     day, year):
        day_capacity = self._df["battery_capacity"][24 * day + self._bess_discharge_start_hour]
        length = len(pre_pv2bess)
        total_soc = 0
        total_pv_charge = np.zeros(length)
        total_grid_charge = np.zeros(length)
        total_grid_discharge = np.zeros(length)
        # if no battery return empty charge/discharge schedule
        if day_capacity <= 0:
            return total_pv_charge, total_grid_charge, total_grid_discharge
        # get order of hours by prices and by quantities (giving priority for pv power when prices are equal and hours
        # were excess pv is used)
        quantities = [-q for q in np.concatenate((pre_pv2bess + pv_rest, grid2bess_no_overlap))]
        excess_pv_used = np.tile(~(pre_pv2bess > 0), 2)
        prices = np.concatenate((p_sell / (1 - self._charge_loss), p_buy / (1 - self._grid_bess_loss)))
        temp = np.array(list(zip(prices, [0] * length + [1] * length, excess_pv_used, quantities)),
                        dtype=[('value', 'f8'), ('type', 'i4'), ('prior', 'i4'), ('quan', 'f8')])
        sorted_hours = temp.argsort(kind='mergesort', order=['value', 'type', 'prior', 'quan'])

        self_cons = np.full(length, self._active_hourly_self_cons[24 * day])
        # calculate hours were pv prices are less (or grid available quantity is 0), and hour were grid prices are less
        # (or pv available is 0)
        temp = np.less_equal(p_sell, p_buy)
        sell_price_less = temp | (grid2bess_no_overlap <= 0)
        buy_price_less = ~temp | (pv_rest <= 0)
        # for each hour assume we want to discharge it this hour and find the best charge schedule
        best_discharge_profits = 0
        best_discharge_hour = -1
        free_power = np.zeros(length)
        best_pv_charge, best_grid_charge = None, None
        discharge_profits = np.zeros(length)
        # get value of each hour
        for j in range(length - 1, 0, -1):
            if max_discharge[j] == 0:
                continue
            # if can charge only from excess pv, there is are no costs (and no need to schedule charge by prices)
            free_power[j] = sum(np.maximum(pre_pv2bess[:j] - self_cons[:j], 0)) * (1 - self._charge_loss)
            self_cons_needed = np.where(pre_pv2bess[:j] > 0, 0, self_cons[:j])
            if free_power[j] > max_discharge[j]:
                discharge_profits[j] = p_sell[j]
                pv_charge = None
            else:
                # use only hour before j
                sorted_hours_j = sorted_hours[sorted_hours % length < j]
                sorted_hours_j[sorted_hours_j >= length] -= length - j
                # schedule charge for the remaining power
                pv_charge, grid_charge = self._calc_power_to_bess(np.concatenate((sell_price_less, np.full(j, True))),
                                                                  np.concatenate((np.full(j, False), buy_price_less)),
                                                                  sorted_hours_j,
                                                                  pv_rest[:j],
                                                                  grid2bess_no_overlap[:j],
                                                                  min(max_discharge[j] - free_power[j],
                                                                      day_capacity - total_soc),
                                                                  self_cons_needed)
                # if charge scheduled is not enough to discharge the whole hour, only discharge the charged power
                total_discharge = (sum(np.maximum(pv_charge * (1 - self._charge_loss) +
                                                  grid_charge * (1 - self._grid_bess_loss)
                                                  - self_cons_needed, 0)) + free_power[j]) * (1 - self._grid_bess_loss)
                if total_discharge == 0:
                    discharge_profits[j] = 0
                else:
                    discharge_profits[j] = round((total_discharge * p_sell[j] -
                                                  np.dot(grid_charge, p_buy[:j])) / total_discharge, 4)
            if discharge_profits[j] > best_discharge_profits:
                best_discharge_profits = discharge_profits[j]
                best_discharge_hour = j
                if pv_charge is not None:
                    best_pv_charge = pv_charge
                    best_grid_charge = grid_charge

        best_discharge_hours = discharge_profits.argsort()
        discharge_profits.sort()

        k = -2
        while total_soc < day_capacity:
            total_grid_discharge[best_discharge_hour] += min(max_discharge[best_discharge_hour],
                                                             day_capacity - total_soc)
            # first uses free power for excess pv
            total_charge = 0
            j = best_discharge_hour - 1
            discharge_amount = min(max_discharge[best_discharge_hour], day_capacity - total_soc)
            if free_power[best_discharge_hour] > 0:
                while round(total_charge * (1 - self._charge_loss), 4) < round(discharge_amount, 4) and j >= 0:
                    charge_amount = min(discharge_amount / (1 - self._charge_loss) - total_charge +
                                        self._active_hourly_self_cons[24 * day],
                                        pre_pv2bess[j])
                    if charge_amount > 0:
                        total_pv_charge[j] += charge_amount
                        total_charge += max(charge_amount - self_cons[j], 0)
                        self_cons[j] = 0
                        pre_pv2bess[j] -= charge_amount
                    j -= 1
            # charge the rest by the schedule and reduce the charge amount from the available
            if round(total_charge * (1 - self._charge_loss), 4) < round(discharge_amount, 4):
                pv_rest[:best_discharge_hour] -= best_pv_charge[:best_discharge_hour]
                total_pv_charge[:best_discharge_hour] += best_pv_charge[:best_discharge_hour]
                grid2bess_no_overlap[:best_discharge_hour] -= best_grid_charge[:best_discharge_hour]
                total_grid_charge[:best_discharge_hour] += best_grid_charge[:best_discharge_hour]
                self_cons[:best_discharge_hour] = np.where(total_pv_charge[:best_discharge_hour] +
                                                           total_grid_charge[:best_discharge_hour] > 0,
                                                           0,
                                                           self_cons[:best_discharge_hour])

            # prevent charge of discharge again in the discharge hour chosen
            pv_rest[best_discharge_hour] = 0
            grid2bess_no_overlap[best_discharge_hour] = 0
            pre_pv2bess[best_discharge_hour] = 0

            # prevent discharge in the hours chosen to charge
            max_discharge[:best_discharge_hour] = np.where(total_pv_charge[:best_discharge_hour] +
                                                           total_grid_charge[:best_discharge_hour] > 0,
                                                           0,
                                                           max_discharge[:best_discharge_hour])

            # update the total discharge from the bess in this day
            total_soc += total_grid_discharge[best_discharge_hour]

            # get next discharge hour
            if discharge_profits[k] <= 0 or total_soc >= day_capacity:
                break
            best_discharge_hour = best_discharge_hours[k]
            # if can charge only from excess pv, there is are no costs (and no need to schedule charge by prices)
            free_power[best_discharge_hour] = sum(np.maximum(pre_pv2bess[:best_discharge_hour] -
                                                             self_cons[:best_discharge_hour], 0)) * \
                                              (1 - self._charge_loss)
            self_cons_needed = np.where(pre_pv2bess[:best_discharge_hour] > 0, 0,
                                        self_cons[:best_discharge_hour])
            if free_power[best_discharge_hour] < max_discharge[best_discharge_hour]:
                # calculate hours were pv prices are less (or grid available quantity is 0), and hour were grid prices
                # are less (or pv available is 0)
                temp = np.less_equal(p_sell, p_buy)
                sell_price_less = temp | (grid2bess_no_overlap <= 0)
                buy_price_less = ~temp | (pv_rest <= 0)
                # use only hours before next best discharge hour
                sorted_hours_j = sorted_hours[sorted_hours % length < best_discharge_hour]
                sorted_hours_j[sorted_hours_j >= length] -= length - best_discharge_hour
                # schedule charge for the remaining power
                best_pv_charge, best_grid_charge = \
                    self._calc_power_to_bess(np.concatenate((sell_price_less, np.full(best_discharge_hour, True))),
                                             np.concatenate((np.full(best_discharge_hour, False), buy_price_less)),
                                             sorted_hours_j,
                                             pv_rest[:best_discharge_hour],
                                             grid2bess_no_overlap[:best_discharge_hour],
                                             min(max_discharge[best_discharge_hour] -
                                                 free_power[best_discharge_hour],
                                                 day_capacity - total_soc),
                                             self_cons_needed)
            k -= 1

        return total_pv_charge, total_grid_charge, total_grid_discharge

    def _calc_power_flow(self, year):
        hourly_tariff = get_hourly_tariff(self._df.index, self._tariff_table)
        pv_overflow = np.minimum(self._df["pv_output"] - self._grid_size / (1 - self._prod_trans_loss),
                                 self._pcs_power / (1 - self._charge_loss))
        self._df["grid2pv"] = np.where(self._df["pv_output"] > 0,
                                       0,
                                       -self._df["pv_output"] / (1 - self._prod_trans_loss))
        # calculate battery self consumption when pv sends power to it
        self._active_hourly_self_cons = self._df["battery_nameplate"] * self._power_storage.active_self_consumption
        pv2bess_pre = np.where(pv_overflow > self._active_hourly_self_cons, pv_overflow, 0)

        # get the maximum extra pv that can be delivered to bess (accounting for available pv and remaining pcs
        # capacity)
        max_extra_pv2bess = np.array(np.minimum(np.maximum(self._df["pv_output"], 0),
                                                self._pcs_power / (1 - self._charge_loss)) -
                                     pv2bess_pre)
        # don't use pv for battery charge if it's smaller than active self consumption
        max_extra_pv2bess = np.where(max_extra_pv2bess + pv2bess_pre > self._active_hourly_self_cons,
                                     max_extra_pv2bess,
                                     0)
        # if pv output is below active self consumption when battery is discharging, the pv energy will be used for
        # battery self consumption
        self._discharge_self_cons = np.where((0 < self._df["pv_output"]) &
                                             (self._df["pv_output"] <= self._active_hourly_self_cons),
                                             self._active_hourly_self_cons - self._df["pv_output"],
                                             self._active_hourly_self_cons)
        # calculate the maximum amount of discharge in each hour
        total_discharge_factor = (1 - self._grid_bess_loss) * self._power_storage.rte_table[year]
        max_discharge = np.where(pv_overflow > 0,
                                 0,
                                 np.maximum((self._grid_size - np.maximum(max_extra_pv2bess *
                                                                          (1 - self._prod_trans_loss), 0)) /
                                            total_discharge_factor, 0))
        max_discharge += np.where(max_discharge > 0, self._discharge_self_cons, 0)

        # the maximum amount of power that can be transmitted from the grid to the bess without blocking transmission
        # from pv to the grid
        max_grid2bess_no_overlap = np.maximum(np.minimum(self._grid_size - (max_extra_pv2bess + self._df["grid2pv"]) *
                                                         (1 - self._prod_trans_loss),
                                                         self._pcs_power - pv2bess_pre * (1 - self._charge_loss)), 0). \
            to_numpy()

        # split data to days
        buy_factor = 1
        pv_prices = np.round(hourly_tariff, 4)
        grid_prices = np.round(hourly_tariff * buy_factor, 4)
        charge_hour_indices = np.where(self._df.index.hour == 0)[0][1:]
        max_extra_pv2bess_split = np.split(max_extra_pv2bess, charge_hour_indices)
        sell_prices_split = np.split(pv_prices, charge_hour_indices)
        buy_prices_split = np.split(grid_prices, charge_hour_indices)
        pv2bess_pre_split = np.split(pv2bess_pre, charge_hour_indices)
        max_grid2bess_no_overlap_split = np.split(max_grid2bess_no_overlap, charge_hour_indices)
        max_discharge_split = np.split(max_discharge, charge_hour_indices)

        pv2bess = np.zeros(len(pv_overflow))
        grid2bess = np.zeros(len(pv_overflow))
        bess2grid = np.zeros(len(pv_overflow))
        insert_indices = np.insert(charge_hour_indices, 0, 0)
        insert_indices = np.append(insert_indices, len(pv_overflow))
        # calculate power flow for each day
        # for i in range(len(sell_prices_split)):
        #     pv2bess[insert_indices[i]: insert_indices[i + 1]], grid2bess[insert_indices[i]: insert_indices[i + 1]], \
        #         bess2grid[insert_indices[i]: insert_indices[i + 1]] = \
        #         self._calc_daily_bess_power_flow(pv2bess_pre_split[i], max_extra_pv2bess_split[i],
        #                                          max_grid2bess_no_overlap_split[i],
        #                                          max_discharge_split[i],
        #                                          sell_prices_split[i], buy_prices_split[i], i, year)

        pv2bess, grid2bess, bess2grid = calc_bess_power_flow(pv2bess_pre_split, max_extra_pv2bess_split,
                                                             max_grid2bess_no_overlap_split, max_discharge_split,
                                                             sell_prices_split, buy_prices_split,
                                                             self._df["battery_capacity"].to_numpy(), self._charge_loss,
                                                             self._grid_bess_loss, self._active_hourly_self_cons,
                                                             charge_hour_indices)

        self._df["pv2bess"] = pv2bess
        self._df["grid2bess"] = grid2bess
        self._df["bess2grid"] = bess2grid
        self._df["pv2grid"] = np.minimum(self._df["pv_output"] - self._df["pv2bess"],
                                         self._grid_size / (1 - self._prod_trans_loss))

        # if pv output is below active self consumption when battery is discharging, the pv energy will be used for
        # battery self consumption
        temp_condition = (0 < self._df["pv2grid"]) & (self._df["pv2grid"] < self._active_hourly_self_cons) & \
                         (self._df["bess2grid"] > 0)
        self._df["pv2grid"] = np.where(temp_condition, 0, self._df["pv2grid"])

    def _calc_neto_power(self, year):
        grid_from_pv = np.where(self._df["pv2grid"] > 0,
                                self._df["pv2grid"] * (1 - self._prod_trans_loss),
                                0)
        # reduce power by the losses for discharge and self consumption in discharge hours
        total_discharge_factor = (1 - self._grid_bess_loss) * self._power_storage.rte_table[
            year]
        grid_from_bess = np.where(self._df["bess2grid"] > self._discharge_self_cons,
                                  (self._df["bess2grid"] - self._discharge_self_cons) * total_discharge_factor,
                                  0)

        # reducing self consumption from pv to grid or add to grid to bess in charge and idle hours
        self._idle_hourly_self_cons = self._df["battery_nameplate"] * self._power_storage.idle_self_consumption
        active_hours = (self._df["bess2grid"] > 0) | (self._df["pv2bess"] > 0) | (self._df["grid2bess"] > 0)
        self._df["grid2bess"] += np.where(((grid_from_pv > self._idle_hourly_self_cons) & (active_hours == False)) |
                                          active_hours,
                                          0,
                                          self._idle_hourly_self_cons)
        grid_from_pv -= np.where((grid_from_pv > self._idle_hourly_self_cons) & (active_hours == False),
                                 self._idle_hourly_self_cons,
                                 0)

        if self._save_all_results:
            # calculate hourly soc (before adding idle self consumption from grid)
            self._df["grid_from_pv"] = grid_from_pv
            self._df["grid_from_bess"] = grid_from_bess

        # calculate hourly power to grid (pv + bess) (after reducing losses)
        self._df["output"] = grid_from_pv + grid_from_bess

    def run(self):
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
            if self._save_all_results:
                self._calc_overflow()
            self._calc_augmentations()
            self._calc_power_flow(year)
            self._calc_neto_power(year)
            self._purchased_from_grid.append(self._df["grid2bess"] + self._df["grid2pv"])
            if self._save_all_results:
                self._results.append(self._df.copy(deep=True))
            self._output.append(self._df["output"].copy(deep=True))


if __name__ == "__main__":
    # grid_size = 5000
    # pcs = 6265
    # capacity = 40000
    # charge_loss = 0.035
    # trans_loss = 0.024
    # grid_bess_loss = 0.04
    # active_self_cons = capacity * 0.004
    start_time = time.time()

    storage = LithiumPowerStorage(25, 180000,
                                  use_default_aug=True)  # aug_table=((12, 948), (84, 1102), (120, 1307), (156, 411), (204, 303),
    # (240, 24)))
    producer = PvProducer("../../../../test docs/Ramat Hovav.csv", pv_peak_power=300000)
    output = OutputCalculatorTariff(25, 180000, producer, storage)
    # cProfile.run('output.run()', sort='cumtime')
    output.run()

    test = FinancialCalculator(output, 2272, capex_per_land_unit=215000, capex_per_kwp=370, opex_per_kwp=5,
                               battery_capex_per_kwh=150, battery_opex_per_kwh=5, battery_connection_capex_per_kw=50,
                               battery_connection_opex_per_kw=0.5, fixed_capex=150000000, fixed_opex=10000000,
                               interest_rate=0.04, cpi=0.02)
    print("irr: ", test.get_irr())
    print("npv: ", test.get_npv(5))
    print("lcoe: ", test.get_lcoe())
    print("lcos: ", test.get_lcos())
    print("lcoe no grid power:", test.get_lcoe_no_power_costs())
    print(f"took {time.time() - start_time}")
