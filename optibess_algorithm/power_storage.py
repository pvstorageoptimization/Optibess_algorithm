import warnings
from abc import ABC, abstractmethod
from math import floor
from typing import Any

import numpy as np

from . import constants


class PowerStorage(ABC):
    """
    Represent a power storage system
    """

    @property
    @abstractmethod
    def num_of_years(self) -> int:
        """
        the number of years the system will be used
        """
        pass

    @property
    @abstractmethod
    def degradation_table(self) -> tuple[float, ...]:
        """
        a tuple with yearly degradation of the system
        """
        pass

    @property
    @abstractmethod
    def dod_table(self) -> tuple[float, ...]:
        """
        a tuple with the yearly dod of the system
        """
        pass

    @property
    @abstractmethod
    def rte_table(self) -> tuple[float, ...]:
        """
        a tuple with the yearly rte of the system
        """
        pass

    @property
    @abstractmethod
    def number_of_blocks(self) -> int:
        """
        the number of storage blocks in the system
        """
        pass

    @property
    @abstractmethod
    def block_size(self) -> float:
        """
        the size of each block
        """
        pass

    @property
    @abstractmethod
    def idle_self_consumption(self) -> float:
        """
        the self consumption percentage for idle state
        """
        pass

    @property
    @abstractmethod
    def active_self_consumption(self) -> float:
        """
        the active self consumption percentage for active state
        """
        pass

    @property
    @abstractmethod
    def aug_table(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        a table with the augmentations of the storage system (month (stating at 0), number of block and storage
        size for each augmentation). None if augmentations are not used
        """
        pass

    @aug_table.setter
    def aug_table(self, value: tuple[tuple[int, int], ...]):
        self.set_aug_table(value)

    @abstractmethod
    def set_aug_table(self, value: tuple[tuple[int, int], ...], month_diff_constraint: bool = True):
        pass


class LithiumPowerStorage(PowerStorage):

    def __init__(self,
                 num_of_years: int,
                 connection_size: int,
                 degradation_table: tuple[float, ...] = constants.DEFAULT_DEG_TABLE,
                 dod_table: tuple[float, ...] = constants.DEFAULT_DOD_TABLE,
                 rte_table: tuple[float, ...] = constants.DEFAULT_RTE_TABLE,
                 block_size: float = constants.DEFAULT_BLOCK_SIZE,
                 pcs_loss: float = 0.015,
                 mvbat_loss: float = 0.01,
                 trans_loss: float = 0.01,
                 battery_hours: float = 5,
                 idle_self_consumption: float = constants.DEFAULT_IDLE_SELF_CONSUMPTION,
                 active_self_consumption: float = constants.DEFAULT_ACTIVE_SELF_CONSUMPTION,
                 aug_table: tuple[tuple[int, int], ...] | None = None,
                 use_default_aug: bool = False
                 ):
        """
        initialize the power storage
        :param num_of_years: number of years the battery is expected to work
        :param connection_size: the size of the connection to the battery (kW)
        :param degradation_table: a tuple containing the degradation level of the battery in each year
        :param dod_table: a tuple containing the dod of the battery in each year
        :param rte_table: a tuple containing the rte of the battery in each year
        :param block_size: size of each block of batteries (kwh)
        :param battery_hours: the number of hours the battery should supply (ignored if aug table is supplied)
        :param idle_self_consumption: factor of the battery size that is needed for self consumption when idle
        :param active_self_consumption: factor of the battery size that is needed for self consumption when active
        :param aug_table: table of augmentation for the storage
        :param use_default_aug: whether to use default augmentations when no aug_table is provided
        """
        self._set_num_of_years(num_of_years)
        self._set_connection_size(connection_size)
        self._connection_loss = 1 - ((1 - pcs_loss) * (1 - mvbat_loss) * (1 - trans_loss))
        self.degradation_table = degradation_table
        self.dod_table = dod_table
        self.rte_table = rte_table
        self._set_block_size(block_size)
        self.idle_self_consumption = idle_self_consumption
        self.active_self_consumption = active_self_consumption

        # augmentation parameters
        self._use_default_aug = use_default_aug
        if aug_table is None:
            self.battery_hours = battery_hours
        else:
            self.aug_table = aug_table

    # region Properties
    @property
    def num_of_years(self):
        return self._num_of_years

    def _set_num_of_years(self, value: int):
        if value <= 0:
            raise ValueError("Number of years should be positive")
        self._num_of_years = value

    @property
    def connection_size(self):
        """
        the size of the connection of the storage system to the rest of the power system (kW)
        """
        return self._connection_size

    def _set_connection_size(self, value):
        if value <= 0:
            raise ValueError("Battery connection size should be positive")
        self._connection_size = value

    @property
    def degradation_table(self):
        return self._degradation_table

    @degradation_table.setter
    def degradation_table(self, value: tuple[float, ...]):
        if len(value) < self._num_of_years + 1:
            raise ValueError(f"Battery deg table should have at least {self.num_of_years + 1} entries")
        for i in range(len(value)):
            if not 0 < value[i] <= 1:
                raise ValueError("Battery deg table should be between 0 (exclusive) and 1 (inclusive)")
            if i >= 1:
                if value[i] >= value[i - 1]:
                    raise ValueError("Battery deg table values should be strictly descending")
        self._degradation_table = value

    @property
    def dod_table(self):
        return self._dod_table

    @dod_table.setter
    def dod_table(self, value: tuple[float, ...]):
        if len(value) < self._num_of_years + 1:
            raise ValueError(f"Battery dod table should have at least {self.num_of_years + 1} entries")
        for i in range(len(value)):
            if not 0 < value[i] <= 1:
                raise ValueError("Battery dod table should be between 0 (exclusive) and 1 (inclusive)")
            if i >= 1:
                if value[i] > value[i - 1]:
                    raise ValueError("Battery dod table values should be descending")
        self._dod_table = value

    @property
    def rte_table(self):
        return self._rte_table

    @rte_table.setter
    def rte_table(self, value: tuple[float, ...]):
        if len(value) < self._num_of_years + 1:
            raise ValueError(f"Battery rte table should have at least {self.num_of_years + 1} entries")
        for i in range(len(value)):
            if not 0 < value[i] <= 1:
                raise ValueError("Battery rte table should be between 0 (exclusive) and 1 (inclusive)")
            if i >= 1:
                if value[i] > value[i - 1]:
                    raise ValueError("Battery rte table values should be descending")
        self._rte_table = value

    @property
    def block_size(self):
        return self._block_size

    def _set_block_size(self, value: float):
        if value <= 0:
            raise ValueError("Battery block size should be positive")
        self._block_size = value

    @property
    def idle_self_consumption(self):
        return self._idle_self_consumption

    @idle_self_consumption.setter
    def idle_self_consumption(self, value: float):
        if not 0 < value < 1:
            raise ValueError("Idle battery self consumption should be between 0 and 1 (exclusive)")
        self._idle_self_consumption = value

    @property
    def active_self_consumption(self):
        return self._active_self_consumption

    @active_self_consumption.setter
    def active_self_consumption(self, value: float):
        if not 0 < value < 1:
            raise ValueError("Active battery self consumption should be between 0 and 1 (exclusive)")
        self._active_self_consumption = value

    @property
    def number_of_blocks(self):
        return self._number_of_blocks

    @property
    def battery_hours(self):
        """
        the number of hours the system should supply (should supply connection size each hour)
        """
        return self._battery_hours

    @battery_hours.setter
    def battery_hours(self, new_value: int):
        """
        changes the number of hours the battery should supply (and variables effected by it). Overwrites current
        augmentation table
        :param new_value: the new value for the hours supplied by the battery
        """
        if not 0 <= new_value <= constants.MAX_BATTERY_HOURS:
            raise ValueError(f"Battery hours should be between 0 and {constants.MAX_BATTERY_HOURS}")
        self._battery_hours = new_value
        prelim_battery_bol = new_value * self._connection_size / (self._rte_table[0] * (1 - self._connection_loss) *
                                                                  self._dod_table[0] * self._degradation_table[0])
        # add self consumption for active hours + 1 (plus a little more for additional self consumption need for the
        # added amount of power for the self consumption
        self_consumption = prelim_battery_bol * self._active_self_consumption * 1.1 * (new_value + 1)
        self._battery_bol = prelim_battery_bol + self_consumption
        self._number_of_blocks = floor(self._battery_bol / self._block_size)
        self._battery_bol = self._number_of_blocks * self._block_size

        # change augmentation table
        if self._use_default_aug:
            self.aug_table = ((0, self._number_of_blocks),
                              (96, floor((1 - self._degradation_table[8]) * self._number_of_blocks)),
                              (192, floor((1 - self._degradation_table[16]) * self._number_of_blocks -
                                          self._degradation_table[8] * (1 - self._degradation_table[8]) *
                                          self._number_of_blocks)))
        else:
            self.aug_table = ((0, self._number_of_blocks),)

    @property
    def aug_table(self):
        return self._aug_table

    @aug_table.setter
    def aug_table(self, value: tuple[tuple[int, int], ...]):
        self.set_aug_table(value)

    def set_aug_table(self, value: tuple[tuple[int, int], ...], month_diff_constraint: bool = True):
        if len(value) < 1:
            raise ValueError("Augmentation table should have at least 1 entry")
        for i in range(len(value)):
            if len(value[i]) != 2:
                raise ValueError("Augmentation table entries should have 2 values")
            if value[i][0] < 0 or value[i][1] <= 0:
                raise ValueError("Augmentation table entries should have non negative and positive values")
            if i >= 1:
                # check distance between augmentations is at least 3 years
                if month_diff_constraint and value[i][0] - 36 < value[i - 1][0]:
                    raise ValueError("Augmentation table entries should have at least a 3 year gap")
        self._aug_table = np.array(value)
        # calculate each augmentation in kwh
        aug_cap = np.array([[aug[1] * self._block_size] for aug in self._aug_table])
        self._aug_table = np.concatenate((self._aug_table, aug_cap), axis=1)

        # check total battery size is not too big
        if not self.check_battery_size(value):
            warnings.warn(f"Storage size is bigger than {constants.MAX_BATTERY_HOURS} battery hours. "
                          f"This can cause charge and discharge to overlap and provide inaccurate results!",
                          UserWarning)

    def check_battery_size(self, value: tuple[tuple[int, int], ...]):
        """
        check if the battery size is bigger the maximum number of hours after each augmentation. return false if it is
        bigger.
        """
        for i in range(len(value)):
            battery_size = 0
            for j in range(i + 1):
                diff = (value[i][0] - value[j][0]) // 12
                battery_size += value[j][1] * self._block_size * self._degradation_table[diff] * \
                    self._dod_table[diff] * self._rte_table[diff] * (1 - self._connection_loss) * \
                    (1 - self._active_self_consumption * 1.1 * (constants.MAX_BATTERY_HOURS + 1))
            if battery_size > self._connection_size * constants.MAX_BATTERY_HOURS:
                return False
        return True
    # endregion
