import datetime
from abc import ABC, abstractmethod
import pandas as pd
from timezonefinder import TimezoneFinder

from Optibess_algorithm.constants import *
from Optibess_algorithm.pv_output_calculator import Tech, MODULE_DEFAULT, INVERTER_DEFAULT, get_pvlib_output, \
    get_pvgis_hourly


class Producer(ABC):
    """
    Represent a power producer
    """

    @property
    @abstractmethod
    def rated_power(self) -> int:
        """
        return the rated power of the producer
        """
        pass

    @property
    @abstractmethod
    def power_output(self) -> pd.DataFrame:
        """
        returns the power output of the producer as a pandas dataframe (index time, 1 column with power output)
        """
        pass

    @property
    @abstractmethod
    def power_cost(self) -> pd.DataFrame:
        """
        returns the time-depended cost of power as pandas dataframe (index time, 1 column with cost in shekels)
        """
        pass

    @property
    @abstractmethod
    def annual_deg(self):
        """
        returns the annual degradation of the system
        """
        pass


class PvProducer(Producer):
    """
    Represent a photovoltaic power producer
    """

    def __init__(self,
                 pv_output_file: str = None,
                 time_zone: str = DEFAULT_TIME_ZONE,
                 latitude: float = None,
                 longitude: float = None,
                 modules_per_string: int = DEFAULT_MODULES_PER_STRING,
                 strings_per_inverter: int = DEFAULT_STRINGS_PER_INVERTER,
                 number_of_inverters: int = None,
                 module: pd.Series = MODULE_DEFAULT,
                 inverter: pd.Series = INVERTER_DEFAULT,
                 tilt: float = TILT.default,
                 azimuth: float = AZIMUTH.default,
                 losses: float = DEFAULT_LOSSES,
                 pv_peak_power: float = None,
                 tech: Tech = Tech.FIXED,
                 annual_deg: float = 0.0035
                 ):
        """
        Initialize the PV producer and calculate the power output
        :param pv_output_file: a file name for hourly output of a pv system. Should contain 2 columns:
               date and hour, pv output
        :param time_zone: the time zone for the system (relevant only if file is supplied)
        :param latitude: latitude of location of the system (ignored if file is supplied)
        :param longitude: longitude of location of the system (ignored if file is supplied)
        :param modules_per_string: number of modules in each string (for pvlib option, ignored if file is supplied)
        :param strings_per_inverter: number of string per inverter (for pvlib option, ignored if file is supplied)
        :param number_of_inverters: number of inverters in the pv system (ignored if file is supplied. if provided, and
               no file is provided, uses pvlib)
        :param module: parameters of the pv module (as pandas series) (for pvlib option, ignored if file is supplied)
        :param inverter: parameters of the inverter (as pandas series) (for pvlib option, ignored if file is supplied)
        :param tilt: the tilt angle of th system (ignored if file is supplied)
        :param azimuth: the azimuth of the system (ignored if file is supplied)
        :param losses: the total losses of the array (ignored if file is supplied)
        :param pv_peak_power: the pv peak power of the array (for pvgis option, ignored if number_of_inverter and other
            pvlib parameters are provided, or file is provided)
        :param tech: the technology the pv system work works with (fixed, tracker or east-west) (ignored if file is
            supplied)
        :param annual_deg: factor for the annual degradation of the system
        """
        self._pv_output_file = pv_output_file
        self._time_zone = time_zone
        self._set_latitude(latitude)
        self._set_longitude(longitude)
        self._set_modules_per_string(modules_per_string)
        self._set_strings_per_inverter(strings_per_inverter)
        self._set_number_of_inverters(number_of_inverters)
        self._module = module
        self._inverter = inverter
        self._set_tilt(tilt)
        self._set_azimuth(azimuth)
        self._set_losses(losses)
        self._tech = tech
        self.annual_deg = annual_deg
        # check all parameters are present for the select pv output choice
        self._check_params_provided()
        self._set_pv_peak_power(pv_peak_power)
        self._calc_pv_output()

    def _check_params_provided(self):
        """
        check the relevant parameters were provided (or defaults are used) for pv output choice
        """
        if not self._pv_output_file:
            if self._number_of_inverters:
                if any(param is None for param in [self._latitude, self._longitude, self._tilt, self._azimuth,
                                                   self._tech, self._modules_per_string, self._strings_per_inverter,
                                                   self._module, self._inverter]):
                    raise ValueError("Missing values for parameters for pvlib option")
            else:
                if any(param is None for param in [self._latitude, self._longitude, self._tilt, self._azimuth,
                                                   self._losses]):
                    raise ValueError("Missing values for parameters for pvgis option")

    def _calc_pv_output(self):
        """
        get the pv output for the first year according to the pv output choice (file/pvgis/pvlib)
        """
        # file option
        if self._pv_output_file:
            if not self._pv_output_file.lower().endswith('.csv'):
                raise ValueError("PV file should be of type csv")
            # check data is numeric
            self._power_output = pd.DataFrame(pd.read_csv(self._pv_output_file, index_col=0).iloc[:, 0].astype(float))
            if self._power_output.shape[0] != 8760:
                raise ValueError("Number of lines in file should be dividable by number of hours in a year (8670)")
        else:
            # pvlib option
            if self._number_of_inverters:
                self._power_output = pd.DataFrame(get_pvlib_output(self._latitude, self._longitude, self._tilt,
                                                                   self._azimuth, self._tech, self._modules_per_string,
                                                                   self._strings_per_inverter,
                                                                   self._number_of_inverters, self._module,
                                                                   self._inverter))
            # pvgis option
            else:
                self._power_output = pd.DataFrame(get_pvgis_hourly(self._latitude, self._longitude, self._tilt,
                                                                   self._azimuth, self._pv_peak_power, self._tech,
                                                                   self._losses))
            tf = TimezoneFinder()
            self._time_zone = tf.timezone_at(lng=self._longitude, lat=self._latitude)
        year_one = datetime.datetime.today().year
        times = pd.date_range(start=f'{year_one}-01-01 00:00', end=f'{year_one}-12-31 23:00', freq='h',
                              tz=self._time_zone)
        self._power_output.index = times
        self._power_output.columns = ['pv_output']

        # create power cost dataframe with constant 0 cost
        self._power_cost = pd.DataFrame(0, index=times, columns=['power_cost'])

    # region Properties
    @property
    def pv_output_file(self):
        return self._pv_output_file

    @property
    def time_zone(self):
        return self._time_zone

    @property
    def latitude(self):
        return self._latitude

    def _set_latitude(self, value):
        if value is not None and not LATITUDE.min <= value <= LATITUDE.max:
            raise ValueError(f"Latitude value should be between {LATITUDE.min} and {LATITUDE.max}")
        self._latitude = value

    @property
    def longitude(self):
        return self._longitude

    def _set_longitude(self, value):
        if value is not None and not LONGITUDE.min <= value <= LONGITUDE.max:
            raise ValueError(f"Longitude value should be between {LONGITUDE.min} and {LONGITUDE.max}")
        self._longitude = value

    @property
    def modules_per_string(self):
        return self._modules_per_string

    def _set_modules_per_string(self, value):
        if value is not None and value <= 0:
            raise ValueError("Number of modules per string should be positive")
        self._modules_per_string = value

    @property
    def strings_per_inverter(self):
        return self._strings_per_inverter

    def _set_strings_per_inverter(self, value):
        if value is not None and value <= 0:
            raise ValueError("Number of strings per inverters should be positive")
        self._strings_per_inverter = value

    @property
    def number_of_inverters(self):
        return self._number_of_inverters

    def _set_number_of_inverters(self, value):
        if value is not None and value <= 0:
            raise ValueError("Number of inverters should be positive")
        self._number_of_inverters = value

    @property
    def module(self):
        return self._module

    @property
    def inverter(self):
        return self._inverter

    @property
    def tilt(self):
        return self._tilt

    def _set_tilt(self, value):
        if value is not None and not TILT.min <= value <= TILT.max:
            raise ValueError(f"Tilt should be between {TILT.min} and {TILT.max}")
        self._tilt = value

    @property
    def azimuth(self):
        return self._azimuth

    def _set_azimuth(self, value):
        if value is not None and not AZIMUTH.min <= value <= AZIMUTH.max:
            raise ValueError(f"Azimuth should be between {AZIMUTH.min} and {AZIMUTH.max}")
        self._azimuth = value

    @property
    def losses(self):
        return self._losses

    def _set_losses(self, value):
        if value is not None and value < 0:
            raise ValueError("Losses percentage should be non negative")
        self._losses = value

    @property
    def rated_power(self):
        return self._pv_peak_power

    def _set_pv_peak_power(self, value):
        # if the pvlib option is used calculate pv peak power based on module and number of modules
        if self._number_of_inverters:
            if 'STC' in self._module:
                module_peak_power = self._module['STC']
            else:
                module_peak_power = self._module['Impo'] * self._module['Vmpo']
            self._pv_peak_power = module_peak_power * self._modules_per_string * self._strings_per_inverter * \
                                  self._number_of_inverters / 1000
        else:
            if value is None:
                raise ValueError("PV peak power should have value for options other than pvlib")
            if value < 0:
                raise ValueError("PV peak power should be non negative")
            self._pv_peak_power = value

    @property
    def tech(self):
        return self._tech

    @property
    def power_output(self) -> pd.DataFrame:
        return self._power_output

    @property
    def power_cost(self) -> pd.DataFrame:
        return self._power_cost

    @property
    def annual_deg(self):
        return self._annual_deg

    @annual_deg.setter
    def annual_deg(self, value):
        if not 0 <= value < 1:
            raise ValueError("Annual degradation should be between 0 (inclusive) and 1 (exclusive)")
        self._annual_deg = value

    # endregion
