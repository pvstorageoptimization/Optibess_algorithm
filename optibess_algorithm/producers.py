import datetime
from abc import ABC, abstractmethod
import pandas as pd
from timezonefinder import TimezoneFinder

from . import constants
from .pv_output_calculator import Tech, MODULE_DEFAULT, INVERTER_DEFAULT, get_pvlib_output, get_pvgis_hourly


class Producer(ABC):
    """
    Represent a power producer
    """

    @property
    @abstractmethod
    def rated_power(self) -> int:
        """
        the rated power of the producer
        """
        pass

    @property
    @abstractmethod
    def power_output(self) -> pd.DataFrame:
        """
        the power output of the producer as a pandas dataframe (index time, 1 column with power output)
        """
        pass

    @property
    @abstractmethod
    def power_cost(self) -> pd.DataFrame:
        """
        the time-depended cost of power as pandas dataframe (index time, 1 column with cost in shekels)
        """
        pass

    @property
    @abstractmethod
    def annual_deg(self):
        """
        the annual degradation of the system
        """
        pass


class PvProducer(Producer):
    """
    Represent a photovoltaic power producer
    """

    def __init__(self,
                 pv_output_file: str | None = None,
                 time_zone: str = constants.DEFAULT_TIME_ZONE,
                 latitude: float | None = None,
                 longitude: float | None = None,
                 modules_per_string: int = constants.DEFAULT_MODULES_PER_STRING,
                 strings_per_inverter: int = constants.DEFAULT_STRINGS_PER_INVERTER,
                 number_of_inverters: int | None = None,
                 module: pd.Series = MODULE_DEFAULT,
                 inverter: pd.Series = INVERTER_DEFAULT,
                 use_bifacial: bool = constants.DEFAULT_USE_BIFACIAL,
                 albedo: float = constants.ALBEDO.default,
                 tilt: float = constants.TILT.default,
                 azimuth: float = constants.AZIMUTH.default,
                 losses: float = constants.DEFAULT_LOSSES,
                 pv_peak_power: float | None = None,
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
        :param use_bifacial: flag for bifacial calculation in pvlib
        :param albedo: ground albedo for bifacial calculation
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
        self._use_bifacial = use_bifacial
        self._set_albedo(albedo)
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
                                                   self._module, self._inverter, self._losses]):
                    raise ValueError("Missing values for parameters for pvlib option")
                if self._use_bifacial and self._albedo is None:
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
                                                                   self._inverter, self._use_bifacial, self._albedo,
                                                                   self._losses))
            # pvgis option
            else:
                self._power_output = pd.DataFrame(get_pvgis_hourly(self._latitude, self._longitude, self._tilt,
                                                                   self._azimuth, self._pv_peak_power, self._tech,
                                                                   self._losses))
            tf = TimezoneFinder()
            self._time_zone = tf.timezone_at(lng=self._longitude, lat=self._latitude)
        # take first 365 to deal with leap years
        year_one = datetime.datetime.today().year
        times = pd.date_range(start=f'{year_one}-01-01 00:00', end=f'{year_one}-12-31 23:00', freq='h',
                              tz=self._time_zone)[:8760]
        self._power_output.index = times
        self._power_output.columns = ['pv_output']

        # create power cost dataframe with constant 0 cost
        self._power_cost = pd.DataFrame(0, index=times, columns=['power_cost'])

    # region Properties
    @property
    def pv_output_file(self):
        """
        the name of the file containing the power output of the producer
        """
        return self._pv_output_file

    @property
    def time_zone(self):
        """
        the timezone in which the system is installed
        """
        return self._time_zone

    @property
    def latitude(self):
        """
        the latitude in which the system is installed
        """
        return self._latitude

    def _set_latitude(self, value):
        if value is not None and not constants.LATITUDE.min <= value <= constants.LATITUDE.max:
            raise ValueError(f"Latitude value should be between {constants.LATITUDE.min} and {constants.LATITUDE.max}")
        self._latitude = value

    @property
    def longitude(self):
        """
        the longitude in which the system is installed
        """
        return self._longitude

    def _set_longitude(self, value):
        if value is not None and not constants.LONGITUDE.min <= value <= constants.LONGITUDE.max:
            raise ValueError(f"Longitude value should be between {constants.LONGITUDE.min} and "
                             f"{constants.LONGITUDE.max}")
        self._longitude = value

    @property
    def modules_per_string(self):
        """
        the number of modules in each electronic string
        """
        return self._modules_per_string

    def _set_modules_per_string(self, value):
        if value is not None and value <= 0:
            raise ValueError("Number of modules per string should be positive")
        self._modules_per_string = value

    @property
    def strings_per_inverter(self):
        """
        the number of string connected to each inverter
        """
        return self._strings_per_inverter

    def _set_strings_per_inverter(self, value):
        if value is not None and value <= 0:
            raise ValueError("Number of strings per inverters should be positive")
        self._strings_per_inverter = value

    @property
    def number_of_inverters(self):
        """
        the number of inverters in the system
        """
        return self._number_of_inverters

    def _set_number_of_inverters(self, value):
        if value is not None and value <= 0:
            raise ValueError("Number of inverters should be positive")
        self._number_of_inverters = value

    @property
    def module(self):
        """
        a pandas series with the parameters for the modules
        """
        return self._module

    @property
    def inverter(self):
        """
        a pandas series with the parameters for the inverter
        """
        return self._inverter

    @property
    def albedo(self):
        """
        The fraction of sunlight diffusely reflected by the ground
        """
        return self._albedo

    def _set_albedo(self, value: float):
        if value is not None and not constants.ALBEDO.min <= value <= constants.ALBEDO.max:
            raise ValueError(f"Albedo should be between {constants.ALBEDO.min} and {constants.ALBEDO.max}")
        self._albedo = value

    @property
    def tilt(self):
        """
        the angle of the PV array from the horizon (in degrees)
        """
        return self._tilt

    def _set_tilt(self, value):
        if value is not None and not constants.TILT.min <= value <= constants.TILT.max:
            raise ValueError(f"Tilt should be between {constants.TILT.min} and {constants.TILT.max}")
        self._tilt = value

    @property
    def azimuth(self):
        """
        the angle from the true south to which the PV array is facing (in degrees) (0 is south)
        """
        return self._azimuth

    def _set_azimuth(self, value):
        if value is not None and not constants.AZIMUTH.min <= value <= constants.AZIMUTH.max:
            raise ValueError(f"Azimuth should be between {constants.AZIMUTH.min} and {constants.AZIMUTH.max}")
        self._azimuth = value

    @property
    def losses(self):
        """
        the total losses of the system
        """
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
        """
        the technology used for the PV tables
        """
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
