import time
import pvlib
import pandas as pd
from enum import Enum, auto
from Optibess_algorithm.constants import *

root_folder = os.path.dirname(os.path.abspath(__file__))
PVGIS_URL = 'https://re.jrc.ec.europa.eu/api/v5_2/'
MODULE_DEFAULT = pvlib.pvsystem._parse_raw_sam_df(os.path.join(root_folder,
                                                               "sam-library-sandia-modules-2015-6-30.csv")) \
    ['Huasun_720']
INVERTER_DEFAULT = pvlib.pvsystem._parse_raw_sam_df(os.path.join(root_folder,
                                                                 "sam-library-cec-inverters-2019-03-05.csv")) \
    ['Huawei_Technologies_Co___Ltd___SUN2000_100KTL_USH0__800V_']


class Tech(Enum):
    FIXED = auto()
    TRACKER = auto()
    EAST_WEST = auto()


# TODO: change module and inverter to something immutable
def get_pvlib_output(latitude: float, longitude: float, tilt: float = TILT.default, azimuth: float = AZIMUTH.default,
                     tech: Tech = Tech.FIXED, modules_per_string: int = DEFAULT_MODULES_PER_STRING,
                     strings_per_inverter: int = DEFAULT_STRINGS_PER_INVERTER,
                     number_of_inverters: int = DEFAULT_NUMBER_OF_INVERTERS, module: pd.Series = MODULE_DEFAULT.copy(),
                     inverter: pd.Series = INVERTER_DEFAULT.copy()) -> pd.Series:
    """
    calculate the output of a pv system using pvlib
    :param latitude: the latitude of the location of the system
    :param longitude: the longitude of the location of the system
    :param tilt: the tilt of the system
    :param azimuth: the azimuth of the system (south is 0, values between -180 and 180)
    :param tech: the pv technology used (fixed, tracker or east-west)
    :param modules_per_string: number modules per string
    :param strings_per_inverter: number of string per inverter
    :param number_of_inverters: number of inverters in the system
    :param module: parameters of the pv module (as pandas series)
    :param inverter: parameters of the inverter (as pandas series)
    :returns:
        a pandas series with the hourly pv output of the system (with date and hour as index)
    """
    if number_of_inverters <= 0:
        raise ValueError("Number of units should be positive")

    # change azimuth to be in range used by pvlib
    if azimuth == 180:
        azimuth = 0
    else:
        azimuth = azimuth + 180

    # get tmy
    tmy = pvlib.iotools.get_pvgis_tmy(latitude=latitude, longitude=longitude, map_variables=True, url=PVGIS_URL)[0]

    # system definition
    location = pvlib.location.Location(latitude=latitude, longitude=longitude)
    temp_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

    # TODO: add check that tech is one of the options
    # use different system for fixed, tracker or east-west
    if tech == Tech.FIXED:
        mounts = [pvlib.pvsystem.FixedMount(surface_tilt=tilt, surface_azimuth=azimuth)]
    elif tech == Tech.TRACKER:
        mounts = [pvlib.pvsystem.SingleAxisTrackerMount(backtrack=False)]
    else:
        east_mount = pvlib.pvsystem.FixedMount(surface_tilt=tilt, surface_azimuth=90)
        west_mount = pvlib.pvsystem.FixedMount(surface_tilt=tilt, surface_azimuth=270)
        mounts = [east_mount, west_mount]

    arrays = [
        pvlib.pvsystem.Array(mount=mount, module_parameters=module,
                             temperature_model_parameters=temp_params,
                             modules_per_string=modules_per_string, strings=strings_per_inverter)
        for mount in mounts
    ]
    system = pvlib.pvsystem.PVSystem(arrays=arrays, inverter_parameters=inverter)

    # calculate the output
    model_chain = pvlib.modelchain.ModelChain(system, location)
    model_chain.run_model(tmy)
    # replace nan with 0 (for tracker option, since we get nan for hours without sun)
    output = model_chain.results.ac.fillna(0)
    # convert results from modelchain from W to kW
    return output / 1000 * (number_of_inverters if tech != Tech.EAST_WEST else number_of_inverters / 2)


def get_pvgis_hourly(latitude: float, longitude: float, tilt: float = TILT.default, azimuth: float = AZIMUTH.default,
                     pv_peak: float = DEFAULT_PV_PEAK_POWER, tech: Tech = Tech.FIXED, losses=DEFAULT_LOSSES):
    """
    get hourly data from pvgis
    :param latitude: latitude of the location
    :param longitude: longitude of the location
    :param tilt: the tilt of the pv array
    :param azimuth: the azimuth og the pv array (south is 0, values between -180 and 180)
    :param pv_peak: the peak power of the pv system
    :param tech: true if using tacker, false for fixed
    :param losses: the system estimate losses
    :returns:
        raw_data: a dataframe with hourly pv output (date and hour as index)
    """
    if tech == Tech.EAST_WEST:
        # get data for east and for west, each with half the peak power
        raw_data = pvlib.iotools.get_pvgis_hourly(latitude, longitude, pvcalculation=True,
                                                  surface_tilt=tilt, surface_azimuth=-90,
                                                  peakpower=pv_peak / 2, outputformat='csv',
                                                  trackingtype=0, loss=losses,
                                                  url=PVGIS_URL)[0] + \
                   pvlib.iotools.get_pvgis_hourly(latitude, longitude, pvcalculation=True,
                                                  surface_tilt=tilt, surface_azimuth=90,
                                                  peakpower=pv_peak / 2, outputformat='csv',
                                                  trackingtype=0, loss=losses,
                                                  url=PVGIS_URL)[0]
    else:
        raw_data = pvlib.iotools.get_pvgis_hourly(latitude, longitude, pvcalculation=True,
                                                  surface_tilt=0 if tech == Tech.TRACKER else tilt,
                                                  surface_azimuth=azimuth, peakpower=pv_peak, outputformat='csv',
                                                  trackingtype=1 if tech == Tech.TRACKER else 0, loss=losses,
                                                  url=PVGIS_URL)[0]
    # convert data from W to kW
    raw_data = raw_data.iloc[:, 0] / 1000
    # use the tmy data to select the "typical months" from the hourly data
    tmy_index = pvlib.iotools.get_pvgis_tmy(latitude, longitude, url=PVGIS_URL, map_variables=False)[0].index
    years = tmy_index[(tmy_index.day == 1) & (tmy_index.hour == 0)].year
    filtered_data = [raw_data[(raw_data.index.month == i + 1) & (raw_data.index.year == year)] for i, year in
                     enumerate(years)]
    return pd.concat(filtered_data)


if __name__ == '__main__':
    # start_time = time.time()
    # raw_data1 = get_pvgis_hourly(30, 34, pv_peak=10000)
    # print(raw_data1[raw_data1.index.month == 5])
    raw_data1 = get_pvlib_output(latitude=52.5, longitude=13, number_of_inverters=1000)
    print(raw_data1.head(24))
