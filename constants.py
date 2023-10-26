import os
from collections import namedtuple
import numpy as np

# degradation of battery for each year
DEFAULT_DEG_TABLE = (1.0, 0.9244, 0.8974, 0.8771, 0.8602, 0.8446, 0.8321, 0.8191, 0.8059, 0.7928, 0.7796, 0.7664,
                     0.7533, 0.7402, 0.7271, 0.7141, 0.7010, 0.6879, 0.6748, 0.6618, 0.6487, 0.6356, 0.6225, 0.6094,
                     0.5963, 0.5832, 0.5701, 0.557, 0.5439, 0.5308, 0.5177)
# dod of battery for each year
DEFAULT_DOD_TABLE = (0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
                     0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
                     0.9, 0.9, 0.9, 0.9, 0.9)
# rte of battery for each year
DEFAULT_RTE_TABLE = (0.95, 0.95, 0.948, 0.948, 0.948, 0.945, 0.945, 0.945, 0.945, 0.943, 0.943, 0.943, 0.943, 0.94,
                     0.94, 0.94, 0.937, 0.937, 0.937, 0.937, 0.935, 0.935, 0.935, 0.935, 0.935, 0.935, 0.933, 0.933,
                     0.933, 0.933, 0.933)

root_folder = os.path.dirname(os.path.abspath(__file__))

EXAMPLE_TARIFFS = np.loadtxt(os.path.join(root_folder, "example_tariffs.csv"), delimiter=",", dtype=float)

# data structure for defualts
RangedValue = namedtuple('RangedValue', 'min max default')
CoordinateLimits = namedtuple('CoordinateLimits', 'min max')

# default values
LATITUDE = CoordinateLimits(-90, 90)
LONGITUDE = CoordinateLimits(-180, 180)
DEFAULT_NUM_OF_YEARS = 25
DEFAULT_GRID_SIZE = 5000
DEFAULT_LAND_SIZE = 97
DEFAULT_TIME_ZONE = "Asia/Jerusalem"
TILT = RangedValue(0, 90, 16)
AZIMUTH = RangedValue(-180, 180, 0)
DEFAULT_PV_PEAK_POWER = 225
DEFAULT_LOSSES = 9
DEFAULT_MODULES_PER_STRING = 1
DEFAULT_STRINGS_PER_INVERTER = 1
DEFAULT_NUMBER_OF_INVERTERS = 10000
DEFAULT_USE_BIFACIAL = False
ALBEDO = RangedValue(0, 1, 0.2)

DEFAULT_CAPEX_PER_LAND_UNIT = 20000
DEFAULT_OPEX_PER_LAND_UNIT = 5000
DEFAULT_CAPEX_PER_KWP = 420
DEFAULT_OPEX_PER_KWP = 1
DEFAULT_BATTERY_CAPEX_PER_KWH = 230
DEFAULT_BATTERY_OPEX_PER_KWH = 1
DEFAULT_BATTERY_CONNECTION_CAPEX_PER_KW = 50
DEFAULT_BATTERY_CONNECTION_OPEX_PER_KW = 1
DEFAULT_FIXED_CAPEX = 10000000
DEFAULT_FIXED_OPEX = 10000
DEFAULT_BATTERY_COST_DEG = 7
DEFAULT_INTEREST_RATE = 5
DEFAULT_CPI = 5

DEFAULT_IDLE_SELF_CONSUMPTION = 0.002
DEFAULT_ACTIVE_SELF_CONSUMPTION = 0.004
DEFAULT_BLOCK_SIZE = 372.736

DEFAULT_BASE_TARIFF = 0.14
DEFAULT_WINTER_LOW_FACTOR = 1.04
DEFAULT_WINTER_HIGH_FACTOR = 3.91
DEFAULT_TRANSITION_LOW_FACTOR = 1
DEFAULT_TRANSITION_HIGH_FACTOR = 1.2
DEFAULT_SUMMER_LOW_FACTOR = 1.22
DEFAULT_SUMMER_HIGH_FACTOR = 6.28
