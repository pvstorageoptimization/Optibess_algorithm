from optibess_algorithm.pv_output_calculator import MODULE_DEFAULT, INVERTER_DEFAULT
from optibess_algorithm.producers import PvProducer, Tech

prod = PvProducer(latitude=30.02, longitude=34.84, tilt=16, azimuth=0, tech=Tech.EAST_WEST, modules_per_string=10,
                  strings_per_inverter=2, number_of_inverters=1000, module=MODULE_DEFAULT, inverter=INVERTER_DEFAULT,
                  use_bifacial=True, albedo=0.2)
