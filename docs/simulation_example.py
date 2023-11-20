from Optibess_algorithm.output_calculator import OutputCalculator
from Optibess_algorithm.producers import PvProducer
from Optibess_algorithm.power_storage import LithiumPowerStorage

import numpy as np

power_storage = LithiumPowerStorage(num_of_years=25, connection_size=5000, block_size=500, battery_hours=2,
                                    use_default_aug=True)
prod = PvProducer("test.csv", pv_peak_power=13000)
output = OutputCalculator(num_of_years=25, grid_size=5000, producer=prod, power_storage=power_storage,
                          producer_factor=1, save_all_results=True)
# run simulation
output.run()

# change print options to show full rows of the matrix
np.set_printoptions(linewidth=1000)
print(output.monthly_averages())
