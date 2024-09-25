from optibess_algorithm.output_calculator import NNOutputCalculator
from optibess_algorithm.producers import PvProducer
from optibess_algorithm.power_storage import LithiumPowerStorage

import numpy as np
import time

start_time = time.time()

power_storage = LithiumPowerStorage(num_of_years=25, connection_size=7000)
prod = PvProducer("test.csv", pv_peak_power=13000)
prices = np.loadtxt("prices.csv")
output = NNOutputCalculator(num_of_years=25, grid_size=7000, producer=prod, power_storage=power_storage,
                            save_all_results=True, sell_prices=prices)
# run simulation
output.run()

print(f" simulation took {time.time() - start_time} seconds")

# change print options to show full rows of the matrix
np.set_printoptions(linewidth=1000, precision=4)
print(output.monthly_averages())
