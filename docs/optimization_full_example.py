from optibess_algorithm.constants import MAX_BATTERY_HOURS
from optibess_algorithm.financial_calculator import FinancialCalculator
from optibess_algorithm.output_calculator import OutputCalculator
from optibess_algorithm.power_storage import LithiumPowerStorage
from optibess_algorithm.producers import PvProducer
from optibess_algorithm.power_system_optimizer import NevergradOptimizer

import logging
import time
# make info logging show
logging.getLogger().setLevel(logging.INFO)
# setup power system
storage = LithiumPowerStorage(25, 5000, use_default_aug=True)
producer = PvProducer("test.csv", pv_peak_power=15000)
output = OutputCalculator(25, 5000, producer, storage, save_all_results=False)
finance = FinancialCalculator(output, 100)

# start optimization
start_time = time.time()
optimizer = NevergradOptimizer(financial_calculator=finance, use_memory=True, max_aug_num=6, initial_aug_num=3,
                               budget=2000)
opt_output, res = optimizer.run()
# print results
print(optimizer.get_candid(opt_output), res)
print(f"Optimization took {time.time() - start_time} seconds")
