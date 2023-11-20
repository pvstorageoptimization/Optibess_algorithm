import logging
import time
from Optibess_algorithm.power_system_optimizer import NevergradOptimizer

# make info logging show
logging.getLogger().setLevel(logging.INFO)
# start optimization
start_time = time.time()
optimizer = NevergradOptimizer(budget=100)
opt_output, res = optimizer.run()
# print results
print(optimizer.get_candid(opt_output), res)
print(f"Optimization took {time.time() - start_time} seconds")
