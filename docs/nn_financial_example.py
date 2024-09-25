from optibess_algorithm.output_calculator import NNOutputCalculator
from optibess_algorithm.constants import *
from optibess_algorithm.producers import PvProducer
from optibess_algorithm.power_storage import LithiumPowerStorage
from optibess_algorithm.financial_calculator import FinancialCalculator

import time
import numpy as np

storage = LithiumPowerStorage(25, 5000, aug_table=((0, 83), (96, 16), (192, 16)))
producer = PvProducer("test.csv", pv_peak_power=15000)
prices = np.loadtxt("prices.csv")
output = NNOutputCalculator(num_of_years=25, grid_size=5000, producer=producer, power_storage=storage,
                            save_all_results=True, sell_prices=prices)
fc = FinancialCalculator(output_calculator=output, land_size=100, capex_per_land_unit=215000, capex_per_kwp=370,
                         opex_per_kwp=5, battery_capex_per_kwh=170, battery_opex_per_kwh=5,
                         battery_connection_capex_per_kw=50, battery_connection_opex_per_kw=0.5, fixed_capex=150000,
                         fixed_opex=10000, interest_rate=0.04, cpi=0.02, hourly_sell_prices=prices,
                         buy_from_grid_factor=1)
start_time = time.time()
output.run()
print("irr: ", fc.get_irr())
print("npv: ", fc.get_npv(5))
print("lcoe: ", fc.get_lcoe())
print("lcos: ", fc.get_lcos())
print("lcoe no grid power:", fc.get_lcoe_no_power_costs())
print(f"calculation took: {(time.time() - start_time)} seconds")