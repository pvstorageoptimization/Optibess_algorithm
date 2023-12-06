from optibess_algorithm.output_calculator import OutputCalculator
from optibess_algorithm.constants import *
from optibess_algorithm.producers import PvProducer
from optibess_algorithm.power_storage import LithiumPowerStorage
from optibess_algorithm.financial_calculator import FinancialCalculator

import time

storage = LithiumPowerStorage(25, 5000, aug_table=((0, 83), (96, 16), (192, 16)))
producer = PvProducer("test.csv", pv_peak_power=15000)

output = OutputCalculator(25, 5000, producer, storage, save_all_results=True, fill_battery_from_grid=False,
                          bess_discharge_start_hour=17, producer_factor=1)
fc = FinancialCalculator(output_calculator=output, land_size=100, capex_per_land_unit=215000, capex_per_kwp=370,
                         opex_per_kwp=5, battery_capex_per_kwh=170, battery_opex_per_kwh=5,
                         battery_connection_capex_per_kw=50, battery_connection_opex_per_kw=0.5, fixed_capex=150000,
                         fixed_opex=10000, interest_rate=0.04, cpi=0.02, battery_cost_deg=0.07, base_tariff=0.14,
                         winter_low_factor=1.1, winter_high_factor=4, transition_low_factor=1,
                         transition_high_factor=1.2, summer_low_factor=1.2, summer_high_factor=6,
                         buy_from_grid_factor=1)
start_time = time.time()
output.run()
print("irr: ", fc.get_irr())
print("npv: ", fc.get_npv(5))
print("lcoe: ", fc.get_lcoe())
print("lcos: ", fc.get_lcos())
print("lcoe no grid power:", fc.get_lcoe_no_power_costs())
print(f"calculation took: {(time.time() - start_time)} seconds")
