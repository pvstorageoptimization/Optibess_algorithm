from optibess_algorithm.power_storage import LithiumPowerStorage

power_storage = LithiumPowerStorage(num_of_years=25, connection_size=5000, block_size=500, battery_hours=2,
                                    use_default_aug=True)
