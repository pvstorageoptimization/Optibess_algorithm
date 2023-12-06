from optibess_algorithm.producers import PvProducer, Tech

prod = PvProducer(latitude=30.02, longitude=34.84, tilt=16, azimuth=0, tech=Tech.TRACKER, pv_peak_power=10000, losses=9)
