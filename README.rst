Optibess Algorithm - optimizing PV system combined with storage
===============================================================

**Optibess Algorithm** is a python 3.10+ library for simulating and optimizing a photovoltaic system with power storage.
It uses data from *pvgis* and algorithms from the *pvlib* and *Nevergrad* python libraries, and is the backend part of
the *Optibess* site.

.. image:: https://github.com/pvstorageoptimization/Optibess_algorithm/workflows/Tests/badge.svg
   :target: https://github.com/pvstorageoptimization/Optibess_algorithm/actions?query=workflow%3ATests

.. image:: https://codecov.io/gh/pvstorageoptimization/Optibess_algorithm/graph/badge.svg?token=L3VLK1Y1SM
    :target: https://codecov.io/gh/pvstorageoptimization/Optibess_algorithm

|Documentation Status|

.. |Documentation Status| image:: https://readthedocs.org/projects/optibess-algorithm/badge/?version=latest
   :target: http://optibess-algorithm.readthedocs.io/?badge=latest

|PyPI download total|

.. |PyPI download total| image:: https://img.shields.io/pypi/dt/optibess-algorithm.svg
   :target: https://pypi.org/project/optibess-algorithm/

Quick start
------------
**Optibess Algorithm** can be installed with:

.. code-block:: bash

    pip install Optibess_algorithm

You can run an optimization on an example system with:

.. code-block:: python

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

documentation
=============

Check out our `documentation <http://optibess-algorithm.readthedocs.io>`_. There are example of how to use the different modules for simulation and
optimization

License
=======

Optibess Algorithm is licensed under the `MIT license <LICENSE>`_

