Optibess Algorithm - optimizing PV system combined with storage
==============================================================

**Optibess Algorithm** is a python 3.10+ library for simulating and optimizing a photovoltaic system with power storage. 
It uses data from *pvgis* and algorithms from the *pvlib* and *Nevergrad* python libraries, and is the backend part of 
the *Optibess* site.

Quick start
------------
**Optibess Algorithm** can be installed with:

.. code-block:: bash

    pip install Optibess_algorithm

You can run an optimization on an example system with:

.. literalinclude:: optimization_example.py
    :language: python