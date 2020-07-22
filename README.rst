Welcome to ddom-kit!
====================

.. image:: ../docsrc/logos/logo.png
    :width: 300

ddom-kit is a Python toolkit for data-driven operations management. The goal of ddom-kit is to provide access to
a variety of algorithms for decision making. Thus, one can run and compare the available approaches with only a few
lines of code. Moreover ddom-kit's interface allows to easily adapt an algorithm for one task to another.

Currently supported are different algorithms for newsvendor decision making.

------------------------------------------------------------

Installation
------------

ddom-kit is available via PyPI using:

.. code-block:: bash

    pip install ddop

The package is actively being developed and some features may
not be stable yet.

Quickstart
----------

Newsvendor decision making
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from ddop.datasets.load_datasets import load_data
    from ddop.newsvendor import EmpiricalRiskMinimizationNewsvendor
    from sklearn.model_selection import train_test_split
    data = load_data("yaz_steak.csv")
    X = data.iloc[:,0:24]
    Y = data.iloc[:,24]
    cu,co = 15,10
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, shuffle=False)
    mdl = EmpiricalRiskMinimizationNewsvendor(cu=15, co=10)
    mdl.fit(X_train, Y_train)
    mdl.score(X_test, Y_test)

For more details, check out our `E-Learning <https://andreasphilippi.github.io/ddom-kit/e_learning.html>`__.

------------------------------------------------------------

Documentation
-------------

* Read the `API reference <https://andreasphilippi.github.io/ddom-kit/api_reference.html>`__ for detailed class and function information.
* Check out our `E-Learning <https://andreasphilippi.github.io/ddom-kit/e_learning.html>`__.

------------------------------------------------------------
