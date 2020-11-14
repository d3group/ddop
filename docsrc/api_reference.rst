.. _api_reference:

=============
API Reference
=============

.. autosummary::
    :toctree: modules/auto_generated/


.. _newsvendor_ref:

:mod:`ddop.newsvendor`: Newsvendor decision making
===================================================

The ``ddop.newsvendor`` module contains different newsvendor approaches for
decision making.

.. automodule:: ddop.newsvendor
    :no-members:
    :no-inherited-members:

.. currentmodule:: ddop

Empirical Risk Minimization
----------------------------

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    newsvendor.DeepLearningNewsvendor
    newsvendor.LightGradientBoostingNewsvendor
    newsvendor.EmpiricalRiskMinimizationNewsvendor
    newsvendor.RandomForestNewsvendor
    newsvendor.DecisionTreeNewsvendor


Weighted Optimization
----------------------

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    newsvendor.SampleAverageApproximationNewsvendor
    newsvendor.RandomForestWeightedNewsvendor
    newsvendor.KNeighborsWeightedNewsvendor
    newsvendor.GaussianWeightedNewsvendor


Parametric
-----------

.. autosummary::
    :toctree: modules/auto_generated/
    :template: class.rst

    newsvendor.ExponentialSmoothingNewsvendor


.. _metrics_ref:

:mod:`ddop.metrics`: Evaluation metrics
========================================

The ``ddop.metrics`` module contains different evaluation metrics

.. automodule:: ddop.metrics
    :no-members:
    :no-inherited-members:


.. currentmodule:: ddop.metrics

.. autosummary::
    :toctree: modules/auto_generated/

    calc_costs
    calc_total_costs
    calc_avg_costs


.. _datasets_ref:

:mod:`ddop.dataset`: Datasets
==============================

``ddop`` comes with a few default datasets that can be loaded using the ``ddop.datasets`` module.


.. automodule:: ddop.datasets
    :no-members:
    :no-inherited-members:

Loaders
-----------

.. currentmodule:: ddop.datasets

.. autosummary::
    :toctree: modules/auto_generated/

    load_yaz
    load_bakery

These datasets are useful to quickly illustrate the behavior of the various algorithms implemented in ddop.