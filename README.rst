.. -*- mode: rst -*-

.. image:: https://travis-ci.com/opimwue/ddop.svg?branch=master
    :target: https://travis-ci.com/opimwue/ddop

.. image:: https://d25lcipzij17d.cloudfront.net/badge.svg?id=py&type=6&v=0.6.5&x2=0
    :target: https://badge.fury.io/py/ddop

.. image:: https://img.shields.io/github/license/andreasphilippi/ddop
    :target: https://github.com/andreasphilippi/ddop/blob/master/LICENSE
    
.. image:: https://www.code-inspector.com/project/22456/status/svg
    :target: https://frontend.code-inspector.com/public/project/22456/ddop/dashboard
    
.. image:: https://joss.theoj.org/papers/0de119f95840b69fcea94309c18058e4/status.svg
    :target: https://joss.theoj.org/papers/0de119f95840b69fcea94309c18058e4   
    

----------------------


Welcome to ddop!
====================

.. image:: /docsrc/logos/logo.png
    :width: 300

``ddop`` is a Python library for data-driven operations management. The goal of ``ddop`` is to provide well-established
data-driven operations management tools within a programming environment that is accessible and easy to use even
for non-experts. At the current state ``ddop`` contains well known data-driven newsvendor models, a set of
performance metrics that can be used for model evaluation and selection, as well as datasets that are useful to
quickly illustrate the behavior of the various algorithms implemented in ``ddop`` or as benchmark for testing new
models. Through its consistent and easy-to-use interface one can run and compare provided models with only a few
lines of code.

------------------------------------------------------------

Installation
------------

ddop is available via PyPI using:

.. code-block:: bash

    pip install ddop

The installation requires the following dependencies:

- numpy==1.18.2
- scipy==1.4.1
- pandas==1.1.4
- statsmodels==0.12.1
- scikit-learn==0.23.0
- tensorflow==2.1.0
- keras==2.3.1
- pulp==2.0
- lightgbm==2.3.1

Note: The package is actively developed and conflicts with other packages may occur during
installation. To avoid any installation conflicts we therefore recommend to install the
package in an empty environment with the above mentioned dependencies

Quickstart
----------
``ddop`` provides a varity of newsvendor models. The following example
shows how to use one of these models for decision making. It assumes
a very basic knowledge of data-driven operations management practices.

As first step we initialize the model we want to use. In this example
`LinearRegressionNewsvendor <https://opimwue.github.io/ddop/modules/auto_generated/ddop.newsvendor.LinearRegressionNewsvendor.html#ddop.newsvendor.LinearRegressionNewsvendor>`__.

.. code-block:: python

    >>> from ddop.newsvendor import LinearRegressionNewsvendor
    >>> mdl = LinearRegressionNewsvendor(cu=2,co=1)

A model can take a set of parameters, each describing the model or the optimization
problem it tries to solve. Here we set the underage costs ``cu`` to 2 and
the overage costs ``co`` to 1.

As next step we load the `Yaz Dataset <https://opimwue.github.io/ddop/modules/auto_generated/ddop.datasets.load_yaz.html#ddop.datasets.load_yaz>`__ and split it into train and test set.

.. code-block:: python

    >>> from ddop.datasets import load_yaz
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = load_yaz(one_hot_encoding=True, return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, random_state=0)

After the model is initialized, the ``fit`` method can be used to learn a decision model from the training data ``X_train``, ``y_train``.

.. code-block:: python

    >>> mdl.fit(X_train, y_train)

We can then use the ``predict`` method to make a decision for new data samples.

.. code-block:: python

    >>> mdl.predict(X_test)
    >>> array([[ 8.32..,  7.34.., 16.92.., ..]])

To get a representation of the model's decision quality we can use the ``score`` function, which takes as input
``X_test`` and  ``y_test``. The score function makes a decision for each sample in ``X_test`` and calculates
the negated average costs with respect to the true values ``y_test`` and the overage and underage costs.

.. code-block:: python

    >>> mdl.score(X_test,y_test)
    -7.05..

------------------------------------------------------------

See also
-----------
* Follow the `API reference <https://opimwue.github.io/ddop/api_reference.html>`__ to get an overview of available functionalities and for detailed class and function information.
* To get familiar with ``ddop`` and to learn more about data-driven operations management check out our `Tutorials <https://opimwue.github.io/ddop/tutorial.html>`__.

------------------------------------------------------------
