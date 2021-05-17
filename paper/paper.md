---
title: 'ddop: A python package for data-driven operations management'

tags:
  - Python
  - scikit-learn
  - data-driven operations management
  - newsvendor

authors:
  - name: Andreas Philippi
    orcid: 0000-0002-6508-9128
    affiliation: 1
  - name: Simone Buttler
    orcid: 0000-0003-3986-057X
    affiliation: 1
  - name: Nikolai Stein
    orcid: 0000-0001-9847-3444
    affiliation: 1

affiliations:
 - name: Chair of Logistics and Quantitative Methods, Julius-Maximilians-Universität Würzburg,  
         Sandering 2, Würzburg 97070, Germany
   index: 1

date: 26 April 2021

bibliography: paper.bib

---

# Background
In today's fast-paced world, companies often have to make important business decisions under uncertainty. For many of these decisions, e.g., deciding upon inventory or capacity levels, the most important source of uncertainty is customer demand. Final products are sold and distributed through different channels, and it is extremely difficult to predict the channels’ demands for individual products on individual days, weeks, or months, making it even more difficult to set the right inventory and capacity levels. On the upside, companies have unprecedented access to rich data sources that offer opportunities to reduce the degree of uncertainty and foster data-driven decision-making. For example, in inventory management the decision-maker may have access to historical demand data as well as additional side information, such as social media data, clickstreams, web searches, weather forecasts or calendar data. Turning such data into better decisions in terms of costs lies at the heart of what is called “data-driven operations management”.

# Summary

In the field of operations management, numerous data-driven approaches have recently been developed that combine machine learning methods and traditional optimization techniques to prescribe cost-optimal decisions directly from historical data. However, researchers and practitioners are missing efficient access to open-source implementations and standard benchmark datasets. In this paper, we present the open-source Python library *ddop* that integrates a wide range of data-driven tools for operations management tasks. Among other things, this includes well-established data-driven decision models such as weighted sample average approximation [@bertsimas2020predictive], empirical risk minimization [@ban2019big], and deep learning based approaches [@oroojlooyjadid2020applying] as well as a set of performance metrics. In addition, the library provides different real-world datasets that can be used to quickly illustrate the behavior of the different algorithms or as a benchmark for testing new models. The application programming interface (API) of *ddop* is designed to be consistent, easy-to-use, and accessible even for non-experts. With only a few lines of code, one can build and compare various models. In *ddop* all models are offered as objects implementing the estimator interface from scikit-learn [@buitinck2013api]. We thus not only provide a uniform API for our models, but also ensure that they safely interact with scikit-learn pipelines, model evaluation and selection tools.  

The library is distributed under the 3-Clause BSD license, encouraging its use in both academic and commercial settings. The full source code is available at https://github.com/opimwue/ddop. The package can be installed via the Python Package Index using `pip install ddop`.  A detailed documentation providing all information required to work with the API can be found at https://opimwue.github.io/ddop/. 

# Usage
Since all models in *ddop* implement the estimator interface from *scikit-learn* consisting of a *fit*, *predict*, and *score* method, usage follows the standard procedure of an *scikit-learn* regressor. First, a model is initialized by calling the class constructor from a given set of constant hyper-parameter values, each describing the model or the optimisation problem the estimator tries to solve. Note that for ease of use, all estimators use reasonable default values. It is therefore not necessary to pass any parameter to the constructor. However, it is recommended to tune them for the respective application, since this can often improve decision quality. After the model has been initialized, the *fit* method is used to learn a decision model from the training data (*X_train*, *y_train*). Once the training process is completed, the function returns the fitted model, which can then be used to make decisions for new data (*X_test*) by using the *predict* method. Finally, the score method can be used to access the decision quality of a model. The method takes as input *X_test* as well as the corresponding true values *y_test* and computes the average costs between *y_test* and *predict(X_test)*. Because all estimators follow the same interface, using a different model is as simple as replacing the constructor.

# References 
