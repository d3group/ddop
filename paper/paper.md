---
title: 'ddop: A python package for data-driven operationsmanagement'

tags:
  - Python
  - scikit-learn
  - data-driven operations management
  - newsvendor

authors:
  - name: Andreas Philippi
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Simone Buttler
    orcid: 0000-0000-0000-0001
    affiliation: 1

affiliations:
 - name: Chair of Logistics and Quantitative Methods, Julius-Maximilians-Universität Würzburg,  
         Sandering 2, Würzburg 97070, Germany
   index: 1

date: 26 April 2021

bibliography: paper.bib

---


# Background

Companies face considerable uncertainty when making important decisions in operations management, for example, when deciding upon inventory and capacities levels. One important source of uncertainty is, of course, customer demand. Final products are sold and distributed through different channels, and it is extremely difficult to predict the channels’ demands for individual products on individual days, weeks, or months, making it even more difficult to set the right inventory and capacity levels. The good news, however, is that companies have unprecedented access to rich data that can typically reduce the degree of focal uncertainty, leading to better decisions. For example, in inventory management, where the key uncertainty is in terms of demand volume, the decision-maker may have access to historical demand data as well as observable side information, such as social media data, clickstreams, web searches, weather forecasts or calendar data. Turning such data into better decisions in terms of costs lies at the heart of what is called “data-driven operations management”.

# Summary
In recent years, several data-driven approaches have been developed that combine machine learning methods and traditional optimization techniques to prescribe cost optimal decisions directly from historical data. However, what is missing is efficient access to open-source code and standard benchmark datasets. In this paper, we present the open-source Python library *ddop* that integrates a wide range of data-driven tools for operations management tasks. Among other things it includes well established decision models, a set of performance metrics, as well as datasets that are useful to quickly illustrate the behaviour of the different algorithms or as benchmark for testing new models. The application programming interface (API) of *ddop* is designed to be easy-to-use and accessible even for non-experts. With only a few lines of code one can build and compare various models. In *ddop* all models are offered as objects implementing the estimator interface from scikit-learn [@buitinck2013api] consisting of a *constructor*, and a *fit*, *predict*, and *score* method. Thus, usage follows the standard procedure used for *scikit-learn* regressors. By adopting this interface design, we not only provide a uniform API for our models, but also ensure that they safely interact with scikit-learn Pipelines and model evaluation and selection tools.  

The library is distributed under the 3-Clause BSD license, encouraging its use in both academic and commercial settings. The full source code is available at https://github.com/AndreasPhilippi/ddop. The package can be installed via the Python Package Index using `pip install ddop`.  A detailed documentation providing all information required to work  with the API can be found at https://andreasphilippi.github.io/ddop/. 


# References 