---
title: 'ddop: A python package for data-driven operationsmanagement'

tags:
  - Python
  - data-driven operations management
  - newsvendor
 
authors:
  - name: Andreas Philippi
    orcid: ?
    affiliation: 1
  - name: Simone Buttler
    orcid: ?
    affiliation: 1

affiliations:
 - name:Chair of Logistics and Quantitative Methods, Julius-Maximilians-Universität Würzburg,     
        Sandering 2, Würzburg 97070, Germany
   index: 1

date: 26 April 2021
bibliography: paper.bib
---

# Summary
In the field of operations management research, several data-driven approaches have recently been developed that combine machine learning methods and traditional optimization techniques to prescribe decisions directly from historical demand data (without having to make assumptions about the underlying demand distribution). 

...

Motivated by the lack of available implementations in this field, we present the open source Python libray *ddop* that integrates a wide range of data-driven tools for operations management tasks. Among other things it includes well-established decision models, a set of performance metrics, as well as datasets that are useful to quickly illustrate the behavior of the different algorithms or as benchmark for testing new models. The application programming interface (API) of *ddop* is designed to be consistent, easy-to-use, and accessible even for non-experts. With only a few lines of code one can build and compare various decision models on different datasets. All models in *ddop* are offered as objects implementing the estimator interface from scikit-learn [@buitinck2013api]. We thus not only provide a uniform API, but also ensure that the models safely interact with scikit-learn Pipelines and model evaluation and selection tools.  
 
The library can be installed via the Python Package Index using `pip install ddop`. The library is distributed under the BSD license, encouraging
its use in both academic and commercial settings. The full source code is available at https://github.com/AndreasPhilippi/ddop. The documentation can be found at https://andreasphilippi.github.io/ddop/. 

# Acknowledgements

# References 