---
title: 'scikit-hubness: Hubness Reduction and Approximate Neighbor Search'

tags:
  - Python
  - scikit-learn
  - hubness
  - curse of dimensionality
  - nearest neighbors

authors:
  - name: Roman Feldbauer
    orcid: 0000-0003-2216-4295
    affiliation: 1
  - name: Thomas Rattei
    orcid: 0000-0002-0592-7791
    affiliation: 1
  - name: Arthur Flexer
    orcid: 0000-0002-1691-737X
    affiliation: 2

affiliations:
 - name: Division of Computational Systems Biology, Department of Microbiology and Ecosystem Science,
         University of Vienna, Althanstra&szlig;e 14, 1090 Vienna, Austria
   index: 1
 - name: Austrian Research Institute for Artificial Intelligence (OFAI),
         Freyung 6/6/7, 1010 Vienna, Austria
   index: 2

date: 09 December 2019

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