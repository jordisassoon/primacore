.. primacore documentation master file

====================================
🌿 primacore Documentation
====================================

PRIMA is a tool for Palaeoclimatic Reconstruction through Interactive Modelling & Analysis. 
This package provides multiple methods for quantitative palaeoclimate reconstruction from fossil pollen data.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   README
   CONTRIBUTING
   API Documentation <api/index>

What is primacore?
==================

primacore implements several statistical and machine learning methods for climate reconstruction:

* **Modern Analogue Technique (MAT)** - A non-parametric method for paleoclimate reconstruction
* **Boosted Regression Trees (BRT)** - Gradient boosting for improved predictions
* **Weighted Averaging Partial Least Squares (WA-PLS)** - Multivariate statistical calibration
* **Random Forest (RF)** - Ensemble learning approach

Installation
============

Install the package using pip:

.. code-block:: bash

   pip install primacore

Or with uv:

.. code-block:: bash

   uv pip install primacore

Quick Start
===========

.. code-block:: python

   from primacore.models import MAT, BRT, RF
   import pandas as pd

   # Load your data
   modern_data = pd.read_csv('modern_pollen.csv')
   
   # Create and train a model
   model = MAT(k=5)
   model.fit(modern_data, climate_data)
   
   # Make predictions
   reconstructions = model.predict(fossil_data)

API Reference
=============

.. autosummary::
   :toctree: api/generated
   :recursive:

   primacore.models

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
