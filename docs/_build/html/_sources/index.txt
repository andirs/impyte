Welcome to impyter
==================
Impyter is a Python module to impute missing values by prediction using machine learning algorithms.

Contents
========

.. toctree::
   :maxdepth: 3

Introduction
============
One essential problem for any person dealing with data is missing values.
There are several possibilities to deal with missing information, ranging
from dropping data points to estimating the value based on other values
in that column (i.e. average or median values).
A more recent method involves machine-learning algorithms.
This module offers a lightweight Python solution to calculate missing
information based on the underlying relationship between data points.

This is some additional cool text.::

    print 'Hello world'
    >> Hello world


Installation
============

Since this module is still in beta, you can install the latest version through its `github`_ repository via pip.

.. code-block:: python

   pip install git+git://github.com/andirs/impyter.git

Requirements
^^^^^^^^^^^^
The requirements are listed in ``requirements.txt`` and will usually be installed when
proceeding through pip. When installing manual,
please make sure following modules are already installed:

- `Python 3.6`_
- `sklearn 0.19`_
- `pandas 0.21`_

.. _github: https://github.com/andirs/impyter
.. _Python 3.6: https://www.python.org/
.. _sklearn 0.19: https://scikit-learn.org/
.. _pandas 0.21: http://pandas.pydata.org/

API Reference
=============
:mod:`.` Module

.. automodule:: impyte
    :members:
    :undoc-members:
    :special-members: __init__
    :show-inheritance:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

