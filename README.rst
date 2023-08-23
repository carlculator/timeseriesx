===========
TimeSeriesX
===========


.. image:: https://img.shields.io/pypi/v/timeseriesx.svg
        :target: https://pypi.python.org/pypi/timeseriesx
        :alt: Current pypi release

.. image:: https://img.shields.io/pypi/dm/timeseriesx.svg
        :target: https://pypi.python.org/pypi/timeseriesx
        :alt: Downloads per month

.. image:: https://img.shields.io/pypi/pyversions/timeseriesx.svg?label=Python%20Versions
        :target: https://pypi.python.org/pypi/timeseriesx
        :alt: Supported python versions

.. image:: https://travis-ci.com/carlculator/timeseriesx.svg?branch=develop
        :target: https://app.travis-ci.com/github/carlculator/timeseriesx
        :alt: build status

.. image:: https://coveralls.io/repos/github/carlculator/timeseriesx/badge.svg?branch=develop
        :target: https://coveralls.io/github/carlculator/timeseriesx
        :alt: test coverage

.. image:: https://readthedocs.org/projects/timeseriesx/badge/?version=latest
        :target: https://timeseriesx.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://pyup.io/repos/github/carlculator/timeseriesx/shield.svg
        :target: https://pyup.io/repos/github/carlculator/timeseriesx/
        :alt: Updates


The eXtended python time series library.

Manage time series data with explicit time zone, frequency and unit.


* Free software: MIT license
* Documentation: https://timeseriesx.readthedocs.io.

About
-----

TimeSeriesX is motivated by handling time series data in a convenient way. Almost all
the features are actually already provided by `pandas`_. TimeSeriesX extends the
pandas time series functionality by the unit functionalities of `pint`_ and
`pint-pandas`_. Further, TimeSeriesX offers an easy and convenient interface to work
with time series without the need to dig deep into these libraries, which nevertheless
is still recommended, since they go way beyond time series data.

The main challenges that arise when handling time series data are time zones and
frequencies. If you ever dealt with time series data from different sources in
different formats, you might know how much pain it can be to bring them all
together in a compatible yet readable way. Further, since time series data is
often obtained by measurements, the values are associated with units. Then these
units can be confused easily, since the units are often not modeled in code.

TimeSeriesX forces the user to handle time zones, frequencies and units explicitly,
while taking care of validation and convenient formats. It also supports deriving
these attributes from raw time series data. It offers a limited set of actions on
time series that are translated to pandas or pint functionality under the hood.
It was designed to guarantee that every transformation of time series data results in
a new valid time series, which would require quite some pandas code if done "manually".


Features
--------

* model time series data with explicit frequency, time zone and unit
* convert time zone or unit
* resample data to new frequency
* fill and get gaps
* perform calculations on time series with python standard operators


Installation
------------

.. code-block:: sh

    pip install timeseriesx


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
.. _`pandas`: https://pandas.pydata.org/
.. _`pint`: https://github.com/hgrecco/pint
.. _`pint-pandas`: https://github.com/hgrecco/pint-pandas
