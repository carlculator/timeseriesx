=======
History
=======


0.1.12 (2022-03-16)
-------------------

* fix equals method
* update documentation


0.1.11 (2022-03-07)
-------------------

* fix resampling method to support nan-values
* update dependencies


0.1.10 (2022-01-21)
-------------------

* fix equals method
* update dependencies


0.1.9 (2021-11-19)
------------------

* allow aggregation functions to return magnitudes or quantities
* update dependencies


0.1.8 (2021-09-28)
------------------

* fix time zone bug in gap handling
* update dependencies
* add more tests


0.1.7 (2021-09-28)
------------------

* improve gap handling
* update dependencies
* improve documentation
* fix calculations with quantity scalar


0.1.6 (2021-09-13)
------------------

* fix time zone issue with UTC in basic calculations for TimestampSeries as 2nd operand
* update pint-pandas version dependency
* use pint's default unit registry
* add support of callables as arguments for frequency resampling
* add more tests


0.1.5 (2021-09-10)
------------------

* fix time zone issue with UTC in basic calculations
* add round-method for TimestampSeries
* fix map-function for series with unit
* add more tests


0.1.4 (2021-09-09)
------------------

* improve test coverage
* improve TimeSeries equality check
* support NaN-removal in as_pd_series-method


0.1.3 (2021-09-08)
------------------

* remove manual timezone checks because it is handled by pandas
* fix skipped tests
* fix repr() method of TimestampSeries
* fix basic calculation with units involved


0.1.2 (2021-09-07)
------------------

* fix timezone handling
* First release on PyPI Index.



0.1.1 (2021-02-16)
------------------

* First release on PyPI Test Index.
