=======
History
=======


0.1.13 (2022-07-19)
-------------------

* fix a few bugs by avoiding `is_compatible_with` in `convert_unit`
* raise only `ValueError` instead of `DimensionalityError` on unit dimensionality mismatch
* remove `pandas` installation dependency because of transitive dependency via `pint-pandas`
* loosen requirements for `pytz` and `dateutil`, no special version requirements known
* extend `__getitem__` functionality by supporting iterables of timestamps or positional indices
* explicitly support indexing by time zone naive timestamps, which is deprecated by `pandas`
* make `coerce_unit` behave like `coerce_freq` and `coerce_time_zone` by passing through `None`


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
