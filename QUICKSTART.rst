Quickstart
==========

.. contents::
    :local:
    :depth: 2

Create a ``TimestampSeries``-object
-----------------------------------

.. code-block:: python

    import datetime as dt
    import numpy as np
    import pandas as pd
    import timeseriesx as tx
    import pint


Create from lists
^^^^^^^^^^^^^^^^^

.. code-block:: python

    timestamps = [
        dt.datetime(2020, 3, 1, 15, 0, 0),
        dt.datetime(2020, 3, 1, 16, 0, 0),
        dt.datetime(2020, 3, 1, 17, 0, 0),
    ]
    values = [0., 1., 2.]
    ts = tx.TimestampSeries.create_from_lists(
        timestamps, values, freq='H', unit='km', time_zone='Europe/Vienna'
    )

Create from dictionary
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    items = {
        dt.datetime(2020, 3, 1, 15, 0, 0): 0.,
        dt.datetime(2020, 3, 1, 16, 0, 0): 1.,
        dt.datetime(2020, 3, 1, 17, 0, 0): 2.,
    }
    ts = tx.TimestampSeries.create_from_dict(
        items, freq='H', unit='km', time_zone='Europe/Vienna'
    )


Create from tuples
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    items = [
        (dt.datetime(2020, 3, 1, 15, 0, 0), 0.),
        (dt.datetime(2020, 3, 1, 16, 0, 0), 1.),
        (dt.datetime(2020, 3, 1, 17, 0, 0), 2.),
    ]
    ts = tx.TimestampSeries.create_from_tuples(
        items, freq='H', unit='km', time_zone='Europe/Vienna'
    )

Create from pandas.Series
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    series = pd.Series(
        data=[0., 1., 2],
        index=pd.DatetimeIndex([
            dt.datetime(2020, 3, 1, 15, 0, 0),
            dt.datetime(2020, 3, 1, 16, 0, 0),
            dt.datetime(2020, 3, 1, 17, 0, 0),
        ])
    )
    ts = tx.TimestampSeries.create_from_pd_series(
        items, freq='H', unit='km', time_zone='Europe/Vienna'
    )


Create constant series from start to end
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    ts = tx.TimestampSeries.create_constant_series(
        start=dt.datetime(2020, 3, 1, 15, 0, 0),
        end=dt.datetime(2020, 3, 1, 17, 0, 0),
        value=1.,
        freq='H', unit='km', time_zone='Europe/Vienna'
    )

.. hint::

    * param ``freq`` also supports types: ``datetime.timedelta``, ``pandas.Offset``,
      ``pandas.Timedelta``

    * param ``time_zone`` also support types: ``tzinfo``-objects (such as ``datetime.timezone``,
      pytz-, or dateutil-timezone-objects)

    * param ``unit`` also supports type ``pint.Unit``


Infer attributes from raw data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    DATE_FORMAT = '%Y-%m-%dT%H:%M:%S%z'

    items = {
        dt.datetime.strptime('2020-03-01T15:00:00+0000', DATE_FORMAT): 0.,
        dt.datetime.strptime('2020-03-01T16:00:00+0000', DATE_FORMAT): 1.,
        dt.datetime.strptime('2020-03-01T17:00:00+0000', DATE_FORMAT): 2.,
    }
    ts = tx.TimestampSeries.create_from_dict(
        items, freq='infer', unit='km', time_zone='infer'
    )


.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> ts.freq
    <Hour>

.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> ts.time_zone
    datetime.datetime.utc

Calculations
------------

.. hint::

    Basic arithmetic operations such as ``+``, ``-``, ``*``, ``/``, ``//``, ``%`` and ``**`` are
    supported via standard python syntax.

.. code-block:: python

    items = [
        (dt.datetime(2020, 3, 1, 15, 0, 0), 0.),
        (dt.datetime(2020, 3, 1, 16, 0, 0), 1.),
        (dt.datetime(2020, 3, 1, 17, 0, 0), 2.),
    ]
    ts = tx.TimestampSeries.create_from_tuples(
        items, freq='H', unit='km', time_zone='Europe/Vienna'
    )


``TimestampSeries`` & ``TimestampSeries``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> print(ts + ts)
    TIME ZONE: Europe/Vienna *** FREQ: H *** UNIT: kilometer
    2020-03-01 15:00:00+01:00    0.0
    2020-03-01 16:00:00+01:00    2.0
    2020-03-01 17:00:00+01:00    4.0
    dtype: pint[kilometer]


Time index differences are handled implicitly
"""""""""""""""""""""""""""""""""""""""""""""

.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> print(ts - ts[:-1])
    TIME ZONE: Europe/Vienna *** FREQ: H *** UNIT: kilometer
    2020-03-01 15:00:00+01:00    0.0
    2020-03-01 16:00:00+01:00    0.0
    2020-03-01 17:00:00+01:00    2.0
    dtype: pint[kilometer]


``TimestampSeries`` & Collection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> print(ts * [1., 2., 3.])
    TIME ZONE: Europe/Vienna *** FREQ: H *** UNIT: kilometer
    2020-03-01 15:00:00+01:00    0.0
    2020-03-01 16:00:00+01:00    2.0
    2020-03-01 17:00:00+01:00    6.0
    dtype: pint[kilometer]

``TimestampSeries`` & Scalar
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> print(ts / 2)
    TIME ZONE: Europe/Vienna *** FREQ: H *** UNIT: kilometer
    2020-03-01 15:00:00+01:00    0.0
    2020-03-01 16:00:00+01:00    0.5
    2020-03-01 17:00:00+01:00    1.0
    dtype: pint[kilometer]


Conversions
-----------

Convert time zone
^^^^^^^^^^^^^^^^^

.. code-block:: python

    items = [
        (dt.datetime(2020, 3, 1, 15, 0, 0), 0.),
        (dt.datetime(2020, 3, 1, 16, 0, 0), 1.),
    ]
    ts = tx.TimestampSeries.create_from_tuples(
        items, freq='H', unit='km', time_zone='Europe/Vienna'
    )

.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> print(ts.convert_time_zone('EST'))
    TIME ZONE: EST *** FREQ: H *** UNIT: kilometer
    2020-03-01 09:00:00-05:00    0.0
    2020-03-01 10:00:00-05:00    1.0
    dtype: pint[kilometer]

.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> print(ts.convert_time_zone(dt.timezone.utc))
    TIME ZONE: UTC *** FREQ: H *** UNIT: kilometer
    2020-03-01 14:00:00+00:00    0.0
    2020-03-01 15:00:00+00:00    1.0
    dtype: pint[kilometer]

.. hint::

    param ``timezone`` supports IANA time zone names, ``tzinfo``-objects, such as
    ``datetime.timezone``, ``pytz``- and ``dateutil`` time zone objects


Convert unit
^^^^^^^^^^^^

.. code-block:: python

    items = [
        (dt.datetime(2020, 3, 1, 15, 0, 0), 1.),
        (dt.datetime(2020, 3, 1, 16, 0, 0), 2.),
    ]
    ts = tx.TimestampSeries.create_from_tuples(
        items, freq='H', unit='km', time_zone='Europe/Vienna'
    )

.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> print(ts.convert_unit('m'))
    TIME ZONE: Europe/Vienna *** FREQ: H *** UNIT: meter
    2020-03-01 15:00:00+00:00    1000.0
    2020-03-01 16:00:00+00:00    2000.0

.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> print(ts.convert_unit('meter'))
    TIME ZONE: Europe/Vienna *** FREQ: H *** UNIT: meter
    2020-03-01 15:00:00+00:00    1000.0
    2020-03-01 16:00:00+00:00    2000.0

.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> print(ts.convert_unit(pint.Unit('nautical_mile'))
    TIME ZONE: Europe/Vienna *** FREQ: H *** UNIT: nautical_mile
    2020-03-01 15:00:00+01:00    0.5399568034557236
    2020-03-01 16:00:00+01:00    1.0799136069114472
    dtype: pint[nautical_mile]


.. hint::

    param ``unit`` supports ``pint.Unit``-objects, unit names and unit symbols


Resample frequency
^^^^^^^^^^^^^^^^^^

.. warning::

    Resampling is only supported for smaller frequencies (larger offsets)

    The resampled series will keep the unit, regardless of the return type of the aggregation
    function.


.. code-block:: python

    items = [
        (dt.datetime(2020, 3, 1, 15,  0, 0), 0.),
        (dt.datetime(2020, 3, 1, 15, 30, 0), 1.),
        (dt.datetime(2020, 3, 1, 16,  0, 0), 2.),
        (dt.datetime(2020, 3, 1, 16, 30, 0), 3.),
        (dt.datetime(2020, 3, 1, 17,  0, 0), 4.),
        (dt.datetime(2020, 3, 1, 17, 30, 0), 5.),
    ]
    ts = tx.TimestampSeries.create_from_tuples(
        items, freq='30Min', unit='km', time_zone='Europe/Vienna'
    )


.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> print(ts.resample('1H', 'sum'))
    TIME ZONE: Europe/Vienna *** FREQ: H *** UNIT: kilometer
    2020-03-01 15:00:00+01:00    1.0
    2020-03-01 16:00:00+01:00    5.0
    2020-03-01 17:00:00+01:00    9.0
    Freq: H, dtype: pint[kilometer]

.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> print(ts.resample(dt.timedelta(hours=1), np.mean))
    TIME ZONE: Europe/Vienna *** FREQ: D *** UNIT: kilometer
    2020-03-01 15:00:00+01:00    0.5
    2020-03-01 16:00:00+01:00    2.5
    2020-03-01 17:00:00+01:00    4.5
    Freq: H, dtype: pint[kilometer]

.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> print(ts.resample(pd.offsets.Day(), max))
    TIME ZONE: Europe/Vienna *** FREQ: D *** UNIT: kilometer
    2020-03-01 00:00:00+01:00    5.0
    Freq: D, dtype: pint[kilometer]

.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> print(ts.resample(pd.Timedelta(hours=2), lambda x: float(len(x) > 2)))
    TIME ZONE: Europe/Vienna *** FREQ: 2H *** UNIT: kilometer
    2020-03-01 14:00:00+01:00    0.0
    2020-03-01 16:00:00+01:00    1.0
    Freq: 2H, dtype: pint[kilometer]


.. hint::

    * param ``freq`` supports types: pandas offset aliases, ``datetime.timedelta``,
      ``pandas.Offset`` and ``pandas.Timedelta``

    * param ``method`` supports strings of aggregation method names, python aggregation functions,
      such as ``any`` numpy aggregation functions or custom aggregation callables


Basics
------

.. code-block:: python

    items = [
        (dt.datetime(2020, 3, 1, 15, 0, 0), 0.),
        (dt.datetime(2020, 3, 1, 16, 0, 0), 1.),
        (dt.datetime(2020, 3, 1, 17, 0, 0), 2.),
    ]
    ts = tx.TimestampSeries.create_from_tuples(
        items, freq='H', unit='km', time_zone='Europe/Vienna'
    )


Indexing
^^^^^^^^

.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>>ts[0]
    <Quantity(0.0, 'kilometer')>

.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>>ts[:2]
    TIME ZONE: Europe/Vienna *** FREQ: H *** UNIT: kilometer
    2020-03-01 15:00:00+01:00    0.0
    2020-03-01 16:00:00+01:00    1.0
    dtype: pint[kilometer]

.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> ts[dt.datetime(2020, 3 , 1, 17)]
    <Quantity(2.0, 'kilometer')>

.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> print(ts[[ts.time_zone.localize(dt.datetime(2020, 3 , 1, 16)), ts.time_zone.localize(dt.datetime(2020, 3 , 1, 17))]])
    TIME ZONE: Europe/Vienna *** FREQ: H *** UNIT: kilometer
    2020-03-01 16:00:00+01:00    1.0
    2020-03-01 17:00:00+01:00    2.0
    dtype: pint[kilometer]


Loop
^^^^

.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> [i for i in ts]
    [<Quantity(0.0, 'kilometer')>, <Quantity(1.0, 'kilometer')>, <Quantity(2.0, 'kilometer')>]


Compare
^^^^^^^

.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> import copy
    >>> copy.deepcopy(ts) == ts.convert_time_zone('Europe/Madrid').convert_unit('meter')
    True

.. note::

    Equality operation takes care of different time zones and different (compatible) units


Gap Handling
------------

.. warning::

    Gaps can only be handled for series with set frequency

.. code-block:: python

    items = [
        (dt.datetime(2020, 3, 1, 15, 0, 0), 0.),
        (dt.datetime(2020, 3, 1, 16, 0, 0), None),
        (dt.datetime(2020, 3, 1, 17, 0, 0), 2.),
    ]
    ts = tx.TimestampSeries.create_from_tuples(
        items, freq='H', unit='km', time_zone='Europe/Vienna'
    )


Get gaps
^^^^^^^^

.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> ts.get_gaps()
    [datetime.datetime(2020, 3, 1, 16, 0, tzinfo=<DstTzInfo 'Europe/Vienna' CET+1:00:00 STD>)]

.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> ts.get_gaps(start=dt.datetime(2020, 3, 1, 14, 0, 0), end=dt.datetime(2020, 3, 1, 18, 0, 0))
    [datetime.datetime(2020, 3, 1, 14, 0, tzinfo=<DstTzInfo 'Europe/Vienna' CET+1:00:00 STD>),
     datetime.datetime(2020, 3, 1, 16, 0, tzinfo=<DstTzInfo 'Europe/Vienna' CET+1:00:00 STD>),
     datetime.datetime(2020, 3, 1, 18, 0, tzinfo=<DstTzInfo 'Europe/Vienna' CET+1:00:00 STD>)]



Fill gaps
^^^^^^^^^

.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> print(ts.fill_gaps(value=-1.))
    TIME ZONE: Europe/Vienna *** FREQ: H *** UNIT: kilometer
    2020-03-01 15:00:00+01:00     0.0
    2020-03-01 16:00:00+01:00    -1.0
    2020-03-01 17:00:00+01:00     2.0
    Freq: H, dtype: pint[kilometer]


Series operations
-----------------

.. code-block:: python

    items = [
        (dt.datetime(2020, 3, 1, 15, 0, 0), 0.),
        (dt.datetime(2020, 3, 1, 16, 0, 0), 1.),
        (dt.datetime(2020, 3, 1, 17, 0, 0), 2.),
    ]
    ts = tx.TimestampSeries.create_from_tuples(
        items, freq='H', unit='km', time_zone='Europe/Vienna'
    )

Map function
^^^^^^^^^^^^

.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> print(ts.map(lambda x: max(x, pint.Quantity(2., 'km')), dimensionless=False))
    TIME ZONE: Europe/Vienna *** FREQ: H *** UNIT: kilometer
    2020-03-01 15:00:00+01:00    2.0
    2020-03-01 16:00:00+01:00    2.0
    2020-03-01 17:00:00+01:00    2.0
    dtype: pint[kilometer]

.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> print(ts.map(lambda x: max(x, 2.)), dimensionless=True))
    TIME ZONE: Europe/Vienna *** FREQ: H *** UNIT: kilometer
    2020-03-01 15:00:00+01:00    2.0
    2020-03-01 16:00:00+01:00    2.0
    2020-03-01 17:00:00+01:00    2.0
    dtype: pint[kilometer]


Aggregate series
^^^^^^^^^^^^^^^^

.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> ts.aggregate(max, with_unit=True)
    <Quantity(2.0, 'kilometer')>

.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> ts.aggregate(np.max, with_unit=True)
    <Quantity(2.0, 'kilometer')>

.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> ts.aggregate('max', with_unit=False)
    2.0


Prepend/append items
^^^^^^^^^^^^^^^^^^^^

.. warning::

    Prepending/appending can only be applied to series with set frequency

.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> print(ts.prepend(-1))
    TIME ZONE: Europe/Vienna *** FREQ: H *** UNIT: kilometer
    2020-03-01 14:00:00+01:00    -1.0
    2020-03-01 15:00:00+01:00     0.0
    2020-03-01 16:00:00+01:00     1.0
    2020-03-01 17:00:00+01:00     2.0
    dtype: pint[kilometer]

.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> print(ts.append(3.))
    TIME ZONE: Europe/Vienna *** FREQ: H *** UNIT: kilometer
    2020-03-01 15:00:00+01:00     0.0
    2020-03-01 16:00:00+01:00     1.0
    2020-03-01 17:00:00+01:00     2.0
    2020-03-01 18:00:00+01:00     3.0


Properties
----------

.. code-block:: python

    DATE_FORMAT = '%Y-%m-%dT%H:%M:%S%z'

    items = {
        dt.datetime.strptime('2020-03-01T15:00:00+0000', DATE_FORMAT): 0.,
        dt.datetime.strptime('2020-03-01T16:00:00+0000', DATE_FORMAT): 1.,
        dt.datetime.strptime('2020-03-01T17:00:00+0000', DATE_FORMAT): 2.,
    }
    ts = tx.TimestampSeries.create_from_dict(
        items, freq='infer', unit='km', time_zone='infer'
    )

.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> ts.empty
    False


.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> ts.timestamps
    [datetime.datetime(2020, 3, 1, 15, 0, tzinfo=datetime.timezone.utc), datetime.datetime(2020, 3, 1, 16, 0, tzinfo=datetime.timezone.utc), datetime.datetime(2020, 3, 1, 17, 0, tzinfo=datetime.timezone.utc)]


.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> ts.values
    [0.0, 1.0, 2.0]


.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> ts.unit
    <Unit('kilometer')>

.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> ts.time_zone
    datetime.timezone.utc


.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> ts.freq
    <Hour>


.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> ts.start
    datetime.datetime(2020, 3, 1, 15, 0, tzinfo=datetime.datetime.utc)


.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> ts.end
    datetime.datetime(2020, 3, 1, 17, 0, tzinfo=datetime.datetime.utc)


.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> ts.first
    (datetime.datetime(2020, 3, 1, 15, 0, tzinfo=datetime.datetime.utc), 0.0)


.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> ts.last
    (datetime.datetime(2020, 3, 1, 17, 0, tzinfo=datetime.datetime.utc), 2.0)


.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> ts.time_range
    (datetime.datetime(2020, 3, 1, 15, 0, tzinfo=datetime.datetime.utc),
     datetime.datetime(2020, 3, 1, 17, 0, tzinfo=datetime.datetime.utc))


Cast Series
-----------

As dictionary
^^^^^^^^^^^^^

.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> ts.as_dict()
    {datetime.datetime(2020, 3, 1, 15, 0, tzinfo=datetime.datetime.utc): 0.0,
     datetime.datetime(2020, 3, 1, 16, 0, tzinfo=datetime.datetime.utc): 1.0,
     datetime.datetime(2020, 3, 1, 17, 0, tzinfo=datetime.datetime.utc): 2.0}


As tuples
^^^^^^^^^

.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> ts.as_tuples()
    [(datetime.datetime(2020, 3, 1, 15, 0, tzinfo=datetime.datetime.utc), 0.0),
     (datetime.datetime(2020, 3, 1, 16, 0, tzinfo=datetime.datetime.utc), 1.0),
     (datetime.datetime(2020, 3, 1, 17, 0, tzinfo=datetime.datetime.utc), 2.0)]


As pandas.Series
^^^^^^^^^^^^^^^^

.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> ts.as_pd_series()
    2020-03-01 15:00:00+00:00    0.0
    2020-03-01 16:00:00+00:00    1.0
    2020-03-01 17:00:00+00:00    2.0
    dtype: float64


As darts.TimeSeries
^^^^^^^^^^^^^^^^^^^

.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> ts.as_darts()
    <TimeSeries (DataArray) (time: 3, component: 1, sample: 1)>
    array([[[0.]],
           [[1.]],
           [[2.]]])
    Coordinates:
      * time       (time) datetime64[ns] 2020-03-01T15:00:00 ... 2020-03-01T17:00:00
      * component  (component) <U5 'value'
    Dimensions without coordinates: sample
    Attributes:
        static_covariates:  None
        hierarchy:          None
