Quickstart
----------

Create a ``TimestampSeries``-object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import datetime as dt
    import pandas as pd
    import timeseriesx as tx

    ##################################### create from lists #####################################
    timestamps = [
        dt.datetime(2020, 3, 1, 15, 0, 0),
        dt.datetime(2020, 3, 1, 16, 0, 0),
        dt.datetime(2020, 3, 1, 17, 0, 0),
    ]
    values = [0., 1., 2.]
    ts = tx.TimestampSeries.create_from_lists(
        timestamps, values, freq='H', unit='km', time_zone='Europe/Vienna'
    )

    ##################################### create from dict ######################################
    items = {
        dt.datetime(2020, 3, 1, 15, 0, 0): 0.,
        dt.datetime(2020, 3, 1, 16, 0, 0): 1.,
        dt.datetime(2020, 3, 1, 17, 0, 0): 2.,
    }
    ts = tx.TimestampSeries.create_from_dict(
        items, freq='H', unit='km', time_zone='Europe/Vienna'
    )

    ##################################### create from tuples ####################################
    items = [
        (dt.datetime(2020, 3, 1, 15, 0, 0), 0.),
        (dt.datetime(2020, 3, 1, 16, 0, 0), 1.),
        (dt.datetime(2020, 3, 1, 17, 0, 0), 2.),
    ]
    ts = tx.TimestampSeries.create_from_tuples(
        items, freq='H', unit='km', time_zone='Europe/Vienna'
    )

    ################################# create from pandas.Series #################################
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

    ########################## create constant series from start to end #########################
    ts = tx.TimestampSeries.create_constant_series(
        start=dt.datetime(2020, 3, 1, 15, 0, 0),
        end=dt.datetime(2020, 3, 1, 17, 0, 0),
        value=1.,
        freq='H', unit='km', time_zone='Europe/Vienna'
    )

    # you can also pass datetime.timedelta/pandas.Offset/pandas.Timedelta for param `freq`
    # you can also pass tzinfo-objects (such as pytz-, or dateutil-timezone-objects) for param `time_zone`
    # you can also pass pint.Unit-objects for param `unit`


Infer attributes from raw data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import datetime as dt
    import timeseriesx as tx

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
    <UTC>


Properties
^^^^^^^^^^


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
    [datetime.datetime(2020, 3, 1, 15, 0, tzinfo=<DstTzInfo 'Europe/Vienna' CET+1:00:00 STD>), datetime.datetime(2020, 3, 1, 16, 0, tzinfo=<DstTzInfo 'Europe/Vienna' CET+1:00:00 STD>), datetime.datetime(2020, 3, 1, 17, 0, tzinfo=<DstTzInfo 'Europe/Vienna' CET+1:00:00 STD>)]


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
    <DstTzInfo 'Europe/Vienna' CET+1:00:00 STD>


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
    datetime.datetime(2020, 3, 1, 15, 0, tzinfo=<DstTzInfo 'Europe/Vienna' CET+1:00:00 STD>)


.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> ts.end
    datetime.datetime(2020, 3, 1, 17, 0, tzinfo=<DstTzInfo 'Europe/Vienna' CET+1:00:00 STD>)


.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> ts.first
    (datetime.datetime(2020, 3, 1, 15, 0, tzinfo=<DstTzInfo 'Europe/Vienna' CET+1:00:00 STD>), 0.0)


.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> ts.last
    (datetime.datetime(2020, 3, 1, 17, 0, tzinfo=<DstTzInfo 'Europe/Vienna' CET+1:00:00 STD>), 2.0)


.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> ts.time_range
    (datetime.datetime(2020, 3, 1, 15, 0, tzinfo=<DstTzInfo 'Europe/Vienna' CET+1:00:00 STD>),
     datetime.datetime(2020, 3, 1, 17, 0, tzinfo=<DstTzInfo 'Europe/Vienna' CET+1:00:00 STD>))


Cast Series
^^^^^^^^^^^

.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> ts.as_dict()
    {datetime.datetime(2020, 3, 1, 15, 0, tzinfo=<DstTzInfo 'Europe/Vienna' CET+1:00:00 STD>): 0.0,
     datetime.datetime(2020, 3, 1, 16, 0, tzinfo=<DstTzInfo 'Europe/Vienna' CET+1:00:00 STD>): 1.0,
     datetime.datetime(2020, 3, 1, 17, 0, tzinfo=<DstTzInfo 'Europe/Vienna' CET+1:00:00 STD>): 2.0}


.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> ts.as_tuples()
    [(datetime.datetime(2020, 3, 1, 15, 0, tzinfo=<DstTzInfo 'Europe/Vienna' CET+1:00:00 STD>),
      0.0),
     (datetime.datetime(2020, 3, 1, 16, 0, tzinfo=<DstTzInfo 'Europe/Vienna' CET+1:00:00 STD>),
      1.0),
     (datetime.datetime(2020, 3, 1, 17, 0, tzinfo=<DstTzInfo 'Europe/Vienna' CET+1:00:00 STD>),
      2.0)]


.. prompt::
    :language: python
    :prompts: >>>
    :modifiers: auto

    >>> ts.as_pd_series()
    2020-03-01 15:00:00+01:00    0.0
    2020-03-01 16:00:00+01:00    1.0
    2020-03-01 17:00:00+01:00    2.0
    dtype: float64




Todo:
- map
- aggregate
- append/prepend
- round
- indexing
- calculations
- compare
- loop
- conversion: tz, unit
- resample freq
- get/fill gaps
