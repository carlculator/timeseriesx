=====
Usage
=====

To use TimeSeriesX in a project::

    import timeseriesx as tx


Create a new series::

    timestamps = [
        dt.datetime(2020, 3, 1, 15, 0, 0),
        dt.datetime(2020, 3, 1, 16, 0, 0),
        dt.datetime(2020, 3, 1, 17, 0, 0),
    ]
    values = [0., 1., 2.]
    ts = tx.TimestampSeries.create_from_lists(timestamps, values, freq='H', unit='km')


Convert time zone::

    # set time zone info
    ts = ts.convert_time_zone('Europe/Stockholm')

    # convert time zone
    ts = ts.convert_time_zone('UTC')


Convert unit::

    ts = ts.convert_unit('nautical_mile')

Resample frequency::

    ts = ts.resample(
        '1D',  # one day
        method='sum'
    )


Perform calculations::

    ts = ts + [2., 3., 4]
    ts = ts * ts
    ts = ts / 42.
