import datetime as dt
from collections import OrderedDict

import dateutil
import numpy as np
import pandas as pd
import pint
import pytest
import pytz
from dateutil.tz import tzutc
from pint_pandas import PintArray
from pytz import UnknownTimeZoneError

from timeseriesx import (
    TimestampSeries,
    ureg,
)


@pytest.fixture
def empty_timestamp_series():
    return TimestampSeries(pd.Series([], dtype=float, index=pd.DatetimeIndex([])))


@pytest.fixture
def default_timestamp_series(n=3):
    return TimestampSeries(
        pd.Series(PintArray(np.arange(n, dtype=float), dtype=ureg.parse_units('m')),
                  index=pd.date_range('2020-01-01', freq='D', periods=n, tz='CET')
                  ),
        time_zone='CET', unit='m', freq='D'
    )


def test_timestamp_series_create_null_timeseries():
    start = pd.Timestamp(2020, 3, 1, 15, 0, 0).tz_localize('UTC').to_pydatetime()
    end = pd.Timestamp(2020, 3, 1, 17, 0, 0).tz_localize('UTC').to_pydatetime()
    freq = '1H'
    ts = TimestampSeries.create_null_timeseries(start, end, freq)
    assert ts.freq == pd.offsets.Hour()
    assert ts.time_zone is pytz.UTC
    assert ts.unit is None
    assert ts.timestamps == [
        pytz.UTC.localize(dt.datetime(2020, 3, 1, 15, 0, 0)),
        pytz.UTC.localize(dt.datetime(2020, 3, 1, 16, 0, 0)),
        pytz.UTC.localize(dt.datetime(2020, 3, 1, 17, 0, 0)),
    ]
    assert np.array_equal(ts.values, [np.NaN] * 3, equal_nan=True)


def test_timestamp_series_create_from_lists():
    timestamps = [
        dt.datetime(2020, 3, 1, 15, 0, 0),
        dt.datetime(2020, 3, 1, 16, 0, 0),
        dt.datetime(2020, 3, 1, 17, 0, 0),
    ]
    values = [-1., 0., 1.]
    ts = TimestampSeries.create_from_lists(timestamps, values)
    assert ts.freq == pd.offsets.Hour()
    assert ts.time_zone is None
    assert ts.unit is None
    assert ts.timestamps == timestamps
    assert ts.values == values


def test_timestamp_series_create_from_lists_mismatching_length():
    timestamps = [
        dt.datetime(2020, 3, 1, 15, 0, 0),
        dt.datetime(2020, 3, 1, 16, 0, 0),
    ]
    values = [-1.]
    with pytest.raises(ValueError):
        TimestampSeries.create_from_lists(timestamps, values)


def test_timestamp_series_properties_time_range_default(default_timestamp_series):
    assert default_timestamp_series.time_range == (
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 1, 0, 0, 0)),
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 3, 0, 0, 0))
    )


def test_timestamp_series_properties_time_range_empty(empty_timestamp_series):
    with pytest.raises(ValueError):
        empty_timestamp_series.time_range


def test_timestamp_series_properties_empty_default(default_timestamp_series):
    assert not default_timestamp_series.empty


def test_timestamp_series_properties_empty_empty(empty_timestamp_series):
    assert empty_timestamp_series.empty


def test_timestamp_series_properties_start_default(default_timestamp_series):
    assert default_timestamp_series.start == \
           pytz.timezone('CET').localize(dt.datetime(2020, 1, 1, 0, 0, 0))


def test_timestamp_series_properties_start_empty(empty_timestamp_series):
    with pytest.raises(ValueError):
        empty_timestamp_series.start


def test_timestamp_series_properties_end_default(default_timestamp_series):
    assert default_timestamp_series.end == \
           pytz.timezone('CET').localize(dt.datetime(2020, 1, 3, 0, 0, 0))


def test_timestamp_series_properties_end_empty(empty_timestamp_series):
    with pytest.raises(ValueError):
        empty_timestamp_series.end


def test_timestamp_series_properties_timestamps_default(default_timestamp_series):
    assert default_timestamp_series.timestamps == [
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 1, 0, 0, 0)),
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 2, 0, 0, 0)),
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 3, 0, 0, 0)),
    ]


def test_timestamp_series_properties_timestamps_empty(empty_timestamp_series):
    assert empty_timestamp_series.timestamps == []


def test_timestamp_series_properties_values_default(default_timestamp_series):
    assert default_timestamp_series.values == [0., 1., 2.]


def test_timestamp_series_properties_values_empty(empty_timestamp_series):
    assert empty_timestamp_series.values == []


def test_timestamp_series_properties_first_default(default_timestamp_series):
    assert default_timestamp_series.first == (
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 1, 0, 0, 0)), 0.
    )


def test_timestamp_series_properties_first_empty(empty_timestamp_series):
    with pytest.raises(ValueError):
        empty_timestamp_series.first


def test_timestamp_series_properties_last_default(default_timestamp_series):
    assert default_timestamp_series.last == (
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 3, 0, 0, 0)), 2.
    )


def test_timestamp_series_properties_last_empty(empty_timestamp_series):
    with pytest.raises(ValueError):
        empty_timestamp_series.last


def test_timestamp_series_as_tuples_default(default_timestamp_series):
    assert default_timestamp_series.as_tuples() == [
        (pytz.timezone('CET').localize(dt.datetime(2020, 1, 1, 0, 0, 0)), 0.),
        (pytz.timezone('CET').localize(dt.datetime(2020, 1, 2, 0, 0, 0)), 1.),
        (pytz.timezone('CET').localize(dt.datetime(2020, 1, 3, 0, 0, 0)), 2.),
    ]


def test_timestamp_series_as_tuples_empty(empty_timestamp_series):
    assert empty_timestamp_series.as_tuples() == []


def test_timestamp_series_as_dict_default(default_timestamp_series):
    assert default_timestamp_series.as_dict() == {
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 1, 0, 0, 0)): 0.,
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 2, 0, 0, 0)): 1.,
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 3, 0, 0, 0)): 2.,
    }


def test_timestamp_series_as_dict_ordered_default(default_timestamp_series):
    assert default_timestamp_series.as_dict(ordered=True) == OrderedDict({
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 1, 0, 0, 0)): 0.,
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 2, 0, 0, 0)): 1.,
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 3, 0, 0, 0)): 2.,
    })


def test_timestamp_series_as_dict_empty(empty_timestamp_series):
    assert empty_timestamp_series.as_dict() == {}


def test_timestamp_series_as_pd_series_default(default_timestamp_series):
    pd.testing.assert_series_equal(
        default_timestamp_series.as_pd_series(),
        pd.Series([0., 1., 2.],
                  index=pd.date_range('2020-01-01', freq='D', periods=3, tz='CET')
                  ),
    )


def test_timestamp_series_as_pd_series_empty(empty_timestamp_series):
    pd.testing.assert_series_equal(
        empty_timestamp_series.as_pd_series(),
        pd.Series([], index=pd.DatetimeIndex([]), dtype=np.float64),
    )


def test_timestamp_series_as_pd_series_include_nan():
    ts = TimestampSeries(
        pd.Series(PintArray([0., np.nan, 2.], dtype=ureg.parse_units('m')),
                  index=pd.date_range('2020-01-01', freq='D', periods=3, tz='CET')
                  ),
        time_zone='CET', unit='m', freq='D'
    )
    pd.testing.assert_series_equal(
        ts.as_pd_series(include_nan=True),
        pd.Series([0., np.nan, 2.],
                  index=pd.date_range('2020-01-01', freq='D', periods=3, tz='CET')
                  ),
    )


def test_timestamp_series_as_pd_series_exclude_nan():
    ts = TimestampSeries(
        pd.Series(PintArray([0., np.nan, 2.], dtype=ureg.parse_units('m')),
                  index=pd.date_range('2020-01-01', freq='D', periods=3, tz='CET')
                  ),
        time_zone='CET', unit='m', freq='D'
    )
    pd.testing.assert_series_equal(
        ts.as_pd_series(include_nan=False),
        pd.Series([0., 2.],
                  index=pd.to_datetime(['2020-01-01', '2020-01-03']).tz_localize('CET')
                  ),
    )


def test_timestamp_series_repr_default(default_timestamp_series):
    from pandas import Series, DatetimeIndex
    from numpy import array
    ts = eval(repr(default_timestamp_series))
    pd.testing.assert_series_equal(ts._series, default_timestamp_series._series)


def test_timestamp_series_repr_empty(empty_timestamp_series):
    from pandas import Series, DatetimeIndex
    from numpy import array
    ts = eval(repr(empty_timestamp_series))
    assert ts.empty


def test_timestamp_series_str_default(default_timestamp_series):
    assert "Freq: D" in str(default_timestamp_series)
    assert "Unit: meter" in str(default_timestamp_series)
    assert "Time zone: CET" in str(default_timestamp_series)


def test_timestamp_series_str_empty(empty_timestamp_series):
    assert "Freq: None" in str(empty_timestamp_series)
    assert "Unit: None" in str(empty_timestamp_series)
    assert "Time zone: None" in str(empty_timestamp_series)


def test_map_dimensionless(default_timestamp_series):
    mapped_ts = default_timestamp_series.map(
        lambda x: max(x, 1.)
    )
    assert mapped_ts.timestamps == default_timestamp_series.timestamps
    assert mapped_ts.values == [1., 1., 2.]
    assert mapped_ts.unit == default_timestamp_series.unit
    assert mapped_ts.freq == default_timestamp_series.freq
    assert mapped_ts.time_zone == default_timestamp_series.time_zone


def test_map_with_dimension(default_timestamp_series):
    mapped_ts = default_timestamp_series.map(
        lambda x: max(x, 1. * ureg.Unit('m')),
        dimensionless=False,
    )
    assert mapped_ts.timestamps == default_timestamp_series.timestamps
    assert mapped_ts.values == [1., 1., 2.]
    assert mapped_ts.unit == default_timestamp_series.unit
    assert mapped_ts.freq == default_timestamp_series.freq
    assert mapped_ts.time_zone == default_timestamp_series.time_zone


def test_aggregate_without_unit():
    timestamps = [
        dt.datetime(2020, 1, 1, 0, 0, 0),
        dt.datetime(2020, 1, 2, 0, 0, 0),
    ]
    values = [1.2, 1.5]
    ts = TimestampSeries.create_from_lists(timestamps, values, time_zone='infer')
    assert ts.aggregate(np.max) == 1.5


def test_aggregate_without_returned_unit(default_timestamp_series):
    assert default_timestamp_series.aggregate(np.max) == 2.


def test_aggregate_with_returned_unit(default_timestamp_series):
    assert default_timestamp_series.aggregate(np.max, with_unit=True) == 2. * pint.Unit('m')


def test_sum_no_unit():
    timestamps = [
        dt.datetime(2020, 1, 1, 0, 0, 0),
        dt.datetime(2020, 1, 2, 0, 0, 0),
    ]
    values = [1.2, 1.5]
    ts = TimestampSeries.create_from_lists(timestamps, values, time_zone='infer')
    assert ts.sum() == 2.7


def test_sum_return_without_unit(default_timestamp_series):
    assert default_timestamp_series.sum() == 3.


def test_sum_return_with_unit(default_timestamp_series):
    assert default_timestamp_series.sum(with_unit=True) == 3. * pint.Unit('m')


def test_mean_no_unit():
    timestamps = [
        dt.datetime(2020, 1, 1, 0, 0, 0),
        dt.datetime(2020, 1, 2, 0, 0, 0),
    ]
    values = [1.2, 1.5]
    ts = TimestampSeries.create_from_lists(timestamps, values, time_zone='infer')
    assert ts.mean() == 1.35


def test_mean_return_without_unit(default_timestamp_series):
    assert default_timestamp_series.mean() == 1.


def test_mean_return_with_unit(default_timestamp_series):
    assert default_timestamp_series.mean(with_unit=True) == 1. * pint.Unit('m')


def test_round_no_unit():
    timestamps = [
        dt.datetime(2020, 1, 1, 0, 0, 0),
        dt.datetime(2020, 1, 2, 0, 0, 0),
    ]
    values = [0.49, -1.51]
    ts = TimestampSeries.create_from_lists(timestamps, values, time_zone='infer')
    ts.round(0)
    assert ts.values == [0., -2.]


def test_round_with_unit():
    timestamps = [
        dt.datetime(2020, 1, 1, 0, 0, 0),
        dt.datetime(2020, 1, 2, 0, 0, 0),
    ]
    values = [0.49, -1.51]
    ts = TimestampSeries.create_from_lists(timestamps, values, unit='m')
    rounded = ts.round(0)
    assert rounded.values == [0., -2.]
    assert rounded.timestamps == timestamps
    assert rounded.unit == ureg.Unit('m')
    assert rounded.time_zone is None
    assert rounded.freq == pd.offsets.Day()


def test_timestamp_series_append_default(default_timestamp_series):
    result = default_timestamp_series.append(3.)
    assert result.timestamps == [
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 1, 0, 0, 0)),
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 2, 0, 0, 0)),
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 3, 0, 0, 0)),
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 4, 0, 0, 0)),
    ]
    assert result.values == [0., 1., 2., 3.]
    assert result.unit == default_timestamp_series.unit
    assert result.time_zone == default_timestamp_series.time_zone
    assert result.freq == default_timestamp_series.freq


def test_timestamp_series_append_empty():
    ts = TimestampSeries(pd.Series([], dtype=float, index=pd.DatetimeIndex([])), freq='10s')
    with pytest.raises(ValueError):
        ts.append(1.)


def test_timestamp_series_append_no_freq():
    ts = TimestampSeries(pd.Series([1.], dtype=float, index=pd.DatetimeIndex(['2020-01-01'])))
    with pytest.raises(ValueError):
        ts.append(1.)


def test_timestamp_series_prepend_default(default_timestamp_series):
    result = default_timestamp_series.prepend(-1.)
    assert result.timestamps == [
        pytz.timezone('CET').localize(dt.datetime(2019, 12, 31, 0, 0, 0)),
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 1, 0, 0, 0)),
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 2, 0, 0, 0)),
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 3, 0, 0, 0)),
    ]
    assert result.values == [-1., 0., 1., 2.]
    assert result.unit == default_timestamp_series.unit
    assert result.time_zone == default_timestamp_series.time_zone
    assert result.freq == default_timestamp_series.freq


def test_timestamp_series_prepend_empty(empty_timestamp_series):
    with pytest.raises(ValueError):
        empty_timestamp_series.prepend(1.)


def test_create_timestamp_series_inferred_time_zone_none():
    timestamps = [
        dt.datetime(2020, 1, 1, 0, 0, 0),
        dt.datetime(2020, 1, 2, 0, 0, 0),
    ]
    values = [0., 1.]
    ts = TimestampSeries.create_from_lists(timestamps, values, time_zone='infer')
    assert ts.time_zone is None
    assert ts.timestamps == timestamps
    assert ts.values == values
    assert ts.unit is None
    assert ts.freq == pd.offsets.Day()


def test_create_timestamp_series_inferred_time_zone_valid_fixed_offset():
    timestamps = [
        pd.Timestamp('2020-01-01T00:00:00+01:00').to_pydatetime(),
        pd.Timestamp('2020-01-03T00:00:00+01:00').to_pydatetime(),
    ]
    values = [0., 1.]
    ts = TimestampSeries.create_from_lists(timestamps, values, time_zone='infer')
    assert ts.time_zone == pytz.FixedOffset(60)
    assert ts.timestamps == timestamps
    assert ts.values == values
    assert ts.unit is None
    assert ts.freq == pd.offsets.Day(2)


def test_create_timestamp_series_inferred_time_zone_inconsistent():
    timestamps = [
        pytz.timezone('Europe/London').localize(dt.datetime(2020, 1, 1, 0, 0, 0)),
        pytz.timezone('Europe/Stockholm').localize(dt.datetime(2020, 1, 3, 0, 0, 0)),
    ]
    values = [0., 1.]
    with pytest.raises(ValueError):
        TimestampSeries.create_from_lists(timestamps, values, time_zone='infer')


def test_create_timestamp_series_inferred_time_zone_valid():
    tz_name = 'EST'
    timestamps = [
        pytz.timezone(tz_name).localize(dt.datetime(2020, 1, 1, 0, 0, 0)),
        pytz.timezone(tz_name).localize(dt.datetime(2020, 1, 3, 0, 0, 0)),
    ]
    values = [0., 1.]
    ts = TimestampSeries.create_from_lists(timestamps, values, time_zone='infer')
    assert ts.time_zone is pytz.timezone(tz_name)
    assert ts.timestamps == timestamps
    assert ts.values == values
    assert ts.freq == pd.offsets.Day(2)
    assert ts.unit is None


def test_create_timestamp_series_inferred_freq_by_pandas():
    timestamps = [
        dt.datetime(2020, 1, 1, 0, 0, 0),
        dt.datetime(2020, 1, 3, 0, 0, 0),
    ]
    values = [0., 1.]
    ts = TimestampSeries.create_from_lists(timestamps, values, freq='infer')
    assert ts.freq == pd.offsets.Day(2)
    assert ts.time_zone is None
    assert ts.timestamps == timestamps
    assert ts.values == values
    assert ts.unit is None


def test_create_timestamp_series_inferred_freq_invalid():
    timestamps = [
        dt.datetime(2020, 1, 1, 0, 0, 0),
        dt.datetime(2020, 1, 2, 22, 0, 0),
        dt.datetime(2020, 1, 3, 13, 30, 0),
    ]
    values = [0., 1., 2.]
    with pytest.raises(ValueError):
        TimestampSeries.create_from_lists(timestamps, values, freq='infer')


def test_create_timestamp_series_inferred_freq_valid():
    timestamps = [
        dt.datetime(2020, 1, 1, 0, 0, 0),
        dt.datetime(2020, 1, 2, 0, 0, 0),
        dt.datetime(2020, 1, 3, 0, 0, 0),
    ]
    values = [0., 1., 2.]
    ts = TimestampSeries(series=pd.Series(values, index=timestamps),
                         freq='infer')
    assert ts.freq == pd.offsets.Day()
    assert ts.time_zone is None
    assert ts.timestamps == timestamps
    assert ts.values == values
    assert ts.unit is None


def test_create_timestamp_series_mismatching_time_zone_1():
    values = [0., 1., 2.]
    timestamps = [
        dt.datetime(2020, 1, 1, 0, 0, 0),
        dt.datetime(2020, 1, 2, 0, 0, 0),
        dt.datetime(2020, 1, 3, 0, 0, 0),
    ]
    ts = TimestampSeries(series=pd.Series(values, index=timestamps),
                         time_zone='UTC')
    assert ts.time_zone == pytz.UTC
    assert ts.timestamps == list(map(lambda x: pytz.UTC.localize(x), timestamps))
    assert ts.values == values
    assert ts.unit is None
    assert ts.freq is None


def test_create_timestamp_series_mismatching_time_zone_2():
    tz_name = 'Europe/Berlin'
    timestamps = [
        pytz.timezone(tz_name).localize(dt.datetime(2020, 1, 1, 0, 0, 0)),
        pytz.timezone(tz_name).localize(dt.datetime(2020, 1, 2, 0, 0, 0)),
        pytz.timezone(tz_name).localize(dt.datetime(2020, 1, 3, 0, 0, 0)),
    ]
    values = [0., 1., 2.]
    ts = TimestampSeries(series=pd.Series(values, index=timestamps),
                         time_zone='UTC')
    assert ts.time_zone == pytz.timezone('UTC')
    assert ts.timestamps == list(map(lambda x: x.astimezone(pytz.UTC), timestamps))
    assert ts.values == values
    assert ts.unit is None
    assert ts.freq is None


def test_create_timestamp_series_mismatching_freq():
    timestamps = [
        dt.datetime(2020, 1, 1, 0, 0, 0),
        dt.datetime(2020, 1, 3, 0, 0, 0),
    ]
    values = [0., 1.]
    with pytest.raises(ValueError):
        TimestampSeries.create_from_lists(timestamps, values, freq='10Min')


def test_create_timestamp_series_mismatching_compatible_unit():
    timestamps = [
        dt.datetime(2020, 1, 1, 0, 0, 0),
        dt.datetime(2020, 1, 2, 0, 0, 0),
        dt.datetime(2020, 1, 3, 0, 0, 0),
    ]
    values = [0., 1., 2.]
    ts = TimestampSeries(
        series=pd.Series(PintArray(values, dtype='km'), index=timestamps),
        unit='m'
    )
    assert ts.unit == ureg.Unit('m')
    assert ts.values == [0., 1000., 2000.]
    assert ts.timestamps == timestamps
    assert ts.freq is None
    assert ts.time_zone is None


def test_create_timestamp_series_mismatching_incompatible_unit():
    with pytest.raises(ValueError):
        TimestampSeries(series=pd.Series(PintArray([0., 1., 2.], dtype='m'), index=[
            dt.datetime(2020, 1, 1, 0, 0, 0),
            dt.datetime(2020, 1, 2, 0, 0, 0),
            dt.datetime(2020, 1, 3, 0, 0, 0),
        ]), unit='km/h')


def test_create_timestamp_series_invalid_time_zone():
    illegal_tz = 'Europe/Nantes'
    with pytest.raises(UnknownTimeZoneError):
        TimestampSeries(series=pd.Series([0., 1., 2.], index=[
            dt.datetime(2020, 1, 1, 0, 0, 0),
            dt.datetime(2020, 1, 2, 0, 0, 0),
            dt.datetime(2020, 1, 3, 0, 0, 0),
        ]), time_zone=illegal_tz)


def test_create_timestamp_series_invalid_freq():
    illegal_freq = 'ABC'
    with pytest.raises(ValueError):
        TimestampSeries(series=pd.Series([], dtype=float, index=pd.DatetimeIndex([])),
                        freq=illegal_freq)


def test_create_timestamp_series_invalid_unit():
    illegal_unit = '123'
    with pytest.raises(ValueError):
        TimestampSeries(series=pd.Series([], dtype=float, index=pd.DatetimeIndex([])),
                        unit=illegal_unit)


def test_create_timestamp_series_invalid_index():
    with pytest.raises(ValueError):
        TimestampSeries(series=pd.Series([0., 1., 2.], index=[
            dt.datetime(2020, 1, 3, 0, 0, 0),
            dt.datetime(2020, 1, 2, 0, 0, 0),
            dt.datetime(2020, 1, 1, 0, 0, 0),
        ]))


def test_create_timestamp_series_valid_unit_str():
    ts = TimestampSeries(series=pd.Series([], index=pd.DatetimeIndex([]), dtype=np.float64),
                         unit='second')
    assert ts.unit == ureg.second
    assert ts.timestamps == []
    assert ts.values == []
    assert ts.freq is None
    assert ts.time_zone is None


def test_create_timestamp_series_valid_unit_obj():
    ts = TimestampSeries(series=pd.Series([], index=pd.DatetimeIndex([]), dtype=np.float64),
                         unit=ureg.second)
    assert ts.unit == ureg.second
    assert ts.timestamps == []
    assert ts.values == []
    assert ts.freq is None
    assert ts.time_zone is None


def test_create_timestamp_series_valid_freq_str():
    ts = TimestampSeries(series=pd.Series([], index=pd.DatetimeIndex([]), dtype=np.float64),
                         freq='30Min')
    assert ts.freq == pd.offsets.Minute(30)
    assert ts.timestamps == []
    assert ts.values == []
    assert ts.time_zone is None
    assert ts.unit is None


def test_create_timestamp_series_valid_freq_timedelta():
    ts = TimestampSeries(series=pd.Series([], index=pd.DatetimeIndex([]), dtype=np.float64),
                         freq=pd.Timedelta(minutes=30))
    assert ts.freq == pd.offsets.Minute(30)
    assert ts.timestamps == []
    assert ts.values == []
    assert ts.time_zone is None
    assert ts.unit is None


def test_create_timestamp_series_valid_freq_offset():
    ts = TimestampSeries(series=pd.Series([], index=pd.DatetimeIndex([]), dtype=np.float64),
                         freq=pd.offsets.Minute(30))
    assert ts.freq == pd.offsets.Minute(30)
    assert ts.timestamps == []
    assert ts.values == []
    assert ts.time_zone is None


def test_create_timestamp_series_valid_timezone_str():
    ts = TimestampSeries(series=pd.Series([], index=pd.DatetimeIndex([]), dtype=np.float64),
                         time_zone='utc')
    assert ts.time_zone == pytz.UTC
    assert ts.timestamps == []
    assert ts.values == []
    assert ts.freq is None
    assert ts.unit is None


def test_create_timestamp_series_valid_timezone_obj_pytz():
    ts = TimestampSeries(series=pd.Series([], index=pd.DatetimeIndex([]), dtype=np.float64),
                         time_zone=pytz.UTC)
    assert ts.time_zone == pytz.UTC
    assert ts.timestamps == []
    assert ts.values == []
    assert ts.freq is None
    assert ts.unit is None


def test_create_timestamp_series_valid_timezone_obj_dateutil():
    ts = TimestampSeries(series=pd.Series([], index=pd.DatetimeIndex([]), dtype=np.float64),
                         time_zone=dateutil.tz.tzutc())
    assert isinstance(ts.time_zone, tzutc)
    assert ts.timestamps == []
    assert ts.values == []
    assert ts.freq is None
    assert ts.unit is None


def test_timestamp_series_add_timestamp_series_different_freq(default_timestamp_series):
    add_ts = TimestampSeries(
        pd.Series(PintArray(np.arange(3, dtype=float), dtype=ureg.parse_units('m')),
                  index=pd.date_range('2020-01-01', freq='H', periods=3, tz='CET')
                  ),
        time_zone='CET', unit='m', freq='H'
    )
    with pytest.raises(ValueError):
        default_timestamp_series + add_ts


def test_timestamp_series_add_timestamp_series_different_tz():
    ts_1 = TimestampSeries(
        pd.Series(PintArray(np.arange(3, dtype=float), dtype=ureg.parse_units('m')),
                  index=pd.date_range('2020-01-01T01:00:00', freq='H', periods=3, tz='CET')
                  ),
        time_zone='CET', unit='m', freq='H'
    )
    ts_2 = TimestampSeries(
        pd.Series(PintArray(np.arange(3, 6, dtype=float), dtype=ureg.parse_units('m')),
                  index=pd.date_range('2020-01-01T00:00:00', freq='H', periods=3, tz='UTC')
                  ),
        time_zone='UTC', unit='m', freq='H'
    )
    result_ts = ts_1 + ts_2
    assert result_ts.values == [3., 5., 7.]
    assert result_ts.timestamps == [
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 1, 1, 0, 0)),
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 1, 2, 0, 0)),
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 1, 3, 0, 0)),
    ]
    assert result_ts.time_zone == ts_1.time_zone
    assert result_ts.unit == ts_1.unit
    assert result_ts.freq == pd.offsets.Hour()


def test_timestamp_series_add_timestamp_series_different_unit(default_timestamp_series):
    add_ts = TimestampSeries(
        pd.Series(PintArray(np.arange(3, dtype=float), dtype=ureg.parse_units('kg')),
                  index=pd.date_range('2020-01-01', freq='D', periods=3, tz='CET')
                  ),
        time_zone='CET', unit='kg', freq='D'
    )
    with pytest.raises(ValueError):
        default_timestamp_series + add_ts


def test_timestamp_series_add_timestamp_series_different_index(default_timestamp_series):
    add_ts = TimestampSeries(
        pd.Series(PintArray(np.arange(3, dtype=float), dtype=ureg.parse_units('m')),
                  index=pd.date_range('2020-01-02', freq='D', periods=3, tz='CET')
                  ),
        time_zone='CET', unit='m', freq='D'
    )
    result_ts = default_timestamp_series + add_ts
    assert result_ts.values == [0., 1., 3., 2.]
    assert result_ts.timestamps == [
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 1)),
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 2)),
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 3)),
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 4)),
    ]
    assert result_ts.time_zone == default_timestamp_series.time_zone
    assert result_ts.unit == default_timestamp_series.unit
    assert result_ts.freq == pd.offsets.Day()


def test_timestamp_series_add_pandas_series():
    ts = TimestampSeries(
        pd.Series(np.arange(3, dtype=float),
                  index=pd.date_range('2020-01-01', freq='D', periods=3, tz='UTC')
                  ),
        time_zone='UTC', freq='D'
    )
    pd_series = pd.Series(np.arange(3),
                          index=pd.date_range('2020-01-02',
                                              freq='D', periods=3, tz='UTC'))
    result_ts = ts + pd_series
    assert result_ts.values == [0., 1., 3., 2.]
    assert result_ts.timestamps == [
        pytz.timezone('UTC').localize(dt.datetime(2020, 1, 1)),
        pytz.timezone('UTC').localize(dt.datetime(2020, 1, 2)),
        pytz.timezone('UTC').localize(dt.datetime(2020, 1, 3)),
        pytz.timezone('UTC').localize(dt.datetime(2020, 1, 4)),
    ]
    assert result_ts.time_zone == ts.time_zone
    assert result_ts.unit == ts.unit
    assert result_ts.freq == pd.offsets.Day()


def test_timestamp_series_add_pandas_series_different_time_zones():
    ts = TimestampSeries(
        pd.Series(np.arange(3, dtype=float),
                  index=pd.date_range('2020-01-01', freq='D', periods=3, tz='UTC')
                  ),
        time_zone='Europe/Stockholm', freq='D'
    )
    pd_series = pd.Series(np.arange(3.),
                          index=pd.date_range('2020-01-01',
                                              freq='D', periods=3, tz='UTC'))
    result_ts = ts + pd_series
    assert result_ts.values == [0., 2., 4.]
    assert result_ts.timestamps == [
        pytz.timezone('UTC').localize(dt.datetime(2020, 1, 1)).astimezone(pytz.timezone('Europe/Stockholm')),
        pytz.timezone('UTC').localize(dt.datetime(2020, 1, 2)).astimezone(pytz.timezone('Europe/Stockholm')),
        pytz.timezone('UTC').localize(dt.datetime(2020, 1, 3)).astimezone(pytz.timezone('Europe/Stockholm')),
    ]
    assert result_ts.time_zone == ts.time_zone
    assert result_ts.unit == ts.unit
    assert result_ts.freq == pd.offsets.Day()


def test_timestamp_series_add_list_error():
    ts = TimestampSeries(
        pd.Series(np.arange(3, dtype=float),
                  index=pd.date_range('2020-01-01', freq='D', periods=3, tz='UTC')
                  ),
        time_zone='UTC', freq='D'
    )
    with pytest.raises(ValueError):
        ts + [1, 2]


def test_timestamp_series_add_list_ok():
    ts = TimestampSeries(
        pd.Series(np.arange(3),
                  index=pd.date_range('2020-01-01', freq='D', periods=3, tz='UTC')
                  ),
        time_zone='UTC', freq='D'
    )
    result_ts = ts + [0., 1., 2.]
    assert result_ts.values == [0, 2., 4.]
    assert result_ts.timestamps == ts.timestamps
    assert result_ts.unit == ts.unit
    assert result_ts.freq == ts.freq
    assert result_ts.time_zone == ts.time_zone


def test_timestamp_series_add_pint_array(default_timestamp_series):
    result_ts = default_timestamp_series + PintArray([0., 100., 200.], dtype='cm')
    assert result_ts.values == [0., 2., 4.]
    assert result_ts.timestamps == default_timestamp_series.timestamps
    assert result_ts.unit == default_timestamp_series.unit
    assert result_ts.freq == default_timestamp_series.freq
    assert result_ts.time_zone == default_timestamp_series.time_zone


def test_timestamp_series_multiply_pint_array(default_timestamp_series):
    result_ts = default_timestamp_series * PintArray([1., 1., 1.], dtype='m')
    assert result_ts.values == [0., 1., 2.]
    assert result_ts.unit == ureg.Unit('m^2')
    assert result_ts.timestamps == default_timestamp_series.timestamps
    assert result_ts.freq == default_timestamp_series.freq
    assert result_ts.time_zone == default_timestamp_series.time_zone


def test_timestamp_series_add_scalar():
    ts = TimestampSeries(
        pd.Series(np.arange(3, dtype=float),
                  index=pd.date_range('2020-01-01', freq='D', periods=3, tz='UTC')
                  ),
        time_zone='UTC', freq='D'
    )
    result_ts = ts + 1
    assert result_ts.values == [1., 2., 3.]
    assert result_ts.timestamps == ts.timestamps
    assert result_ts.unit == ts.unit
    assert result_ts.freq == ts.freq
    assert result_ts.time_zone == ts.time_zone


def test_timestamp_series_multiply_scalar(default_timestamp_series):
    result_ts = default_timestamp_series * 2
    assert result_ts.values == [0., 2., 4.]
    assert result_ts.timestamps == default_timestamp_series.timestamps
    assert result_ts.unit == default_timestamp_series.unit
    assert result_ts.freq == default_timestamp_series.freq
    assert result_ts.time_zone == default_timestamp_series.time_zone


def test_timestamp_series_multiply_pint_scalar():
    ts = TimestampSeries(
        pd.Series(np.arange(3, dtype=float),
                  index=pd.date_range('2020-01-01', freq='D', periods=3)
                  ),
        unit='m'
    )
    result_ts = ts * (2 * ureg.Unit('m'))
    assert result_ts.values == [0., 2., 4.]
    assert result_ts.unit == ureg.Unit('m^2')
    assert result_ts.timestamps == ts.timestamps
    assert result_ts.freq == ts.freq
    assert result_ts.time_zone == ts.time_zone


def test_timestamp_series_div_scalar(default_timestamp_series):
    result_ts = default_timestamp_series / 2
    assert result_ts.values == [0., .5, 1.]
    assert result_ts.timestamps == default_timestamp_series.timestamps
    assert result_ts.unit == default_timestamp_series.unit
    assert result_ts.freq == default_timestamp_series.freq
    assert result_ts.time_zone == default_timestamp_series.time_zone


def test_timestamp_series_floordiv_pint_scalar(default_timestamp_series):
    result_ts = default_timestamp_series // (2 * ureg.Unit('m'))
    assert result_ts.values == [0., 0., 1.]
    assert result_ts.unit is None
    assert result_ts.timestamps == default_timestamp_series.timestamps
    assert result_ts.freq == default_timestamp_series.freq
    assert result_ts.time_zone == default_timestamp_series.time_zone


def test_timestamp_series_div_pint_scalar():
    ts = TimestampSeries(
        pd.Series(np.arange(3, dtype=float),
                  index=pd.date_range('2020-01-01', freq='D', periods=3)
                  ),
        unit='m^2'
    )
    result_ts = ts / (2 * ureg.Unit('m'))
    assert result_ts.values == [0., .5, 1.]
    assert result_ts.unit == ureg.Unit('m')
    assert result_ts.timestamps == ts.timestamps
    assert result_ts.freq == ts.freq
    assert result_ts.time_zone == ts.time_zone


def test_timestamp_series_subtract_list():
    ts = TimestampSeries(
        pd.Series(np.arange(3, dtype=float),
                  index=pd.date_range('2020-01-01', freq='D', periods=3)
                  ))
    result_ts = ts - [0., 1., 2.]
    assert result_ts.values == [0., 0., 0.]
    assert result_ts.timestamps == ts.timestamps
    assert result_ts.freq == ts.freq
    assert result_ts.unit == ts.unit
    assert result_ts.time_zone == ts.time_zone


def test_timestamp_series_subtract_pd_series_mismatch():
    ts = TimestampSeries(
        pd.Series(np.arange(3, dtype=float),
                  index=pd.date_range('2020-01-01', freq='D', periods=3)),
    )
    result_ts = ts - pd.Series(np.arange(3),
                               index=pd.date_range('2020-01-02', freq='D', periods=3))
    assert result_ts.values == [0., 1., 1., -2.]
    assert result_ts.timestamps == [
        dt.datetime(2020, 1, 1),
        dt.datetime(2020, 1, 2),
        dt.datetime(2020, 1, 3),
        dt.datetime(2020, 1, 4),
    ]
    assert result_ts.freq == ts.freq
    assert result_ts.unit == ts.unit
    assert result_ts.time_zone == ts.time_zone


def test_get_item_positional_index(default_timestamp_series):
    assert default_timestamp_series[0] == 0 * ureg.Unit('m')


def test_get_item_positional_slice_index(default_timestamp_series):
    result_ts = default_timestamp_series[:2]
    assert result_ts.timestamps == default_timestamp_series.timestamps[:2]
    assert result_ts.values == default_timestamp_series.values[:2]
    assert result_ts.unit == default_timestamp_series.unit
    assert result_ts.freq == default_timestamp_series.freq
    assert result_ts.time_zone == default_timestamp_series.time_zone


def test_get_item_positional_list_index(default_timestamp_series):
    result_ts = default_timestamp_series[[0, 1]]
    assert result_ts.timestamps == default_timestamp_series.timestamps[:2]
    assert result_ts.values == default_timestamp_series.values[:2]
    assert result_ts.unit == default_timestamp_series.unit
    assert result_ts.freq == default_timestamp_series.freq
    assert result_ts.time_zone == default_timestamp_series.time_zone


def test_get_item_datetime_index(default_timestamp_series):
    assert default_timestamp_series[default_timestamp_series.start] == 0 * ureg.Unit('m')


def test_get_item_naive_datetime_index(default_timestamp_series):
    assert default_timestamp_series[
               default_timestamp_series.start.replace(tzinfo=None)] == 0 * ureg.Unit('m')


def test_get_item_datetime_list_index(default_timestamp_series):
    result_ts = default_timestamp_series[default_timestamp_series.timestamps[:2]]
    assert result_ts.timestamps == default_timestamp_series.timestamps[:2]
    assert result_ts.values == default_timestamp_series.values[:2]
    assert result_ts.unit == default_timestamp_series.unit
    assert result_ts.freq == default_timestamp_series.freq
    assert result_ts.time_zone == default_timestamp_series.time_zone


def test_loop(default_timestamp_series):
    for idx, item in enumerate(default_timestamp_series):
        assert item == idx * ureg.Unit('m')


def test_eq(default_timestamp_series):
    ts1 = TimestampSeries.create_from_tuples(
        [(pd.Timestamp(2020, 1, 1).tz_localize('UTC').tz_convert('EST'), 1.),
         (pd.Timestamp(2020, 1, 2).tz_localize('UTC').tz_convert('EST'), 2.)],
        unit='km', freq=None,
    )
    ts2 = TimestampSeries.create_from_tuples(
        [(pd.Timestamp(2020, 1, 1).tz_localize('UTC').tz_convert('GMT'), 1000.),
         (pd.Timestamp(2020, 1, 2).tz_localize('UTC').tz_convert('GMT'), 2000.)],
        unit='m', freq=None,
    )
    assert ts1 == ts2


def test_neq_1(default_timestamp_series):
    ts1 = TimestampSeries.create_from_tuples(
        [(pd.Timestamp(2020, 1, 1).tz_localize('UTC').tz_convert('EST'), 1.),
         (pd.Timestamp(2020, 1, 2).tz_localize('EST'), 2.)],
        unit='km', freq=None,
    )
    ts2 = TimestampSeries.create_from_tuples(
        [(pd.Timestamp(2020, 1, 1).tz_localize('UTC').tz_convert('GMT'), 1000.),
         (pd.Timestamp(2020, 1, 2).tz_localize('UTC').tz_convert('GMT'), 2000.)],
        unit='m', freq=None,
    )
    assert ts1 != ts2


def test_neq_2(default_timestamp_series):
    ts1 = TimestampSeries.create_from_tuples(
        [(pd.Timestamp(2020, 1, 1).tz_localize('UTC').tz_convert('EST'), 1.),
         (pd.Timestamp(2020, 1, 2).tz_localize('UTC').tz_convert('EST'), 3.)],
        unit='km', freq=None,
    )
    ts2 = TimestampSeries.create_from_tuples(
        [(pd.Timestamp(2020, 1, 1).tz_localize('UTC').tz_convert('GMT'), 1000.),
         (pd.Timestamp(2020, 1, 2).tz_localize('UTC').tz_convert('GMT'), 2000.)],
        unit='m', freq=None,
    )
    assert ts1 != ts2


def test_neq_3(default_timestamp_series):
    ts1 = TimestampSeries.create_from_tuples(
        [(pd.Timestamp(2020, 1, 1).tz_localize('UTC'), 1.),
         (pd.Timestamp(2020, 1, 2).tz_localize('UTC'), 2.)],
        unit='m', freq=None,
    )
    ts2 = TimestampSeries.create_from_tuples(
        [(pd.Timestamp(2020, 1, 1).tz_localize('UTC'), 1.),
         (pd.Timestamp(2020, 1, 2).tz_localize('UTC'), 2.)],
        unit=None, freq=None,
    )
    assert ts1 != ts2


def test_neq_4(default_timestamp_series):
    ts1 = TimestampSeries.create_from_tuples(
        [(pd.Timestamp(2020, 1, 1).tz_localize('UTC'), 1.),
         (pd.Timestamp(2020, 1, 2).tz_localize('UTC'), 2.)],
        unit=None, freq=None,
    )
    ts2 = TimestampSeries.create_from_tuples(
        [(pd.Timestamp(2020, 1, 1).tz_localize('UTC'), 1.),
         (pd.Timestamp(2020, 1, 2).tz_localize('UTC'), 2.)],
        unit='m', freq=None,
    )
    assert ts1 != ts2


def test_timestamp_series_convert_time_zone_str():
    tz_name = 'Europe/Stockholm'
    ts = TimestampSeries(
        pd.Series(np.arange(3, dtype=float),
                  index=pd.date_range('2020-01-01T00:00:00', freq='H', periods=3)
                  ),
        time_zone='UTC', freq='H'
    )
    result_ts = ts.convert_time_zone(tz_name)
    assert result_ts.timestamps == [
        pytz.timezone(tz_name).localize(dt.datetime(2020, 1, 1, 1)),
        pytz.timezone(tz_name).localize(dt.datetime(2020, 1, 1, 2)),
        pytz.timezone(tz_name).localize(dt.datetime(2020, 1, 1, 3)),
    ]
    assert result_ts.time_zone == pytz.timezone(tz_name)
    assert result_ts.unit == ts.unit
    assert result_ts.freq == ts.freq
    assert result_ts.values == ts.values


def test_timestamp_series_convert_time_zone_pytz():
    time_zone = pytz.timezone('Europe/Stockholm')
    ts = TimestampSeries(
        pd.Series(np.arange(3, dtype=float),
                  index=pd.date_range('2020-01-01T00:00:00', freq='H', periods=3)
                  ),
        time_zone='UTC', freq='H'
    )
    result_ts = ts.convert_time_zone(time_zone)
    assert result_ts.timestamps == [
        time_zone.localize(dt.datetime(2020, 1, 1, 1)),
        time_zone.localize(dt.datetime(2020, 1, 1, 2)),
        time_zone.localize(dt.datetime(2020, 1, 1, 3)),
    ]
    assert result_ts.time_zone == time_zone
    assert result_ts.unit == ts.unit
    assert result_ts.freq == ts.freq
    assert result_ts.values == ts.values


def test_timestamp_series_convert_time_zone_from_none():
    tz_name = 'Europe/Stockholm'
    ts = TimestampSeries(
        pd.Series(np.arange(3, dtype=float),
                  index=pd.date_range('2020-01-01', freq='H', periods=3)
                  ),
        time_zone=None, freq='H'
    )
    result_ts = ts.convert_time_zone(tz_name)
    assert result_ts.timestamps == [
        pytz.timezone(tz_name).localize(dt.datetime(2020, 1, 1, 0)),
        pytz.timezone(tz_name).localize(dt.datetime(2020, 1, 1, 1)),
        pytz.timezone(tz_name).localize(dt.datetime(2020, 1, 1, 2)),
    ]
    assert result_ts.time_zone == pytz.timezone(tz_name)
    assert result_ts.unit == ts.unit
    assert result_ts.freq == ts.freq
    assert result_ts.values == ts.values


def test_timestamp_series_convert_unit_from_none():
    ts = TimestampSeries(
        pd.Series(np.arange(3, dtype=float),
                  index=pd.date_range('2020-01-01', freq='D', periods=3, tz='UTC')
                  ),
        time_zone='UTC', freq='D'
    )
    ts = ts.convert_unit('meter')
    assert ts.unit == ureg.parse_units('meter')
    assert ts.values == [0., 1., 2.]
    assert ts.timestamps == [
        pytz.UTC.localize(dt.datetime(2020, 1, 1)),
        pytz.UTC.localize(dt.datetime(2020, 1, 2)),
        pytz.UTC.localize(dt.datetime(2020, 1, 3)),
    ]
    assert ts.time_zone == pytz.UTC
    assert ts.freq == pd.offsets.Day()


def test_timestamp_series_convert_unit_to_none(default_timestamp_series):
    ts = default_timestamp_series.convert_unit(None)
    assert ts.unit is None
    assert ts.values == [0., 1., 2.]
    assert ts.timestamps == [
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 1)),
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 2)),
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 3)),
    ]
    assert ts.time_zone == pytz.timezone('CET')
    assert ts.freq == pd.offsets.Day()


def test_timestamp_series_convert_unit_incompatible(default_timestamp_series):
    with pytest.raises(ValueError):
        default_timestamp_series.convert_unit('liter')


def test_timestamp_series_convert_unit_success(default_timestamp_series):
    ts = default_timestamp_series * 1000
    ts = ts.convert_unit('nautical_mile')
    assert ts.unit == ureg.parse_units('nautical_mile')
    assert np.allclose(ts.values, [0., 0.54, 1.08], atol=.001)
    assert ts.timestamps == [
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 1)),
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 2)),
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 3)),
    ]
    assert ts.time_zone == pytz.timezone('CET')
    assert ts.freq == pd.offsets.Day()


def test_timestamp_series_fill_gaps():
    tz = 'Europe/Berlin'
    timestamps = [
        dt.datetime(2020, 3, 1, 15, 0, 0),
        dt.datetime(2020, 3, 1, 16, 0, 0),
        dt.datetime(2020, 3, 1, 17, 0, 0),
    ]
    values = [1., 1., 1.]
    ts = TimestampSeries.create_from_lists(timestamps, values, freq=pd.offsets.Hour(),
                                           time_zone=tz)
    ts = ts.fill_gaps(ts.start - dt.timedelta(hours=1),
                      ts.end + dt.timedelta(hours=1), value=2)
    assert ts.timestamps == [
        pytz.timezone(tz).localize(dt.datetime(2020, 3, 1, 14, 0, 0)),
        pytz.timezone(tz).localize(dt.datetime(2020, 3, 1, 15, 0, 0)),
        pytz.timezone(tz).localize(dt.datetime(2020, 3, 1, 16, 0, 0)),
        pytz.timezone(tz).localize(dt.datetime(2020, 3, 1, 17, 0, 0)),
        pytz.timezone(tz).localize(dt.datetime(2020, 3, 1, 18, 0, 0)),
    ]
    assert ts.values == [2., 1., 1., 1., 2.]
    assert ts.unit is None
    assert ts.freq == pd.offsets.Hour()
    assert ts.time_zone == pytz.timezone(tz)


def test_timestamp_series_fill_gaps_start_and_end_different_timezone():
    timestamps = [
        dt.datetime(2020, 3, 1, 15, 0, 0),
        dt.datetime(2020, 3, 1, 16, 0, 0),
        dt.datetime(2020, 3, 1, 17, 0, 0),
    ]
    values = [1., 1., 1.]
    ts = TimestampSeries.create_from_lists(
        timestamps,
        values,
        freq=pd.offsets.Hour(),
        time_zone="UTC"
    )
    with pytest.raises(ValueError):
        ts.fill_gaps(
            pytz.timezone('Europe/Berlin').localize(timestamps[0]) - dt.timedelta(hours=1),
            pytz.timezone('Europe/Berlin').localize(timestamps[-1]) + dt.timedelta(hours=1),
            value=2
        )


def test_timestamp_series_fill_gaps_start_different_timezone():
    timestamps = [
        dt.datetime(2020, 3, 1, 15, 0, 0),
        dt.datetime(2020, 3, 1, 16, 0, 0),
        dt.datetime(2020, 3, 1, 17, 0, 0),
    ]
    values = [1., 1., 1.]
    ts = TimestampSeries.create_from_lists(
        timestamps,
        values,
        freq=pd.offsets.Hour(),
        time_zone="UTC"
    )
    with pytest.raises(ValueError):
        ts.fill_gaps(
            pytz.timezone('Europe/Berlin').localize(timestamps[0]) - dt.timedelta(hours=1),
            value=2
        )


def test_timestamp_series_fill_gaps_start_and_end_no_timezone():
    tz = 'Europe/Berlin'
    timestamps = [
        dt.datetime(2020, 3, 1, 15, 0, 0),
        dt.datetime(2020, 3, 1, 16, 0, 0),
        dt.datetime(2020, 3, 1, 17, 0, 0),
    ]
    values = [1., 1., 1.]
    ts = TimestampSeries.create_from_lists(
        timestamps,
        values,
        freq=pd.offsets.Hour(),
        time_zone=tz
    )
    ts.fill_gaps(
        timestamps[0] - dt.timedelta(hours=1),
        timestamps[-1] + dt.timedelta(hours=1),
        value=2
    )
    assert ts.timestamps == [
        pytz.timezone(tz).localize(dt.datetime(2020, 3, 1, 14, 0, 0)),
        pytz.timezone(tz).localize(dt.datetime(2020, 3, 1, 15, 0, 0)),
        pytz.timezone(tz).localize(dt.datetime(2020, 3, 1, 16, 0, 0)),
        pytz.timezone(tz).localize(dt.datetime(2020, 3, 1, 17, 0, 0)),
        pytz.timezone(tz).localize(dt.datetime(2020, 3, 1, 18, 0, 0)),
    ]
    assert ts.values == [2., 1., 1., 1., 2.]
    assert ts.unit is None
    assert ts.freq == pd.offsets.Hour()
    assert ts.time_zone == pytz.timezone(tz)


def test_timestamp_series_fill_gaps_no_start_and_end():
    timestamps = [
        dt.datetime(2020, 3, 1, 15, 0, 0),
        dt.datetime(2020, 3, 1, 16, 0, 0),
        dt.datetime(2020, 3, 1, 17, 0, 0),
    ]
    values = [1., 1., np.nan]
    ts = TimestampSeries.create_from_lists(timestamps, values, freq=pd.offsets.Hour())
    ts = ts.fill_gaps(value=2)
    assert ts.timestamps == timestamps
    assert ts.values == [1., 1., 2.]
    assert ts.time_zone is None
    assert ts.unit is None
    assert ts.freq == pd.offsets.Hour()


def test_timestamp_series_fill_gaps_with_unit(default_timestamp_series):
    ts = default_timestamp_series.fill_gaps(
        end=default_timestamp_series.timestamps[-1] + default_timestamp_series.freq,
        value=0.
    )
    assert ts.values == [0., 1., 2., 0.]
    assert ts.timestamps == [
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 1)),
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 2)),
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 3)),
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 4)),
    ]
    assert ts.unit == ureg.parse_units('meter')
    assert ts.time_zone == pytz.timezone('CET')
    assert ts.freq == pd.offsets.Day()


def test_timestamp_series_fill_gaps_empty_series(empty_timestamp_series):
    with pytest.raises(ValueError):
        empty_timestamp_series.fill_gaps()


def test_timestamp_series_get_gaps():
    timestamps = [
        dt.datetime(2020, 3, 1, 15, 0, 0),
        dt.datetime(2020, 3, 1, 16, 0, 0),
        dt.datetime(2020, 3, 1, 17, 0, 0),
    ]
    values = [1., 1., 1.]
    ts = TimestampSeries.create_from_lists(timestamps, values, freq=pd.offsets.Hour())
    gaps = ts.get_gaps(timestamps[0] - dt.timedelta(hours=1),
                       timestamps[-1] + dt.timedelta(hours=1))
    assert gaps == [
        timestamps[0] - dt.timedelta(hours=1),
        timestamps[-1] + dt.timedelta(hours=1)
    ]


def test_timestamp_series_get_gaps_start_and_end_different_timezone():
    timestamps = [
        dt.datetime(2020, 3, 1, 15, 0, 0),
        dt.datetime(2020, 3, 1, 16, 0, 0),
        dt.datetime(2020, 3, 1, 17, 0, 0),
    ]
    values = [1., 1., 1.]
    ts = TimestampSeries.create_from_lists(
        timestamps,
        values,
        freq=pd.offsets.Hour(),
        time_zone="UTC"
    )
    with pytest.raises(ValueError):
        ts.get_gaps(
            pytz.timezone('Europe/Berlin').localize(timestamps[0]) - dt.timedelta(hours=1),
            pytz.timezone('Europe/Berlin').localize(timestamps[-1]) + dt.timedelta(hours=1),
        )


def test_timestamp_series_get_gaps_start_and_end_no_timezone():
    timestamps = [
        dt.datetime(2020, 3, 1, 15, 0, 0),
        dt.datetime(2020, 3, 1, 16, 0, 0),
        dt.datetime(2020, 3, 1, 17, 0, 0),
    ]
    values = [1., 1., 1.]
    ts = TimestampSeries.create_from_lists(
        timestamps,
        values,
        freq=pd.offsets.Hour(),
        time_zone="Europe/Berlin"
    )
    gaps = ts.get_gaps(
        timestamps[0] - dt.timedelta(hours=1),
        timestamps[-1] + dt.timedelta(hours=1),
    )
    assert gaps == [
        pytz.timezone('Europe/Berlin').localize(timestamps[0]) - dt.timedelta(
            hours=1),
        pytz.timezone('Europe/Berlin').localize(timestamps[-1]) + dt.timedelta(
            hours=1),
    ]


def test_timestamp_series_get_gaps_no_start_and_end():
    timestamps = [
        dt.datetime(2020, 3, 1, 15, 0, 0),
        dt.datetime(2020, 3, 1, 16, 0, 0),
        dt.datetime(2020, 3, 1, 17, 0, 0),
    ]
    values = [np.nan, 1., 1.]
    ts = TimestampSeries.create_from_lists(timestamps, values, freq=pd.offsets.Hour())
    gaps = ts.get_gaps()
    assert gaps == [
        timestamps[0]
    ]


def test_timestamp_series_get_gaps_empty_series(empty_timestamp_series):
    with pytest.raises(ValueError):
        empty_timestamp_series.get_gaps()


def test_timestamp_series_get_gaps_with_unit(default_timestamp_series):
    end = default_timestamp_series.timestamps[-1] + default_timestamp_series.freq
    gaps = default_timestamp_series.get_gaps(
        end=end,
    )
    assert gaps == [end]


def test_timestamp_series_resample_without_unit():
    ts = TimestampSeries(
        pd.Series(np.ones(48),
                  index=pd.date_range('2020-01-01', freq='H', periods=48, tz='CET')
                  ),
        time_zone='CET', freq='H',
    )
    ts = ts.resample('D', 'sum')
    assert ts.timestamps == [
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 1)),
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 2)),
    ]
    assert ts.freq == pd.offsets.Day()
    assert ts.values == [24., 24.]
    assert ts.unit is None
    assert ts.time_zone == pytz.timezone('CET')


def test_timestamp_series_resample_str_method():
    ts = TimestampSeries(
        pd.Series(np.ones(48),
                  index=pd.date_range('2020-01-01', freq='H', periods=48, tz='CET')
                  ),
        time_zone='CET', freq='H', unit='m',
    )
    ts = ts.resample('D', 'sum')
    assert ts.timestamps == [
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 1)),
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 2)),
    ]
    assert ts.freq == pd.offsets.Day()
    assert ts.values == [24., 24.]
    assert ts.unit == ureg.Unit('m')
    assert ts.time_zone == pytz.timezone('CET')


def test_timestamp_series_resample_missing_values():
    ts = TimestampSeries(
        pd.Series(list(np.ones(12)) + list(np.repeat(np.nan, 12)),
                  index=pd.date_range('2020-01-01', freq='H', periods=24, tz='CET')
                  ),
        time_zone='CET', freq='H', unit='m',
    )
    ts = ts.resample('D', 'sum')
    assert ts.timestamps == [
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 1)),
    ]
    assert ts.freq == pd.offsets.Day()
    assert ts.values == [12.]
    assert ts.unit == ureg.Unit('m')
    assert ts.time_zone == pytz.timezone('CET')


def test_timestamp_series_resample_with_function_and_nan():
    ts = TimestampSeries(
        pd.Series(np.repeat(np.nan, 48),
                  index=pd.date_range('2020-01-01', freq='H', periods=48, tz='CET')
                  ),
        time_zone='CET', freq='H', unit='m',
    )
    ts = ts.resample('D', np.mean)
    assert ts.timestamps == [
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 1)),
        pytz.timezone('CET').localize(dt.datetime(2020, 1, 2)),
    ]
    assert ts.freq == pd.offsets.Day()
    assert np.array_equal(ts.values, [np.nan, np.nan], equal_nan=True)
    assert ts.unit == ureg.Unit('m')
    assert ts.time_zone == pytz.timezone('CET')
