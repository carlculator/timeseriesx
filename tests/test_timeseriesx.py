import datetime as dt
from collections import OrderedDict

import dateutil
import numpy as np
import pandas as pd
import pytest
import pytz
from dateutil.tz import tzutc
from pint import DimensionalityError
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
        pd.Series(PintArray(np.arange(n), dtype=ureg.parse_units('m')),
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


def test_timestamp_series_create_from_lists():
    timestamps = [
        dt.datetime(2020, 3, 1, 15, 0, 0),
        dt.datetime(2020, 3, 1, 16, 0, 0),
        dt.datetime(2020, 3, 1, 17, 0, 0),
    ]
    values = [-1, 0, 1]
    ts = TimestampSeries.create_from_lists(timestamps, values)
    assert ts.freq == pd.offsets.Hour()
    assert ts.time_zone is None
    assert ts.unit is None


def test_timestamp_series_create_from_lists_mismatching_length():
    timestamps = [
        dt.datetime(2020, 3, 1, 15, 0, 0),
        dt.datetime(2020, 3, 1, 16, 0, 0),
    ]
    values = [-1,]
    with pytest.raises(ValueError):
        TimestampSeries.create_from_lists(timestamps, values)


def test_timestamp_series_properties_time_range_default(default_timestamp_series):
    assert default_timestamp_series.time_range == (
           pd.Timestamp('2020-01-01T00:00:00').tz_localize('CET').to_pydatetime(),
           pd.Timestamp('2020-01-03T00:00:00').tz_localize('CET').to_pydatetime()
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
           pd.Timestamp('2020-01-01T00:00:00').tz_localize('CET').to_pydatetime()


def test_timestamp_series_properties_start_empty(empty_timestamp_series):
    with pytest.raises(ValueError):
        empty_timestamp_series.start


def test_timestamp_series_properties_end_default(default_timestamp_series):
    assert default_timestamp_series.end == \
           pd.Timestamp('2020-01-03T00:00:00').tz_localize('CET').to_pydatetime()


def test_timestamp_series_properties_end_empty(empty_timestamp_series):
    with pytest.raises(ValueError):
        empty_timestamp_series.end


def test_timestamp_series_properties_timestamps_default(default_timestamp_series):
    assert default_timestamp_series.timestamps == [
        pd.Timestamp('2020-01-01T00:00:00').tz_localize('CET').to_pydatetime(),
        pd.Timestamp('2020-01-02T00:00:00').tz_localize('CET').to_pydatetime(),
        pd.Timestamp('2020-01-03T00:00:00').tz_localize('CET').to_pydatetime()
    ]


def test_timestamp_series_properties_timestamps_empty(empty_timestamp_series):
    assert empty_timestamp_series.timestamps == []


def test_timestamp_series_properties_values_default(default_timestamp_series):
    assert default_timestamp_series.values == [0., 1., 2.]


def test_timestamp_series_properties_values_empty(empty_timestamp_series):
    assert empty_timestamp_series.values == []


def test_timestamp_series_properties_first_default(default_timestamp_series):
    assert default_timestamp_series.first == (
        pd.Timestamp('2020-01-01T00:00:00').tz_localize('CET').to_pydatetime(), 0.
    )


def test_timestamp_series_properties_first_empty(empty_timestamp_series):
    with pytest.raises(ValueError):
        empty_timestamp_series.first


def test_timestamp_series_properties_last_default(default_timestamp_series):
    assert default_timestamp_series.last == (
        pd.Timestamp('2020-01-03T00:00:00').tz_localize('CET').to_pydatetime(), 2.
    )


def test_timestamp_series_properties_last_empty(empty_timestamp_series):
    with pytest.raises(ValueError):
        empty_timestamp_series.last


def test_timestamp_series_as_tuples_default(default_timestamp_series):
    assert default_timestamp_series.as_tuples() == [
        (pd.Timestamp('2020-01-01T00:00:00').tz_localize('CET').to_pydatetime(), 0.),
        (pd.Timestamp('2020-01-02T00:00:00').tz_localize('CET').to_pydatetime(), 1.),
        (pd.Timestamp('2020-01-03T00:00:00').tz_localize('CET').to_pydatetime(), 2.),
    ]


def test_timestamp_series_as_tuples_empty(empty_timestamp_series):
    assert empty_timestamp_series.as_tuples() == []


def test_timestamp_series_as_dict_default(default_timestamp_series):
    assert default_timestamp_series.as_dict() == {
        pd.Timestamp('2020-01-01T00:00:00').tz_localize('CET').to_pydatetime(): 0.,
        pd.Timestamp('2020-01-02T00:00:00').tz_localize('CET').to_pydatetime(): 1.,
        pd.Timestamp('2020-01-03T00:00:00').tz_localize('CET').to_pydatetime(): 2.,
    }


def test_timestamp_series_as_dict_ordered_default(default_timestamp_series):
    assert default_timestamp_series.as_dict(ordered=True) == OrderedDict({
        pd.Timestamp('2020-01-01T00:00:00').tz_localize('CET').to_pydatetime(): 0.,
        pd.Timestamp('2020-01-02T00:00:00').tz_localize('CET').to_pydatetime(): 1.,
        pd.Timestamp('2020-01-03T00:00:00').tz_localize('CET').to_pydatetime(): 2.,
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
        pd.Series([], index=pd.DatetimeIndex([])),
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
    assert "Freq: " in str(empty_timestamp_series)
    assert "Unit: " in str(empty_timestamp_series)
    assert "Time zone: " in str(empty_timestamp_series)


def test_timestamp_series_append_default(default_timestamp_series):
    assert default_timestamp_series.append(3.).last == (
        pd.Timestamp('2020-01-04T00:00:00').tz_localize('CET').to_pydatetime(), 3.
    )


def test_timestamp_series_append_empty():
    ts = TimestampSeries(pd.Series([], dtype=float, index=pd.DatetimeIndex([])), freq='10s')
    with pytest.raises(ValueError):
        ts.append(1.)


def test_timestamp_series_append_no_freq():
    ts = TimestampSeries(pd.Series([1.], dtype=float, index=pd.DatetimeIndex(['2020-01-01'])))
    with pytest.raises(ValueError):
        ts.append(1.)


def test_timestamp_series_prepend_default(default_timestamp_series):
    assert default_timestamp_series.prepend(-1).first == (
        pd.Timestamp('2019-12-31T00:00:00').tz_localize('CET').to_pydatetime(), -1.
    )


def test_timestamp_series_prepend_empty(empty_timestamp_series):
    with pytest.raises(ValueError):
        empty_timestamp_series.prepend(1.)


def test_create_timestamp_series_inferred_time_zone_none():
    timestamps = [
        pd.Timestamp('2020-01-01T00:00:00').tz_localize(None),
        pd.Timestamp('2020-01-03T00:00:00').tz_localize(None),
    ]
    values = [0., 1.]
    ts = TimestampSeries.create_from_lists(timestamps, values, time_zone='infer')
    assert ts.time_zone is None
    assert ts._series.index.tzinfo is None


def test_create_timestamp_series_inferred_time_zone_valid():
    timestamps = [
        pd.Timestamp('2020-01-01T00:00:00+01:00'),
        pd.Timestamp('2020-01-03T00:00:00+01:00'),
    ]
    values = [0., 1.]
    ts = TimestampSeries.create_from_lists(timestamps, values, time_zone='infer')
    assert ts.time_zone == pytz.FixedOffset(60)


def test_create_timestamp_series_inferred_time_zone_inconsistent():
    timestamps = [
        pd.Timestamp('2020-01-01T00:00:00').tz_localize('Europe/London'),
        pd.Timestamp('2020-01-03T00:00:00').tz_localize('Europe/Stockholm'),
    ]
    values = [0., 1.]
    with pytest.raises(ValueError):
        TimestampSeries.create_from_lists(timestamps, values, time_zone='infer')


def test_create_timestamp_series_inferred_time_zone_valid():
    tz_name = 'EST'
    timestamps = [
        pd.Timestamp('2020-01-01T00:00:00').tz_localize(tz_name),
        pd.Timestamp('2020-01-03T00:00:00').tz_localize(tz_name),
    ]
    values = [0., 1.]
    ts = TimestampSeries.create_from_lists(timestamps, values, time_zone='infer')
    assert ts.time_zone is pytz.timezone(tz_name)
    assert ts._series.index.tzinfo is pytz.timezone(tz_name)


def test_create_timestamp_series_inferred_freq_by_pandas():
    timestamps = [
        pd.Timestamp('2020-01-01T00:00:00').tz_localize(None),
        pd.Timestamp('2020-01-03T00:00:00').tz_localize(None),
    ]
    values = [0., 1.]
    ts = TimestampSeries.create_from_lists(timestamps, values, freq='infer')
    assert ts.freq == pd.offsets.Day(2)
    assert ts._series.index.freq == pd.offsets.Day(2)


def test_create_timestamp_series_inferred_freq_invalid():
    timestamps = [
        pd.Timestamp('2020-01-01T00:00:00').tz_localize(None),
        pd.Timestamp('2020-01-02T22:00:00').tz_localize(None),
        pd.Timestamp('2020-01-03T13:30:00').tz_localize(None),
    ]
    values = [0., 1., 2.]
    with pytest.raises(ValueError):
        TimestampSeries.create_from_lists(timestamps, values, freq='infer')


def test_create_timestamp_series_inferred_freq_valid():
    ts = TimestampSeries(series=pd.Series([0., 1., 2.], index=[
        pd.Timestamp('2020-01-01T00:00:00').tz_localize(None).to_pydatetime(),
        pd.Timestamp('2020-01-02T00:00:00').tz_localize(None).to_pydatetime(),
        pd.Timestamp('2020-01-03T00:00:00').tz_localize(None).to_pydatetime(),
    ]), freq='infer')
    assert ts.freq == pd.offsets.Day()
    assert ts._series.index.freq == pd.offsets.Day()


def test_create_timestamp_series_mismatching_time_zone_1():
    ts = TimestampSeries(series=pd.Series([0., 1., 2.], index=[
        pd.Timestamp('2020-01-01T00:00:00').tz_localize(None).to_pydatetime(),
        pd.Timestamp('2020-01-02T00:00:00').tz_localize(None).to_pydatetime(),
        pd.Timestamp('2020-01-03T00:00:00').tz_localize(None).to_pydatetime(),
    ]), time_zone='UTC')
    assert ts.time_zone == pytz.UTC
    assert ts.start == pd.Timestamp('2020-01-01T00:00:00')\
                       .tz_localize('UTC')\
                       .to_pydatetime()


def test_create_timestamp_series_mismatching_time_zone_2():
    ts = TimestampSeries(series=pd.Series([0., 1., 2.], index=[
        pd.Timestamp('2020-01-01T00:00:00').tz_localize('Europe/Berlin').to_pydatetime(),
        pd.Timestamp('2020-01-02T00:00:00').tz_localize('Europe/Berlin').to_pydatetime(),
        pd.Timestamp('2020-01-03T00:00:00').tz_localize('Europe/Berlin').to_pydatetime(),
    ]), time_zone='UTC')
    assert ts.time_zone == pytz.timezone('UTC')
    assert ts.start == pd.Timestamp('2020-01-01T00:00:00')\
                       .tz_localize('Europe/Berlin').tz_convert('UTC')\
                       .to_pydatetime()


def test_create_timestamp_series_mismatching_freq():
    timestamps = [
        pd.Timestamp('2020-01-01T00:00:00'),
        pd.Timestamp('2020-01-03T00:00:00'),
    ]
    values = [0., 1.]
    with pytest.raises(ValueError):
        TimestampSeries.create_from_lists(timestamps, values, freq='10Min')


def test_create_timestamp_series_mismatching_compatible_unit():
    ts = (
        TimestampSeries(series=pd.Series(PintArray([0., 1., 2.], dtype='km'), index=[
            pd.Timestamp('2020-01-01T00:00:00').to_pydatetime(),
            pd.Timestamp('2020-01-02T00:00:00').to_pydatetime(),
            pd.Timestamp('2020-01-03T00:00:00').to_pydatetime(),
        ]), unit='m')
    )
    assert ts.unit == ureg.Unit('m')
    assert ts.values == [0., 1000., 2000.]


def test_create_timestamp_series_mismatching_incompatible_unit():
    with pytest.raises(DimensionalityError):
        TimestampSeries(series=pd.Series(PintArray([0., 1., 2.], dtype='m'), index=[
            pd.Timestamp('2020-01-01T00:00:00').to_pydatetime(),
            pd.Timestamp('2020-01-02T00:00:00').to_pydatetime(),
            pd.Timestamp('2020-01-03T00:00:00').to_pydatetime(),
        ]), unit='km/h')


def test_create_timestamp_series_invalid_time_zone():
    illegal_tz = 'Europe/Nantes'
    with pytest.raises(UnknownTimeZoneError):
        TimestampSeries(series=pd.Series([0., 1., 2.], index=[
            pd.Timestamp('2020-01-01T00:00:00').to_pydatetime(),
            pd.Timestamp('2020-01-02T00:00:00').to_pydatetime(),
            pd.Timestamp('2020-01-03T00:00:00').to_pydatetime(),
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
            pd.Timestamp('2020-01-03T00:00:00').to_pydatetime(),
            pd.Timestamp('2020-01-02T00:00:00').to_pydatetime(),
            pd.Timestamp('2020-01-01T00:00:00').to_pydatetime(),
        ]))


def test_create_timestamp_series_valid_unit_str():
    ts = TimestampSeries(series=pd.Series([], index=pd.DatetimeIndex([])),
                         unit='second')
    assert ts.unit == ureg.second


def test_create_timestamp_series_valid_unit_obj():
    ts = TimestampSeries(series=pd.Series([], index=pd.DatetimeIndex([])),
                         unit=ureg.second)
    assert ts.unit == ureg.second


def test_create_timestamp_series_valid_freq_str():
    ts = TimestampSeries(series=pd.Series([], index=pd.DatetimeIndex([])),
                         freq='30Min')
    assert ts.freq == pd.offsets.Minute(30)


def test_create_timestamp_series_valid_freq_timedelta():
    ts = TimestampSeries(series=pd.Series([], index=pd.DatetimeIndex([])),
                         freq=pd.Timedelta(minutes=30))
    assert ts.freq == pd.offsets.Minute(30)


def test_create_timestamp_series_valid_freq_offset():
    ts = TimestampSeries(series=pd.Series([], index=pd.DatetimeIndex([])),
                         freq=pd.offsets.Minute(30))
    assert ts.freq == pd.offsets.Minute(30)


def test_create_timestamp_series_valid_timezone_str():
    ts = TimestampSeries(series=pd.Series([], index=pd.DatetimeIndex([])),
                         time_zone='utc')
    assert ts.time_zone == pytz.UTC


def test_create_timestamp_series_valid_timezone_obj_pytz():
    ts = TimestampSeries(series=pd.Series([], index=pd.DatetimeIndex([])),
                         time_zone=pytz.UTC)
    assert ts.time_zone == pytz.UTC


def test_create_timestamp_series_valid_timezone_obj_dateutil():
    ts = TimestampSeries(series=pd.Series([], index=pd.DatetimeIndex([])),
                         time_zone=dateutil.tz.tzutc())
    assert isinstance(ts.time_zone, tzutc)


def test_timestamp_series_add_timestamp_series_different_freq(default_timestamp_series):
    add_ts = TimestampSeries(
        pd.Series(PintArray(np.arange(3), dtype=ureg.parse_units('m')),
                  index=pd.date_range('2020-01-01', freq='H', periods=3, tz='CET')
                  ),
        time_zone='CET', unit='m', freq='H'
    )
    with pytest.raises(ValueError):
        default_timestamp_series + add_ts


def test_timestamp_series_add_timestamp_series_different_tz(default_timestamp_series):
    add_ts = TimestampSeries(
        pd.Series(PintArray(np.arange(3), dtype=ureg.parse_units('m')),
                  index=pd.date_range('2020-01-01', freq='D', periods=3, tz='UTC')
                  ),
        time_zone='UTC', unit='m', freq='D'
    )
    result = default_timestamp_series + add_ts
    assert default_timestamp_series._series.index.union(
        add_ts._series.index).tolist() == result.timestamps
    assert result.values == [0., 0., 1., 1., 2., 2.]


def test_timestamp_series_add_timestamp_series_different_unit(default_timestamp_series):
    add_ts = TimestampSeries(
        pd.Series(PintArray(np.arange(3), dtype=ureg.parse_units('kg')),
                  index=pd.date_range('2020-01-01', freq='D', periods=3, tz='CET')
                  ),
        time_zone='CET', unit='kg', freq='D'
    )
    with pytest.raises(ValueError):
        default_timestamp_series + add_ts


def test_timestamp_series_add_timestamp_series_different_index(default_timestamp_series):
    add_ts = TimestampSeries(
        pd.Series(PintArray(np.arange(3), dtype=ureg.parse_units('m')),
                  index=pd.date_range('2020-01-02', freq='D', periods=3, tz='CET')
                  ),
        time_zone='CET', unit='m', freq='D'
    )
    result_ts = default_timestamp_series + add_ts
    assert result_ts.first == (
        pd.Timestamp('2020-01-01').tz_localize('CET'), 0.
    )
    assert result_ts.values == [0., 1., 3., 2.]
    assert result_ts.last == (
        pd.Timestamp('2020-01-04').tz_localize('CET'), 2.
    )


def test_timestamp_series_add_pandas_series():
    ts = TimestampSeries(
        pd.Series(np.arange(3),
                  index=pd.date_range('2020-01-01', freq='D', periods=3, tz='UTC')
                  ),
        time_zone='UTC', freq='D'
    )
    pd_series = pd.Series(np.arange(3),
                          index=pd.date_range('2020-01-02',
                                              freq='D', periods=3, tz='UTC'))
    result_ts = ts + pd_series
    assert result_ts.first == (
        pd.Timestamp('2020-01-01').tz_localize('UTC'), 0.
    )
    assert result_ts.values == [0., 1., 3., 2.]
    assert result_ts.last == (
        pd.Timestamp('2020-01-04').tz_localize('UTC'), 2.
    )


def test_timestamp_series_add_list_error():
    ts = TimestampSeries(
        pd.Series(np.arange(3),
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
    assert result_ts.values == [0., 2., 4.]


def test_timestamp_series_add_pint_array(default_timestamp_series):
    result_ts = default_timestamp_series + PintArray([0., 100., 200.], dtype='cm')
    assert result_ts.values == [0., 2., 4.]
    assert result_ts.unit == ureg.Unit('m')


def test_timestamp_series_multiply_pint_array(default_timestamp_series):
    result_ts = default_timestamp_series * PintArray([1., 1., 1.], dtype='m')
    assert result_ts.values == [0., 1., 2.]
    assert result_ts.unit == ureg.Unit('m^2')


def test_timestamp_series_add_scalar():
    ts = TimestampSeries(
        pd.Series(np.arange(3),
                  index=pd.date_range('2020-01-01', freq='D', periods=3, tz='UTC')
                  ),
        time_zone='UTC', freq='D'
    )
    result_ts = ts + 1
    assert result_ts.values == [1., 2., 3.]


def test_timestamp_series_multiply_scalar(default_timestamp_series):
    result_ts = default_timestamp_series * 2
    assert result_ts.values == [0., 2., 4.]


@pytest.mark.skip('unit should be qm') # Todo: to be reported @pint_pandas
def test_timestamp_series_multiply_pint_scalar(default_timestamp_series):
    ts = TimestampSeries(
        pd.Series(np.arange(3),
                  index=pd.date_range('2020-01-01', freq='D', periods=3)
                  ),
        unit='m'
    )
    result_ts = ts * (2 * ureg.Unit('m'))
    assert result_ts.values == [0., 2., 4.]
    assert result_ts.unit == ureg.Unit('m')


def test_timestamp_series_div_scalar(default_timestamp_series):
    result_ts = default_timestamp_series / 2
    assert result_ts.values == [0., .5, 1.]


@pytest.mark.skip('should work')  # Todo: to be reported @pint_pandas
def test_timestamp_series_floordiv_scalar(default_timestamp_series):
    result_ts = default_timestamp_series // 2
    assert result_ts.values == [0., 0., 1.]


@pytest.mark.skip('unit should be m')  # Todo: to be reported @pint_pandas
def test_timestamp_series_floordiv_pint_scalar():
    ts = TimestampSeries(
        pd.Series(np.arange(3),
                  index=pd.date_range('2020-01-01', freq='D', periods=3)
                  ),
        unit='m^2'
    )
    result_ts = ts // (2 * ureg.Unit('m'))
    assert result_ts.values == [0., 0., 1.]
    assert result_ts.unit == ureg.Unit('m')


def test_timestamp_series_subtract_list():
    ts = TimestampSeries(
        pd.Series(np.arange(3),
                  index=pd.date_range('2020-01-01', freq='D', periods=3)
                  ))
    result_ts = ts - [0., 1., 2.]
    assert result_ts.values == [0., 0., 0.]


def test_timestamp_series_subtract_pd_series_mismatch():
    ts = TimestampSeries(
        pd.Series(np.arange(3),
                  index=pd.date_range('2020-01-01', freq='D', periods=3)),
    )
    result_ts = ts - pd.Series(np.arange(2),
                               index=pd.date_range('2020-01-01', freq='D', periods=2))
    assert result_ts.timestamps == ts.timestamps
    assert result_ts.values == [0., 0., 2.]


def test_get_item(default_timestamp_series):
    assert default_timestamp_series[0] == 0 * ureg.Unit('m')


def test_loop(default_timestamp_series):
    for idx, item in enumerate(default_timestamp_series):
        assert item == idx * ureg.Unit('m')


def test_eq(default_timestamp_series):
    ts1 = TimestampSeries.create_from_tuples(
        [(pd.Timestamp(2020, 1, 1).tz_localize('UTC').tz_convert('EST'), 1.)],
        unit='km', freq=None,
    )
    ts2 = TimestampSeries.create_from_tuples(
        [(pd.Timestamp(2020, 1, 1).tz_localize('UTC').tz_convert('GMT'), 1000.)],
        unit='m', freq=None,
    )
    assert ts1 == ts2


def test_timestamp_series_convert_unit_from_none():
    ts = TimestampSeries(
        pd.Series(np.arange(3),
                  index=pd.date_range('2020-01-01', freq='D', periods=3, tz='UTC')
                  ),
        time_zone='UTC', freq='D'
    )
    ts = ts.convert_unit('meter')
    assert ts.unit == ureg.parse_units('meter')
    assert ts._series.dtype.units == ts.unit


def test_timestamp_series_convert_unit_to_none(default_timestamp_series):
    ts = default_timestamp_series.convert_unit(None)
    assert ts.unit is None
    assert not hasattr(ts._series.dtype, 'units')


def test_timestamp_series_convert_unit_incompatible(default_timestamp_series):
    with pytest.raises(ValueError):
        default_timestamp_series.convert_unit('liter')


def test_timestamp_series_convert_unit_success(default_timestamp_series):
    ts = default_timestamp_series * 1000
    ts = ts.convert_unit('nautical_mile')
    assert ts.unit == ureg.parse_units('nautical_mile')
    assert np.array(ts.values).round(2).tolist() == [0., 0.54, 1.08]
    assert ts._series.dtype.units == ts.unit


def test_timestamp_series_fill_gaps():
    timestamps = [
        dt.datetime(2020, 3, 1, 15, 0, 0),
        dt.datetime(2020, 3, 1, 16, 0, 0),
        dt.datetime(2020, 3, 1, 17, 0, 0),
    ]
    values = [1, 1, 1]
    ts = TimestampSeries.create_from_lists(timestamps, values, freq=pd.offsets.Hour())
    ts = ts.fill_gaps(timestamps[0] - dt.timedelta(hours=1),
                      timestamps[-1] + dt.timedelta(hours=1), value=2)
    assert len(ts) == 5
    assert ts[timestamps[0] - dt.timedelta(hours=1)] == 2
    assert ts[timestamps[-1] + dt.timedelta(hours=1)] == 2


def test_timestamp_series_get_gaps():
    timestamps = [
        dt.datetime(2020, 3, 1, 15, 0, 0),
        dt.datetime(2020, 3, 1, 16, 0, 0),
        dt.datetime(2020, 3, 1, 17, 0, 0),
    ]
    values = [1, 1, 1]
    ts = TimestampSeries.create_from_lists(timestamps, values, freq=pd.offsets.Hour())
    gaps = ts.get_gaps(timestamps[0] - dt.timedelta(hours=1),
                       timestamps[-1] + dt.timedelta(hours=1))
    assert gaps == [
        timestamps[0] - dt.timedelta(hours=1),
        timestamps[-1] + dt.timedelta(hours=1)
    ]


def test_timestamp_series_resample():
    ts = TimestampSeries(
        pd.Series(np.ones(48),
                  index=pd.date_range('2020-01-01', freq='H', periods=48, tz='CET')
                  ),
        time_zone='CET', freq='H'
    )
    ts = ts.resample('D', 'sum')
    assert ts.freq == pd.offsets.Day()
    assert ts.values == [24, 24]
