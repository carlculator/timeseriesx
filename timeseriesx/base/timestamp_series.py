import collections
import copy
import numbers
import warnings

import numpy as np
import pandas as pd
from pint_pandas import PintArray

from timeseriesx.validation.timestamp_index import (
    index_is_datetime,
    index_is_sorted,
)
from timeseriesx.base.base_time_series import BaseTimeSeries
from timeseriesx.mixins.frequency import FrequencyMixin
from timeseriesx.mixins.time_zone import TimeZoneMixin
from timeseriesx.mixins.unit import UnitMixin


class TimestampSeries(UnitMixin, TimeZoneMixin, FrequencyMixin, BaseTimeSeries):

    @staticmethod
    def create_null_timeseries(start, end, freq, unit=None, time_zone='infer'):
        """
        create a `TimestampSeries`-object from `start` to `end``with NaN-values

        :param str/datetime.datetime/pandas.Timestamp start: the start timestamp of the
            series (included)
        :param str/datetime.datetime/pandas.Timestamp end: the end timestamp of the
            series (included)
        :param str/datetime.timedelta/pandas.Offset/pandas.Timedelta freq:
            the frequency of the timestamp series, `pandas offset aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_
            supported
        :param str/pint.Unit unit: the unit of the series's values, many string
            representations of common units are supported, such as `m`, `s`, `kg`
            and many more
        :param str/tzinfo time_zone: the name of the time zone, (see `IANA <https://www.iana.org/time-zones>`_)
            or a tzinfo-object, pass `'infer'` if you want the time zone to be derived
            from `start` and `end`
        :return: a new TimestampSeries-object
        :rtype:TimestampSeries
        """
        return TimestampSeries.create_constant_timeseries(
            start, end, np.NaN, freq, unit, time_zone=time_zone)

    @staticmethod
    def create_constant_timeseries(start, end, value, freq, unit=None,
                                   time_zone='infer'):
        """
        create a `TimestampSeries`-object from `start` to `end` with constant value

        :param str/datetime.datetime/pandas.Timestamp start: the start timestamp of the
            series (included)
        :param str/datetime.datetime/pandas.Timestamp end: the end timestamp of the
            series (included)
        :param int/float value: the constant value for each element
        :param str/datetime.timedelta/pandas.Offset/pandas.Timedelta freq:
            the frequency of the timestamp series, `pandas offset aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_
            supported
        :param str/pint.Unit unit: the unit of the series's values, many string
            representations of common units are supported, such as `m`, `s`, `kg`
            and many more
        :param str/tzinfo time_zone: the name of the time zone, (see `IANA <https://www.iana.org/time-zones>`_)
            or a tzinfo-object, pass `'infer'` if you want the time zone to be derived
            from `start` and `end`
        :return: a new TimestampSeries-object
        :rtype: TimestampSeries
        """
        index = pd.date_range(start, end, freq=freq)
        series = pd.Series([value] * len(index), index=index)
        return TimestampSeries.create_from_pd_series(series, freq=freq, unit=unit,
                                                     time_zone=time_zone)

    @staticmethod
    def create_from_lists(timestamps, values, freq='infer', unit=None,
                          time_zone='infer'):
        """
        create a `TimestampSeries`-object from a list of timestamps and values matched
        by their index

        :param list timestamps: the timestamps of the series
        :param list values: the values of the series
        :param str/datetime.timedelta/pandas.Offset/pandas.Timedelta freq:
            the frequency of the timestamp series, `pandas offset aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_
            supported, pass `'infer'` if you want the frequency to be derived by the
            timestamps
        :param str/pint.Unit unit: the unit of the series's values, many string
            representations of common units are supported, such as `m`, `s`, `kg`
            and many more
        :param str/tzinfo time_zone: the name of the time zone, (see `IANA <https://www.iana.org/time-zones>`_)
            or a tzinfo-object, pass `'infer'` if you want the time zone to be derived
            from `start` and `end`
        :return: a new TimestampSeries-object
        :rtype: TimestampSeries
        """
        if not len(timestamps) == len(values):
            raise ValueError('lengths of timestamps and values do not not match')
        tuples = list(zip(timestamps, values))
        return TimestampSeries.create_from_tuples(tuples, freq=freq, unit=unit,
                                                  time_zone=time_zone)

    @staticmethod
    def create_from_tuples(tuples, freq='infer', unit=None, time_zone='infer'):
        """
        create a `TimestampSeries`-object from a list of tuples of timestamps and values

        :param list tuples: list of tuples holding timestamp and value
        :param str/datetime.timedelta/pandas.Offset/pandas.Timedelta freq:
            the frequency of the timestamp series, `pandas offset aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_
            supported, pass `'infer'` if you want the frequency to be derived by the
            timestamps
        :param str/pint.Unit unit: the unit of the series's values, many string
            representations of common units are supported, such as `m`, `s`, `kg`
            and many more
        :param str/tzinfo time_zone: the name of the time zone, (see `IANA <https://www.iana.org/time-zones>`_)
            or a tzinfo-object, pass `'infer'` if you want the time zone to be derived
            from `start` and `end`
        :return: a new TimestampSeries-object
        :rtype: TimestampSeries
        """
        dictionary = {k: v for k, v in tuples}
        return TimestampSeries.create_from_dict(dictionary, freq=freq, unit=unit,
                                                time_zone=time_zone)

    @staticmethod
    def create_from_dict(dictionary, freq='infer', unit=None, time_zone='infer'):
        """
        create a `TimestampSeries`-object from a dict timestamps as keys and values as
        values

        :param list tuples: list of tuples holding timestamp and value
        :param str/datetime.timedelta/pandas.Offset/pandas.Timedelta freq:
            the frequency of the timestamp series, `pandas offset aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_
            supported, pass `'infer'` if you want the frequency to be derived by the
            timestamps
        :param str/pint.Unit unit: the unit of the series's values, many string
            representations of common units are supported, such as `m`, `s`, `kg`
            and many more
        :param str/tzinfo time_zone: the name of the time zone, (see `IANA <https://www.iana.org/time-zones>`_)
            or a tzinfo-object, pass `'infer'` if you want the time zone to be derived
            from `start` and `end`
        :return: a new TimestampSeries-object
        :rtype: TimestampSeries
        """
        series = pd.Series(dictionary)
        return TimestampSeries.create_from_pd_series(series, freq=freq, unit=unit,
                                                     time_zone=time_zone)

    @classmethod
    def create_from_pd_series(cls, series, freq='infer', unit=None, time_zone='infer'):
        """
        create a `TimestampSeries`-object from a pandas `Series` with `DatetimeIndex`

        :param pandas.Series: a pandas series-object with `DatetimeIndex`
        :param str/datetime.timedelta/pandas.Offset/pandas.Timedelta freq:
            the frequency of the timestamp series, `pandas offset aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_
            supported, pass `'infer'` if you want the frequency to be derived by the
            timestamps
        :param str/pint.Unit unit: the unit of the series's values, many string
            representations of common units are supported, such as `m`, `s`, `kg`
            and many more
        :param str/tzinfo time_zone: the name of the time zone, (see `IANA <https://www.iana.org/time-zones>`_)
            or a tzinfo-object, pass `'infer'` if you want the time zone to be derived
            from `start` and `end`
        :return: a new TimestampSeries-object
        :rtype: TimestampSeries
        """
        return cls(series, freq=freq, unit=unit, time_zone=time_zone)

    # ------------------------------ constructor ----------------------------- #

    def __init__(self, series, freq=None, unit=None, time_zone=None):
        """
        :param series: a pandas series-object with `DatetimeIndex`
        :param str/datetime.timedelta/pandas.Offset/pandas.Timedelta freq:
            the frequency of the timestamp series or None, `pandas offset aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_
            supported, pass `'infer'` if you want the frequency to be derived by the
            timestamps
        :param str/pint.Unit unit: the unit of the series's values or None, many string
            representations of common units are supported, such as `'m'`, `'s'`, `'kg'`
            and many more
        :param str/tzinfo time_zone: the name of the time zone or None (see `IANA <https://www.iana.org/time-zones>`_)
            or a tzinfo-object, pass `'infer'` if you want the time zone to be derived
            from `start` and `end`
        """
        self._series = series
        self._validate()
        super().__init__(freq=freq, unit=unit, time_zone=time_zone)

    # ------------------------------ properties ------------------------------ #

    @property
    def values(self):
        # would like to use self._series.pint.magnitude.tolist() here, but
        # it is not updated, when updating self._series, to be reported @pint_array
        return list(map(lambda value: getattr(value, 'magnitude', value),
                        self._series.values))

    @property
    def timestamps(self):
        return self._series.index.to_pydatetime().tolist()

    @property
    def first(self):
        if self.empty:
            raise ValueError('empty series')
        return self.timestamps[0], self.values[0]

    @property
    def last(self):
        if self.empty:
            raise ValueError('empty series')
        return self.timestamps[-1], self.values[-1]

    @property
    def start(self):
        if self.empty:
            raise ValueError('empty series')
        else:
            return self._series.index[0].to_pydatetime()

    @property
    def end(self):
        if self.empty:
            raise ValueError('empty series')
        else:
            return self._series.index[-1].to_pydatetime()

    @property
    def time_range(self):
        return self.start, self.end

    # ---------------------------- functionality ----------------------------- #

    def append(self, value):
        """
        append a new value to a series with frequency

        :param float/int value: the value to append
        :return: the series with the new appended value
        :rtype: TimestampSeries
        """
        if not self.freq:
            raise ValueError('cannot append to series without freq')
        if self.empty:
            raise ValueError('cannot append to empty series, '
                             'use __setitem__: ts[timestamp] = value instead')
        values = [value]
        if self.unit:
            values = PintArray(values, dtype=self.unit)
        self._series = self._series.append(
            pd.Series(values,
                      index=[self._series.index.shift(periods=1, freq=self.freq)[-1]])
        )
        return self

    def prepend(self, value):
        """
        prepend a new value to a series with frequency

        :param float/int value: the value to prepend
        :return: the series with the new prepended value
        :rtype: TimestampSeries
        """
        if not self.freq:
            raise ValueError('cannot prepend to series without freq')
        if self.empty:
            raise ValueError('cannot prepend to empty series, '
                             'use __setitem__: ts[timestamp] = value instead')
        values = [value]
        if self.unit:
            values = PintArray(values, dtype=self.unit)
        self._series = (
            pd.Series(values,
                      index=[self._series.index.shift(periods=1, freq=-self.freq)[0]])
            .append(self._series)
        )
        return self

    def join(self, other_ts, fit=True):
        raise NotImplementedError()

    # --------------------------------- cast --------------------------------- #

    def as_tuples(self):
        return list(zip(self.timestamps, self.values))

    def as_dict(self, ordered=False):
        dict_class = dict if not ordered else collections.OrderedDict
        return self.as_pd_series().to_dict(into=dict_class)

    def as_time_period_series(self, align_left=True):
        raise NotImplementedError()

    # ---------------------------- magic methods  ---------------------------- #

    def __str__(self):
        return f"Time zone: {str(self._time_zone)}, " \
               f"Freq: {getattr(self._freq, 'freqstr', '')}, " \
               f"Unit: {str(self._unit or None)}\n" \
               f"{str(self._series)}"

    def __repr__(self):
        return "{klass}(series=Series(PintArray({values}, dtype={unit}), " \
               "index=DatetimeIndex({index}, tz={tz}, freq={freq})), " \
               "freq={freq}, unit={unit}, time_zone={tz})".format(
            klass=self.__class__.__name__,
            values=self.values,
            index=repr(self._series.index.values),
            tz=f"'{self.time_zone}'" if self.time_zone else None,
            freq=f"'{self.freq.freqstr}'" if self.freq else None,
            unit=f"'{self.unit or ''}'")

    def __eq__(self, other):
        if not isinstance(other, TimestampSeries):
            return False
        return (
            self.tz_equal(other.time_zone) and
            self.freq == other.freq and
            self.unit == other.unit and
            self.timestamps == other.timestamps and
            self.values == other.values
        )

    def __setitem__(self, key, value):
        raise NotImplementedError()

    # ---------------------------- calculations ------------------------------ #

    def _basic_calc(self, operation, other, *args, **kwargs):
        if isinstance(other, TimestampSeries):
            return self._basic_calc_time_series(operation, other, **kwargs)
        elif isinstance(other, pd.Series):
            return self._basic_calc_pd_series(operation, other, **kwargs)
        elif isinstance(other, (collections.Sequence, np.ndarray)):
            return self._basic_calc_collection(operation, other)
        else:
            return self._basic_calc_scalar(operation, other)

    def _basic_calc_time_series(self, operation, other, **kwargs):
        tmp_series = copy.deepcopy(self)
        if self.freq != other.freq:
            raise ValueError("The time series have different frequencies")
        if not self.tz_equal(other.time_zone):
            raise ValueError("The time series have different time zones")
        if not self.unit == other.unit:
            raise ValueError("The time series have different units")
        if not self._series.index.equals(other._series.index):
            warnings.warn("timestamps do not match, values are auto-filled")
        tmp_series._series = getattr(tmp_series._series, operation)(
            other._series, **kwargs)
        return tmp_series

    def _basic_calc_pd_series(self, operation, other, **kwargs):
        tmp_series = copy.deepcopy(self)
        if not isinstance(other.index, pd.DatetimeIndex):
            raise ValueError("The series has no proper DatetimeIndex")
        if not self.tz_equal(other.index.tz):
            raise ValueError("The series have different time zones")
        if not all(map(lambda x: isinstance(x, numbers.Number), other)):
            raise ValueError("sequence contains non-numeric values")
        if not self._series.index.equals(other.index):
            warnings.warn("timestamps do not match, values are auto-filled")
        tmp_series._series = getattr(tmp_series._series, operation)(other, **kwargs)
        return tmp_series

    def _basic_calc_collection(self, operation, other):
        tmp_series = copy.deepcopy(self)
        if len(other) != len(self):
            raise ValueError("sequence has different length")
        if not all(map(lambda x: isinstance(x, numbers.Number), other)):
            raise ValueError("sequence contains non-numeric values")
        tmp_series._series = getattr(tmp_series._series, operation)(other)
        return tmp_series

    def _basic_calc_scalar(self, operation, other):
        tmp_series = copy.deepcopy(self)
        if not isinstance(other, numbers.Number):
            raise ValueError('value is not numeric')
        tmp_series._series = getattr(tmp_series._series, operation)(other)
        return tmp_series

    # ---------------------------- validation ------------------------------- #

    def validate_all(self):
        self._validate()
        super()._validate_all()

    def _validate(self):
        index_is_datetime(self._series)
        index_is_sorted(self._series)
