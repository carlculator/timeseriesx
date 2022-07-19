from __future__ import annotations

import collections
import copy
import datetime as dt
import numbers
import warnings
from collections.abc import Iterable
from typing import List, Tuple, Union, Dict, Callable

import numpy as np
import pandas as pd
import pint
from pint_pandas import PintArray, PintType

from timeseriesx.base.base_time_series import BaseTimeSeries
from timeseriesx.base.types import (
    TimestampType,
    FreqType,
    UnitType,
    InferableTimeZoneType,
    InferableFreqType,
    TimeZoneType,
    BasicCalcOperationType,
    BasicCalcOperandType,
)
from timeseriesx.mixins.frequency import FrequencyMixin
from timeseriesx.mixins.time_zone import TimeZoneMixin
from timeseriesx.mixins.unit import UnitMixin
from timeseriesx.validation.timestamp_index import (
    index_is_datetime,
    index_is_sorted,
)


class TimestampMismatchWarning(RuntimeWarning):
    """
    warning about implicit handling of mismatching timestamps
    """
    pass


class TimestampSeries(UnitMixin, TimeZoneMixin, FrequencyMixin, BaseTimeSeries):

    @staticmethod
    def create_null_timeseries(
        start: TimestampType,
        end: TimestampType,
        freq: FreqType,
        unit: UnitType = None,
        time_zone: InferableTimeZoneType = 'infer',
    ) -> TimestampSeries:
        """
        create a `TimestampSeries`-object from `start` to `end` with NaN-values

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
        :rtype: TimestampSeries
        """
        return TimestampSeries.create_constant_timeseries(
            start, end, np.NaN, freq, unit, time_zone=time_zone)

    @staticmethod
    def create_constant_timeseries(
        start: TimestampType,
        end: TimestampType,
        value: float,
        freq: FreqType,
        unit: UnitType = None,
        time_zone: InferableTimeZoneType = 'infer',
    ) -> TimestampSeries:
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
    def create_from_lists(
        timestamps: List[TimestampType],
        values: List[float],
        freq: InferableFreqType = 'infer',
        unit: UnitType = None,
        time_zone: InferableTimeZoneType = 'infer',
    ) -> TimestampSeries:
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
            by the timestamps
        :return: a new TimestampSeries-object
        :rtype: TimestampSeries
        """
        if not len(timestamps) == len(values):
            raise ValueError('lengths of timestamps and values do not not match')
        tuples = list(zip(timestamps, values))
        return TimestampSeries.create_from_tuples(tuples, freq=freq, unit=unit,
                                                  time_zone=time_zone)

    @staticmethod
    def create_from_tuples(tuples: List[Tuple[dt.datetime, float]],
                           freq: InferableFreqType = 'infer',
                           unit: UnitType = None,
                           time_zone: TimeZoneType = 'infer') -> TimestampSeries:
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
            by the timestamps
        :return: a new TimestampSeries-object
        :rtype: TimestampSeries
        """
        dictionary = {k: v for k, v in tuples}
        return TimestampSeries.create_from_dict(dictionary, freq=freq, unit=unit,
                                                time_zone=time_zone)

    @staticmethod
    def create_from_dict(
        dictionary: Dict[dt.datetime, float],
        freq: InferableFreqType = 'infer',
        unit: UnitType = None,
        time_zone: InferableTimeZoneType = 'infer',
    ) -> TimestampSeries:
        """
        create a `TimestampSeries`-object from a dict timestamps as keys and values as
        values

        :param dict dictionary: dict with timestamps as keys and timeseries-values as
            dict-values
        :param str/datetime.timedelta/pandas.Offset/pandas.Timedelta freq:
            the frequency of the timestamp series, `pandas offset aliases <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_
            supported, pass `'infer'` if you want the frequency to be derived by the
            timestamps
        :param str/pint.Unit unit: the unit of the series's values, many string
            representations of common units are supported, such as `m`, `s`, `kg`
            and many more
        :param str/tzinfo time_zone: the name of the time zone, (see `IANA <https://www.iana.org/time-zones>`_)
            or a tzinfo-object, pass `'infer'` if you want the time zone to be derived
            by the timestamps
        :return: a new TimestampSeries-object
        :rtype: TimestampSeries
        """
        series = pd.Series(dictionary)
        return TimestampSeries.create_from_pd_series(series, freq=freq, unit=unit,
                                                     time_zone=time_zone)

    @staticmethod
    def create_from_pd_series(
        series: pd.Series,
        freq: InferableFreqType = 'infer',
        unit: UnitType = None,
        time_zone: InferableTimeZoneType = 'infer'
    ) -> TimestampSeries:
        """
        create a `TimestampSeries`-object from a pandas `Series` with `DatetimeIndex`

        :param pandas.Series series: a pandas series-object with `DatetimeIndex`
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
        return TimestampSeries(series, freq=freq, unit=unit, time_zone=time_zone)

    # ------------------------------ constructor ----------------------------- #

    def __init__(self,
                 series: pd.Series,
                 freq: InferableFreqType = None,
                 unit: UnitType = None,
                 time_zone: InferableTimeZoneType = None):
        """
        :param pandas.Series series: a pandas series-object with `DatetimeIndex`
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
        self._series: pd.Series = series
        self._validate()
        super().__init__(freq=freq, unit=unit, time_zone=time_zone)

    # ------------------------------ properties ------------------------------ #

    @property
    def values(self) -> List[float]:
        # would like to use self._series.pint.magnitude.tolist() here, but
        # it is not updated, when updating self._series, to be reported @pint_array
        return list(map(lambda value: getattr(value, 'magnitude', value),
                        self._series.values))

    @property
    def timestamps(self) -> List[dt.datetime]:
        return self._series.index.to_pydatetime().tolist()

    @property
    def first(self) -> Tuple[dt.datetime, float]:
        if self.empty:
            raise ValueError('empty series')
        return self.timestamps[0], self.values[0]

    @property
    def last(self) -> Tuple[dt.datetime, float]:
        if self.empty:
            raise ValueError('empty series')
        return self.timestamps[-1], self.values[-1]

    @property
    def start(self) -> dt.datetime:
        if self.empty:
            raise ValueError('empty series')
        else:
            return self._series.index[0].to_pydatetime()

    @property
    def end(self) -> dt.datetime:
        if self.empty:
            raise ValueError('empty series')
        else:
            return self._series.index[-1].to_pydatetime()

    @property
    def time_range(self) -> Tuple[dt.datetime, dt.datetime]:
        return self.start, self.end

    # ---------------------------- functionality ----------------------------- #

    def map(self, func: Callable, dimensionless: bool = True) -> TimestampSeries:
        """
        apply a custom function to each value of the series

        :param function func: a function mapping a scalar to another scalar
        :param bool dimensionless: if set to True, the mapping function takes
            an argument of type Number (no unit, dimensionless). The resulting
            timestamp series will keep the original unit. If set to False,
            the mapping function takes an argument of type pint.Quantity.
            The resulting timestamp series will have the unit of the mapped
            values. Mapping values of one series to different units results in
            an error. Mapping with dimensionless=False will result in a loop
            and therefore perform slower.
        :return: the series with mapped values
        :rtype: TimestampSeries
        """
        if self.empty:
            return self

        if isinstance(self._series.dtype, PintType):
            if dimensionless:
                mapped_values = self._get_magnitude_series().apply(func).values
                self._series = pd.Series(PintArray(mapped_values, dtype=self.unit),
                                         index=self._series.index)
            else:
                mapped_values = list(map(func, self._series.values))
                mapped_unit = mapped_values[0].u
                if any(map(lambda x: x.u != mapped_unit, mapped_values)):
                    raise ValueError("the mapped values do not have the same unit")
                magnitudes = [v.magnitude for v in mapped_values]
                self._series = pd.Series(PintArray(magnitudes, dtype=mapped_unit),
                                         index=self._series.index)
        else:
            self._series = self._series.apply(func)
        return self

    def round(self, decimals: int) -> TimestampSeries:
        """
        round the values of the series

        :param int decimals: no of decimal places to round to
        :return: the series with rounded values
        :rtype: TimestampSeries
        """
        # ToDo: feature request at pint-pandas
        if isinstance(self._series.dtype, PintType):
            rounded_values = self._get_magnitude_series().values.round(decimals)
            self._series = pd.Series(PintArray(rounded_values, dtype=self.unit),
                                     index=self._series.index)
        else:
            self._series = self._series.round(decimals)
        return self

    def append(self, value: float) -> TimestampSeries:
        """
        append a new value to a series with frequency

        :param float value: the value to append
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

    def prepend(self, value: float) -> TimestampSeries:
        """
        prepend a new value to a series with frequency

        :param float value: the value to prepend
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

    def as_tuples(
        self,
        time_zone: bool = True,
        unit: bool = False,
        include_nan: bool = True
    ) -> List[Tuple[dt.datetime, Union[float, pint.Quantity]]]:
        if time_zone:
            timestamps = self.timestamps
        else:
            timestamps = self._series.index.tz_localize(None).to_pydatetime().tolist()
        if unit:
            values = self._series.values.tolist()
        else:
            values = self.values
        tuples = list(zip(timestamps, values))
        if not include_nan:
            tuples = [(t, v) for t, v in tuples if not np.isnan(v)]
        return tuples

    def as_dict(
        self,
        time_zone: bool = True,
        unit: bool = False,
        ordered: bool = False,
        include_nan: bool = True
    ) -> Union[Dict[dt.datetime, Union[float, pint.Quantity]],
               collections.OrderedDict[Dict[dt.datetime, Union[float, pint.Quantity]]]]:
        if ordered:
            return collections.OrderedDict(
                self.as_tuples(time_zone=time_zone, unit=unit, include_nan=include_nan)
            )
        else:
            return dict(
                self.as_tuples(time_zone=time_zone, unit=unit, include_nan=include_nan)
            )

    def as_pd_series(
        self,
        time_zone: bool = True,
        unit: bool = False,
        include_nan: bool = True
    ) -> pd.Series:
        if unit:
            series = self._series
        else:
            series = self._get_magnitude_series()
        if not time_zone:
            series = series.tz_localize(None)
        if not include_nan:
            series = series[series.notnull()]
        return series

    def as_time_period_series(self, align_left: bool = True):
        raise NotImplementedError()

    # ---------------------------- magic methods  ---------------------------- #

    def __str__(self) -> str:
        return f"Time zone: {str(self._time_zone)}, " \
               f"Freq: {getattr(self._freq, 'freqstr', '')}, " \
               f"Unit: {str(self._unit or None)}\n" \
               f"{str(self._series)}"

    def __repr__(self) -> str:
        return "{klass}(series=Series(PintArray({values}, dtype={unit}), " \
               "index={index}), " \
               "freq={freq}, unit={unit}, time_zone={tz})".format(
                    klass=self.__class__.__name__,
                    values=self.values,
                    index=repr(self._series.index),
                    tz=f"'{self.time_zone}'" if self.time_zone else None,
                    freq=f"'{self.freq.freqstr}'" if self.freq else None,
                    unit=f"'{self.unit or ''}'"
               )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TimestampSeries):
            return False
        self_values = self.values
        other_values = other.values
        self_timestamps = self.timestamps
        other_timestamps = other.timestamps
        if self.unit:
            self_values = list(self._series.pint.to_base_units().values)
        if other.unit:
            other_values = list(other._series.pint.to_base_units().values)
        if self.time_zone:
            tmp_self = copy.deepcopy(self)
            tmp_self.convert_time_zone('UTC')
            self_timestamps = tmp_self.timestamps
        if other.time_zone:
            tmp_other = copy.deepcopy(other)
            tmp_other.convert_time_zone('UTC')
            other_timestamps = tmp_other.timestamps

        return self_timestamps == other_timestamps and self_values == other_values

    def __getitem__(self, item) -> None:
        if isinstance(item, (slice, Iterable)):
            new_ts = copy.deepcopy(self)
            new_ts._series = new_ts._series[item]
            return new_ts
        else:
            if isinstance(item, dt.datetime) and item.tzinfo is None \
                    and self.time_zone is not None:
                item = self.time_zone.localize(item)
            return self._series[item]

    def __setitem__(self, key, value):
        raise NotImplementedError()

    # ---------------------------- calculations ------------------------------ #

    def _basic_calc(
        self,
        operation: BasicCalcOperationType,
        other: BasicCalcOperandType,
        *args, **kwargs
    ) -> TimestampSeries:
        if isinstance(other, TimestampSeries):
            return self._basic_calc_time_series(operation, other, **kwargs)
        elif isinstance(other, pd.Series):
            return self._basic_calc_pd_series(operation, other, **kwargs)
        elif isinstance(other, (collections.abc.Sequence, np.ndarray, PintArray)):
            return self._basic_calc_collection(operation, other)
        else:
            return self._basic_calc_scalar(operation, other)

    def _basic_calc_time_series(
        self,
        operation: BasicCalcOperationType,
        other: TimestampSeries,
        **kwargs
    ) -> TimestampSeries:
        tmp_series = copy.deepcopy(self)
        if self.freq != other.freq:
            raise ValueError("The time series have different frequencies")
        if not self.unit == other.unit:
            raise ValueError("The time series have different units")
        if not self._series.index.equals(other._series.index):
            warnings.warn("timestamps do not match, values are auto-filled",
                          category=TimestampMismatchWarning)
        tmp_series._series = getattr(tmp_series._series, operation)(
            other._series, **kwargs)
        tmp_series.convert_time_zone(self.time_zone)
        return tmp_series

    def _basic_calc_pd_series(
        self,
        operation: BasicCalcOperationType,
        other: pd.Series,
        **kwargs
    ) -> TimestampSeries:
        tmp_series = copy.deepcopy(self)
        if not isinstance(other.index, pd.DatetimeIndex):
            raise ValueError("The series has no proper DatetimeIndex")
        if not all(map(lambda x: isinstance(x, numbers.Number), other)):
            raise ValueError("sequence contains non-numeric values")
        if not self._series.index.equals(other.index):
            warnings.warn("timestamps do not match, values are auto-filled",
                          category=RuntimeWarning)
        tmp_series._series = getattr(tmp_series._series, operation)(other, **kwargs)
        # enforce resulting TimestampSeries' time zone to be equal to initial
        # TimestampSeries (self)
        tmp_series.convert_time_zone(self.time_zone)
        if isinstance(tmp_series._series.dtype, PintType):
            tmp_series._unit = tmp_series._series.pint.u
        return tmp_series

    def _basic_calc_collection(
        self,
        operation: BasicCalcOperationType,
        other: Union[collections.Sequence, np.ndarray, PintArray]
    ) -> TimestampSeries:
        tmp_series = copy.deepcopy(self)
        if len(other) != len(self):
            raise ValueError("sequence has different length")
        if not all(
            map(
                lambda x: isinstance(x, (numbers.Number, pint.Quantity)), other
            )
        ):
            raise ValueError("sequence contains non-numeric values")
        tmp_series._series = getattr(tmp_series._series, operation)(other)
        if isinstance(tmp_series._series.dtype, PintType):
            tmp_series._unit = tmp_series._series.pint.u
        return tmp_series

    def _basic_calc_scalar(
        self,
        operation: BasicCalcOperationType,
        other: Union[numbers.Number, pint.Quantity]
    ) -> TimestampSeries:
        tmp_series = copy.deepcopy(self)
        if not isinstance(other, (numbers.Number, pint.Quantity)):
            raise ValueError('value is not numeric')
        tmp_series._series = getattr(tmp_series._series, operation)(other)
        if isinstance(other, pint.Quantity):
            tmp_series._unit = tmp_series._series.pint.u
        return tmp_series

    # ---------------------------- validation ------------------------------- #

    def validate_all(self) -> None:
        self._validate()
        super()._validate_all()

    def _validate(self) -> None:
        index_is_datetime(self._series)
        index_is_sorted(self._series)
