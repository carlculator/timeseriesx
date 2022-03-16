import copy

import numpy as np
import pandas as pd

from timeseriesx.mixins import BaseMixin
from timeseriesx.validation.frequency import (
    coerce_freq,
    infer_freq,
)


class FrequencyMixin(BaseMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._freq = kwargs.get('freq', None)
        if self._freq == 'infer':
            self._freq = infer_freq(self._series)
        self._validate_freq()

    @property
    def freq(self):
        return self._freq

    def fill_gaps(self, start=None, end=None, value=np.NaN):
        """
        fill all gaps between `start` and `end` in a series with a frequency with a
        constant value

        :param datetime.datetime start: the start timestamps of the period that will be
            investigated (included). If None, then the first timestamp in the
            time series is considered as start. Defaults to None
        :param datetime.datetime end: the end timestamps of the period that will be
            investigated (included). If None, then the last timestamp in the
            time series is considered as end. Defaults to None
        :param float/int/np.float value: the constant fill value
        :return: return the series with filled gaps
        :rtype: BaseTimeSeries
        """
        if not self._freq:
            raise ValueError('cannot determine gaps when freq is not set')
        if (not start or not end) and self._series.empty:
            raise ValueError('cannot fill the gaps for empty series '
                             'without parameters providing start and end.')

        start = start or self._series.index[0].to_pydatetime()
        end = end or self._series.index[-1].to_pydatetime()

        try:
            expected_index = pd.date_range(
                start, end, freq=self._freq, tz=self._get_time_zone())
        except (AssertionError, TypeError):
            raise ValueError('time zone of parameter start or end does not match '
                             'the time zone of the series')
        self._series = self._series.reindex(
            self._series.index.join(expected_index, how='right'))
        self._series.loc[self._series.isnull()] = value
        return self

    def get_gaps(self, start=None, end=None):
        """
        get all timestamps between `start` and `end` from a series with a frequency,
        where the value is missing or NaN

        :param datetime.datetime start: the start timestamps of the period that will be
            investigated (included). If None, then the first timestamp in the
            time series is considered as start. Defaults to None
        :param datetime.datetime end: the end timestamps of the period that will be
            investigated (included). If None, then the last timestamp in the
            time series is considered as end. Defaults to None
        :return: list of timestamps
        :rtype: list of datetime.datetime
        """
        if not self._freq:
            raise ValueError('cannot determine gaps when freq is not set')
        if (not start or not end) and self._series.empty:
            raise ValueError('cannot determine the gaps from empty series '
                             'without parameters providing start and end.')

        tmp_series = copy.deepcopy(self)
        start = start or self._series.index[0].to_pydatetime()
        end = end or self._series.index[-1].to_pydatetime()

        tmp_series.fill_gaps(start, end)
        gap_series = tmp_series._series[tmp_series._series.isnull()]
        return gap_series.index.to_pydatetime().tolist()

    def resample(self, freq, method):
        """
        resample the series to a smaller frequency, aggregate the values

        :param str/datetime.timedelta/pandas.Offset/pandas.Timedelta freq:
            the new frequency, has to be smaller than the current frequency
            (greater offset)
        :param str/Callable method: aggregation method, currently supported
            are "all", "any", "min", "max", "sum", "mean", "median", or function
            that a collection (e.g. pandas.Series or list) of numeric values as
            its argument and returns a scalar
        :return: the resamples time series
        :rtype: BaseTimeSeries
        """
        if not self._freq:
            raise ValueError('cannot resample when freq is not set')
        freq = coerce_freq(freq)
        if self._freq >= freq:
            raise ValueError(
                'can only resample to smaller frequencies (larger offsets)'
            )
        freq = coerce_freq(freq)
        # perform the aggregation on a tmp_series to avoid
        # potential problems with units. Issue with unit aggregations has been
        # reported to pint-pandas https://github.com/hgrecco/pint-pandas/issues/117
        tmp_series = self.as_pd_series()
        tmp_series = getattr(tmp_series.resample(freq), 'aggregate')(method)
        self._series = tmp_series.astype(self._series.dtype)
        self._freq = freq
        return self

    def _validate_freq(self):
        self._freq = coerce_freq(self._freq)
        try:
            self._series.index.freq = self._freq
        except ValueError:
            raise ValueError('frequency does not conform to timestamps')

    def _validate_all(self):
        super()._validate_all()
        self._validate_freq()

    def _get_time_zone(self):
        if hasattr(self, 'time_zone'):
            return self.time_zone
        else:
            return None
