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

    def fill_gaps(self, start, end, value=np.NaN):
        """
        fill all gaps between `start` and `end` in a series with a frequency with a
        constant value

        :param datetime.datetime start: the start timestamps of the period that will be
            investigated (included)
        :param datetime.datetime end: the end timestamps of the period that will be
            investigated (included)
        :param float/int/np.float value: the constant fill value
        :return: return the series with filled gaps
        :rtype BaseTimeSeries
        """
        if not self._freq:
            raise ValueError('cannot determine gaps when freq is not set')
        expected_index = pd.date_range(
            start, end, freq=self._freq, tz=self._get_time_zone())
        self._series = self._series.reindex(
            self._series.index.join(expected_index, how='outer'))
        self._series.loc[self._series.isnull()] = value
        return self

    def get_gaps(self, start, end):
        """
        get all timestamps between `start` and `end` from a series with a frequency,
        where the value is missing or NaN

        :param datetime.datetime start: the start timestamps of the period that will be
            investigated (included)
        :param datetime.datetime end: the end timestamps of the period that will be
            investigated (included)
        :return: list of timestamps
        :rtype list of datetime.datetime
        """
        if not self._freq:
            raise ValueError('cannot determine gaps when freq is not set')
        tmp_series = copy.deepcopy(self)
        tmp_series.fill_gaps(start, end)
        gap_series = tmp_series._series[tmp_series._series.isnull()]
        return gap_series.index.to_pydatetime().tolist()

    def resample(self, freq, method):
        """
        resample the series to a smaller frequency, aggregate the values

        :param freq: the new frequency, has to be smaller than the current
            frequency (greater offset)
        :param str method: aggregation method, e.g. 'mean', 'sum', 'min', 'max'
        :return: the resamples time series
        :rtype BaseTimeSeries
        """
        if not self._freq:
            raise ValueError('cannot resample when freq is not set')
        freq = coerce_freq(freq)
        if self._freq >= freq:
            raise ValueError(
                'can only resample to smaller frequencies (larger offsets)'
            )
        freq = coerce_freq(freq)
        self._series = getattr(self._series.resample(freq), method)()
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
