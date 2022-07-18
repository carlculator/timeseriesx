import warnings

import pandas as pd

from timeseriesx.mixins import BaseMixin
from timeseriesx.validation.time_zone import (
    coerce_time_zone,
    infer_tz_from_series,
)


class TimeZoneWarning(RuntimeWarning):
    """
    warning about implicit time zone handling
    """
    pass


class TimeZoneMixin(BaseMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        time_zone = kwargs.get('time_zone', None)
        if time_zone == 'infer':
            time_zone = infer_tz_from_series(self._series)
        self._validate_time_zone(time_zone)

    @property
    def time_zone(self):
        return self._series.index.tz

    def convert_time_zone(self, tz):
        """
        convert time series index to another time zone, or make an time zone naive
        index time zone aware (or the other way round)

        :param str/datetime.tzinfo tz: tzinfo object or name of the new time zone
            or None
        :return: the series with converted index
        :rtype: BaseTimeSeries
        """
        tz = coerce_time_zone(tz)
        if self._series.index.tz is None or tz is None:
            # series.freq is dropped by tz_localize: see https://github.com/pandas-dev/pandas/issues/33677
            # therefore restore it manually
            freq = self._get_freq()
            self._series.index = self._series.index.tz_localize(tz)
            self._series.index = pd.DatetimeIndex(self._series.index, freq=freq)
        if tz is not None:
            self._series.index = self._series.index.tz_convert('UTC').tz_convert(tz)
        return self

    def _validate_time_zone(self, tz):
        inferred_tz = infer_tz_from_series(self._series)
        if inferred_tz != tz:
            warnings.warn('time zone and given timestamps do not conform, '
                          'converted timestamps to given time zone',
                          category=TimeZoneWarning)
        self.convert_time_zone(tz)

    def _get_freq(self):
        return getattr(self, 'freq', None)
