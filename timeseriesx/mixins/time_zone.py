import warnings

from timeseriesx.mixins import BaseMixin
from timeseriesx.validation.time_zone import (
    coerce_time_zone,
    infer_tz_from_series,
)


class TimeZoneMixin(BaseMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._time_zone = kwargs.get('time_zone', None)
        if self._time_zone == 'infer':
            self._time_zone = infer_tz_from_series(self._series)
        self._validate_time_zone()

    @property
    def time_zone(self):
        return self._time_zone

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
            self._series.index = self._series.index.tz_localize(tz)
        if tz is not None:
            self._series.index = self._series.index.tz_convert('UTC').tz_convert(tz)
        self._time_zone = tz
        return self

    def _validate_time_zone(self):
        inferred_tz = infer_tz_from_series(self._series)
        self.convert_time_zone(self._time_zone)
        if inferred_tz != self._time_zone:
            warnings.warn('time zone and given timestamps do not conform, '
                          'converted timestamps to given time zone')

    def _validate_all(self):
        super()._validate_all()
        self._validate_time_zone()
