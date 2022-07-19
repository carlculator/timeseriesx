import datetime as dt
from typing import Optional, Union
import pandas as pd
import pytz

from timeseriesx.base.types import TimeZoneType


def coerce_time_zone(tz: TimeZoneType) -> Optional[dt.tzinfo]:
    """
    returns the convenient representation of a time zone as a pytz-based `tzinfo`-object

    :param str/datetime.tzinfo tz:
    :return: the coerced time zone object
    :rtype: datetime.tzinfo
    """
    if tz is None:
        return tz
    if isinstance(tz, str):
        return pytz.timezone(tz)
    elif isinstance(tz, dt.tzinfo):
        return tz
    else:
        raise ValueError(f'{tz} is not a valid timezone')


def infer_tz_from_timestamp(
    timestamp: Union[pd.Timestamp, dt.datetime],
) -> Optional[dt.tzinfo]:
    """
    infer the the time zone from a timestamp object

    :param pandas.Timestamp/datetime.datetime timestamp: the target timestamp
    :return: the inferred time zone
    :rtype: datetime.tzinfo
    """
    return timestamp.tzinfo


def infer_tz_from_series(series: pd.Series) -> Optional[dt.tzinfo]:
    """
    infer the the time zone from a pandas series with `DatetimeIndex`

    :param pandas.Series series: the target series
    :return: the inferred time zone
    :rtype: datetime.tzinfo
    """
    return getattr(series.index, 'tzinfo', None)
