import math
from datetime import tzinfo

import pandas as pd
import pytz


def tz_equal(tz1, tz2, trange=None):
    """
    Compares two time zones based on their utc offset on twelve timestamps within
    `trange` (if passed) or within the last 12 years covering every possible month.
    Therefore it is likely to cover all DST-effects. This approach does not guarantee
    equality, since it only tests the utc-offsets on a subset of 12 timestamps. It is an
    approximation, that should only fail in very(!) special cases.

    :param str/datetime.tzinfo tz1:
    :param str/datetime.tzinfo tz2:
    :param tuple trange:
    :return:
    :rtype: bool
    """
    if tz1 is None or tz2 is None:
        return tz1 == tz2

    if trange:
        start, end = trange[0], trange[1]
        delta = end - start
        if delta < pd.Timedelta(0):
            raise ValueError('start of trange has to be lower than end')
        periods = max(math.ceil(delta / pd.Timedelta(days=30)), 12)
        dates_1 = pd.date_range(start, end, periods=periods, tz=tz1)
        dates_2 = pd.date_range(start, end, periods=periods, tz=tz2)
    else:
        offset = - pd.DateOffset(years=1, months=1, days=1.0487)
        dates_1 = pd.date_range(pd.Timestamp.now(), periods=12, freq=offset, tz=tz1)
        dates_2 = pd.date_range(pd.Timestamp.now(), periods=12, freq=offset, tz=tz2)
    return all(
        map(
            lambda t: t[0].tz.utcoffset(t[0]) == t[1].tz.utcoffset(t[1]),
            zip(dates_1, dates_2),
        )
    )


def coerce_time_zone(tz):
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
    elif isinstance(tz, tzinfo):
        return tz
    else:
        raise ValueError(f'{tz} is not a valid timezone')


def infer_tz_from_timestamp(timestamp):
    """
    infer the the time zone from a timestamp object

    :param pandas.Timestamp/datetime.datetime timestamp: the target timestamp
    :return: the inferred time zone
    :rtype: datetime.tzinfo
    """
    return timestamp.tzinfo


def infer_tz_from_series(series):
    """
    infer the the time zone from a pandas series with `DatetimeIndex`

    :param pandas.Series series: the target series
    :return: the inferred time zone
    :rtype: datetime.tzinfo
    """
    return getattr(series.index, 'tzinfo', None)
