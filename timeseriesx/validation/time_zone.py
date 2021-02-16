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


def tzname_is_valid(tz, exception=False):
    """
    validate if a string represents a time zone

    :param str tz: the name of the timezone to be validated
    :param exception exception: if True, raise a ValueError when validation fails
    :return: True, if timezone is valid, else False
    :rtype: bool
    """
    if not isinstance(tz, str):
        raise ValueError(f'{tz} is not a valid time zone name')
    if not str.lower(tz) in map(str.lower, pytz.all_timezones):
        if exception:
            raise ValueError("invalid time zone")
        else:
            return False
    return True


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
        tzname_is_valid(tz, exception=True)
        return pd.Timestamp.now(tz=tz).tz
    elif isinstance(tz, tzinfo):
        tz_name = tz.tzname(pd.Timestamp.now())
        tzname_is_valid(tz_name, exception=True)
        return pd.Timestamp.now(tz=tz).tz_convert(tz_name).tz  # coerce pytz
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
