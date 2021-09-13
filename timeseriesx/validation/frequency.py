import pandas as pd


def infer_freq(series):
    """
    infer the frequency from a pandas Series with `DatetimeIndex`

    :param pandas.Series series: a pandas Series with `DatetimeIndex`
    :return: the inferred frequency object
    :rtype: pandas.DateOffset
    """
    if series.index.freq is not None and hasattr(series.index.freq, "delta"):
        delta = series.index.freq.delta
    else:
        if len(series) <= 1:
            raise ValueError("cannot infer freq from series with one or less elements")
        else:
            delta = series.index[1] - series.index[0]
            diffs = series.index[1:] - series.index[:-1]
            if any([delta != f for f in diffs]):
                raise ValueError("could not infer freq from series")
    return coerce_freq(delta)


def coerce_freq(freq):
    """
    return a convenient representation of a frequency as a pandas.DateOffset object

    :param str/datetime.timedelta/pandas.Timedelta/pandas.DateOffset freq:
        a frequency string or frequency object to coerce
    :return: coerced frequency object
    :rtype: pandas.DateOffset
    """
    if freq is None:
        return freq
    else:
        try:
            return pd.tseries.frequencies.to_offset(freq)
        except ValueError:
            raise ValueError('invalid frequency')
