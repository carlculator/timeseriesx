import pandas as pd


def index_is_datetime(series, exception=True):
    """
    check if the index of a pandas.Series is a valid DatetimeIndex

    :param pandas.Series series: the series holding the index to check
    :param bool exception: if True, raise an exception in case of invalid DatetimeIndex
    :return: True if index is a valid DatetimeIndex, False otherwise.
    :rtype: bool
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        if exception:
            raise ValueError('series index is no proper DatetimeIndex')
        else:
            return False
    return True


def index_is_sorted(series, ascending=True, exception=True):
    """
    check if the (datetime-) index of a pandas.Series is sorted

    :param pandas.Series series: the series holding the index to check
    :param bool ascending: if true, check for ascending order, if false for
        descending order
    :param bool exception: if True, raise an exception in case of unsorted index
    :return: True if index is sorted, False otherwise.
    :rtype: bool
    """
    if not all(series.index.sort_values() == series.index):
        if exception:
            raise ValueError('time series index is not sorted (ascending)')
        else:
            return False
    return True
