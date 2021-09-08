import pandas as pd


def index_is_datetime(series, exception=True):
    """
    :param class:`pandas.Series` series:
    :param bool exception:
    :return:
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
    :param class:`pandas.Series` series:
    :param bool ascending:
    :param bool exception:
    :return:
    :rtype: bool
    """
    if not all(series.index.sort_values() == series.index):
        if exception:
            raise ValueError('time series index is not sorted (ascending)')
        else:
            return False
    return True
