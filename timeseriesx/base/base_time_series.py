import copy
from abc import ABCMeta, abstractmethod

import numpy as np


class BaseTimeSeries(metaclass=ABCMeta):

    # ------------------------------ properties ------------------------------ #

    @property
    def empty(self):
        return self._series.empty

    @property
    def values(self):
        raise NotImplementedError()

    @property
    def first(self):
        raise NotImplementedError()

    @property
    def last(self):
        raise NotImplementedError()

    # ---------------------------- functionality ----------------------------- #

    def map(self, func, **kwargs):
        """
        apply a custom function to each value of the series

        :param function func: a function mapping a scalar to another scalar
        :return: self
        :rtype: BaseTimeSeries
        """
        self._series = self._series.apply(func)
        return self

    def aggregate(self, func):
        """
        aggregate all values of the series with a custom aggregation function

        :param function func: a function mapping a numeric list/array/vector to a scalar
        :return: the aggregated value
        :rtype: numpy.float/numpy.int
        """
        return self._series.agg(func)

    def sum(self):
        return self.aggregate(np.sum)

    def mean(self):
        return self.aggregate(np.mean)

    def round(self, decimals):
        raise NotImplementedError()

    def append(self, value):
        raise NotImplementedError()

    def prepend(self, value):
        raise NotImplementedError()

    def join(self, other_ts, fit=True):
        raise NotImplementedError()

    # --------------------------------- cast --------------------------------- #

    @abstractmethod
    def as_tuples(self):
        raise NotImplementedError()

    @abstractmethod
    def as_dict(self, ordered=False):
        raise NotImplementedError()

    def as_pd_series(self, include_nan=True):
        if include_nan:
            return self._series
        else:
            return self._series[self._series.notnull()]

    # ---------------------------- magic methods ----------------------------- #

    def __getitem__(self, item):
        if isinstance(item, slice):
            new_ts = copy.deepcopy(self)
            new_ts._series = new_ts._series[item]
            return new_ts
        else:
            return self._series[item]

    def __setitem__(self, key, value):
        return NotImplemented()

    def __len__(self):
        return self._series.size

    def __eq__(self, other):
        return NotImplemented()

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if self._i < len(self):
            result = self[self._i]
            self._i += 1
            return result
        else:
            raise StopIteration

    def __add__(self, other):
        return self._basic_calc('add', other, fill_value=0)

    def __sub__(self, other):
        return self._basic_calc('subtract', other, fill_value=0)

    def __mul__(self, other):
        return self._basic_calc('__mul__', other)

    def __truediv__(self, other):
        return self._basic_calc('__truediv__', other)

    def __floordiv__(self, other):
        return self._basic_calc('__floordiv__', other)

    def __str__(self):
        raise NotImplementedError()

    def __repr__(self):
        raise NotImplementedError()

    # ---------------------------- calculations ------------------------------ #

    @abstractmethod
    def _basic_calc(self, operation, other, **kwargs):
        """
        :param str operation:
        :param class:`BaseTimeSeries`/class:`collections.Sequence`/
               class:`numpy.ndarray`/class:`numbers.Number` other:
        :return: a new time series copy holding the results of the calculation
        :rtype: BaseTimeSeries
        """
        raise NotImplementedError()
