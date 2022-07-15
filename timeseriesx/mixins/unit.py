import warnings

import numpy as np
import pandas as pd
from pint import DimensionalityError
from pint_pandas import (
    PintArray,
    PintType,
)
from timeseriesx.mixins import BaseMixin
from timeseriesx.validation.unit import coerce_unit


class UnitWarning(Warning):
    pass


class UnitMixin(BaseMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        unit = kwargs.get("unit", None)
        self._validate_unit(unit)

    @property
    def unit(self):
        if isinstance(self._series.dtype, PintType):
            return self._series.pint.u
        else:
            return None

    def _get_magnitude_series(self):
        if self.unit is None:
            return self._series
        else:
            return self._series.pint.magnitude

    def aggregate(self, func, with_unit=False):
        """
        aggregate all values of the series with a custom aggregation function

        :param function func: a function mapping a numeric list/array/vector to a scalar
        :param boolean with_unit: flag whether to return the result as a pint
            object, defaults to False
        :return: the aggregated value
        :rtype: numpy.float/numpy.int/pint.Quantity
        """
        if self.unit is None or with_unit:
            return self._series.agg(func)
        else:
            return self._get_magnitude_series().agg(func)

    def sum(self, with_unit=False):
        """
        calculate the sum of all values of the series

        :param boolean with_unit: flag whether to return the result as a pint
            object, defaults to False
        :return: the sum of the values
        :rtype: numpy.float/numpy.int/pint.Quantity
        """
        return self.aggregate(np.sum, with_unit)

    def mean(self, with_unit=False):
        """
        calculate the mean of all values of the series

        :param boolean with_unit: flag whether to return the result as a pint
            object, defaults to False
        :return: the mean of the values
        :rtype: numpy.float/numpy.int/pint.Quantity
        """
        return self.aggregate(np.mean, with_unit)

    def as_pd_series(self, include_nan=True):
        tmp_series = self._get_magnitude_series()
        if include_nan:
            return tmp_series
        else:
            return tmp_series[tmp_series.notnull()]

    def convert_unit(self, unit):
        """
        convert the unit of the series

        :param str/pint.Unit unit:
        :return: the time series with converted units
        :rtype: BaseTimeSeries
        """
        if unit is None:
            if isinstance(self._series.dtype, PintType):
                self._series = pd.Series(
                    self._series.pint.magnitude,
                    index=self._series.index
                )
        else:
            unit = coerce_unit(unit)
            if not isinstance(self._series.dtype, PintType):
                self._series = pd.Series(
                    PintArray(self._series.values, dtype=unit),
                    index=self._series.index,
                )
            else:
                try:
                    self._series = self._series.pint.to(unit)
                except DimensionalityError:
                    raise ValueError(f'{unit} unit is not compatible with {self.unit}')
        return self

    def _validate_unit(self, unit):
        coerce_unit(unit)
        if isinstance(self._series.dtype, PintType):
            if self.unit != unit:
                warnings.warn('passed unit and unit of series do not conform, '
                              'converted unit to the given unit',
                              category=UnitWarning)
        self.convert_unit(unit)
