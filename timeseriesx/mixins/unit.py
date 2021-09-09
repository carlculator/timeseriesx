import warnings

import pandas as pd
from pint_pandas import (
    PintArray,
    PintType,
)

from timeseriesx import ureg
from timeseriesx.mixins import BaseMixin
from timeseriesx.validation.unit import coerce_unit


class UnitMixin(BaseMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._unit = kwargs.get("unit", None)
        if self._unit:
            self._unit = coerce_unit(self._unit)
        self._validate_unit()

    @property
    def unit(self):
        return self._unit

    def _get_magnitude_series(self):
        if self._unit is None:
            return self._series
        else:
            return self._series.pint.magnitude

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
        :return:
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
                if ureg.is_compatible_with(self._unit, unit):
                    self._series = self._series.pint.to(unit)
                else:
                    raise ValueError(f'{unit} unit is not compatible with {self._unit}')
        self._unit = unit
        return self

    def _validate_unit(self):
        coerce_unit(self._unit)
        if isinstance(self._series.dtype, PintType):
            if self._series.pint.u != self._unit:
                try:
                    self.convert_unit(self._unit)
                except ValueError:
                    raise ValueError()
                else:
                    warnings.warn('passed unit and unit of series do not conform')
        else:
            self.convert_unit(self._unit)

    def _validate_all(self):
        super()._validate_all()
        self._validate_unit()
