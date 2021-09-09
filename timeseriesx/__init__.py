"""Top-level package for TimeSeriesX."""

__author__ = """Alexander Schulz"""
__email__ = 'info@alexander-schulz.eu'
__version__ = '0.1.4'

import pint
import pint_pandas

ureg = pint.UnitRegistry()
pint_pandas.PintType.ureg = ureg

from timeseriesx.base.timestamp_series import TimestampSeries
