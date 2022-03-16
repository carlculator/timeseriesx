"""Top-level package for TimeSeriesX."""

__author__ = """Alexander Schulz"""
__email__ = 'info@alexander-schulz.eu'
__version__ = '0.1.12'

import pint

ureg = pint.get_application_registry()

from timeseriesx.base.timestamp_series import TimestampSeries
