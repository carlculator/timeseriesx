from __future__ import annotations

import collections
import datetime as dt
import numbers
from typing import Union, TypeVar, Optional

import numpy as np
import pandas as pd
import pint
import pint_pandas
from typing_extensions import TypeAlias, Literal

TimestampType: TypeAlias = Union[dt.datetime, pd.Timestamp]
TimestampSeriesType = TypeVar("TimestampSeriesType", bound="TimestampSeries")
FreqType: TypeAlias = Optional[Union[str, dt.timedelta, pd.offsets.BaseOffset,
                                     pd.Timedelta]]
UnitType: TypeAlias = Optional[Union[str, pint.Unit]]
TimeZoneType: TypeAlias = Optional[Union[str, dt.tzinfo]]
InferableTimeZoneType: TypeAlias = Union[Literal["infer"], TimeZoneType]
InferableFreqType: TypeAlias = Union[Literal["infer"], FreqType]
BasicCalcOperationType: TypeAlias = Literal[
    "add", "subtract", "__mul__", "__div__", "__floordiv__"
]
BasicCalcOperandType: TypeAlias = Union[
    TimestampSeriesType, pd.Series, collections.Sequence, np.ndarray,
    pint_pandas.PintArray, numbers.Number
]
