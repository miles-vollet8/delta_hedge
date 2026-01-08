from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd

ContractSide = Literal["long", "short"]
OptionType = Literal["call", "put"]


###########################
# Option contract dataclass 
# has long/short call/put support
###########################
@dataclass()
class OptionContract:
    id: str
    strike: float
    expiration: pd.Timestamp
    entry_timestamp: pd.Timestamp
    option_price: pd.Series # series of observed option prices, indexed by timestamp
    entry_price: float
    stock_price_at_entry: float

    exit_timestamp: Optional[pd.Timestamp] = None  # If None, defaults to expiration
    option_type: OptionType = "call"
    contracts: int = 1
    side: ContractSide = "long"
    contract_multiplier: float = 100.0
    option_mode: str = 'option_price' # can also be iv if the terms are iv values

    @property
    def effective_exit_date(self) -> pd.Timestamp:
        """Returns exit_date if set, otherwise expiration."""
        return self.exit_timestamp if self.exit_timestamp is not None else self.expiration

    @property
    def signed_contracts(self) -> float:
        multiplier = 1.0 if self.side == "long" else -1.0
        return multiplier * self.contracts

    def _option_price_series(self, stock_index: pd.Index) -> pd.Series:
        if self.option_price is None or len(self.option_price) == 0:
            raise ValueError("OptionContract.option_series is empty / None")

        s = self.option_price.copy()
        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index)
        s = s.sort_index()

        if stock_index is not None:
            stock_index = pd.DatetimeIndex(stock_index)
            if s.index.min() < stock_index.min() or s.index.max() > stock_index.max():
                raise ValueError(
                    "Option series index is not fully contained within stock series index"
                )
        return s
    