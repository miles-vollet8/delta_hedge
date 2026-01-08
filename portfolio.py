from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import timedelta

from contracts import OptionContract


TRADING_DAYS = 365.0
from pricing import iv_solve, compute_greeks
from config import PnLModelConfig
from hedging import compute_hedges
from single_contract import SingleContractPnL
import math

class PortfolioPnL():
    """
    Aggregates PnL across multiple option contracts.
    
    Each contract is processed independently using SingleContractPnL,
    then results are summed by date to produce portfolio-level PnL.
    """

    def __init__(
        self,
        stock_series:pd.Series,
        contracts: List[OptionContract],
        config: PnLModelConfig
    ):
        self.stock_series = stock_series
        self.contracts = contracts
        self.config = config or PnLModelConfig()

    def run(self, hedge_config=None):
        if hedge_config is None:
            hedge_config = self.config
        paths = self.build_paths()
        port_path = self._aggregate_paths(paths)
        hedged_path = compute_hedges(port_path, hedge_config)
        port_pnl = self._compute_port_pnl(hedged_path)
        return hedged_path, port_pnl
            


    def build_paths(self)->dict[str, pd.DataFrame]: #builds paths for each contract and returns a dict with each id and path_df

        paths: dict[str, pd.DataFrame] = {}
        for contract in self.contracts:
            if contract.id is None:
                raise ValueError("Portfolio level paths require a unique id for each option contract")
            if contract.id in paths:
                raise ValueError(f'Contract id {contract.id} is already in the dict')
            current = SingleContractPnL(self.stock_series, contract, self.config)
            path = current.build_path()
            paths[contract.id] = path
        self.paths = paths
        return paths


    def _aggregate_paths(self, paths: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        combines all paths on each respective timestamp and shows how many contracts are active on each interval
        """
        aggregate_df = next(iter(paths.values()))
        idx = aggregate_df.index

        port = pd.DataFrame(index = idx)
        port['stock_price'] = aggregate_df['stock_price']

        port['position_delta'] = 0.0
        port['position_gamma'] = 0.0
        port['position_theta'] = 0.0
        port['position_vega'] = 0.0
        port['position_rho'] = 0.0
        port['vega_pnl'] = 0.0
        port['option_pnl'] = 0.0
        port['option_value'] = 0.0
        port['active_positions'] = 0

        for id, df in paths.items():
            temp = df.reindex(idx)
            dIV = temp["implied_vol"].diff().fillna(0.0)*100
            temp['vega_pnl'] = (
                0.5*(temp["position_vega"] + temp['position_vega'].shift(1)) * dIV
            )
            active = temp['position_delta'].notna()

            port.loc[active, 'position_delta'] += temp.loc[active, 'position_delta'] #will also need to aggregate other greeks
            port.loc[active, 'position_gamma'] += temp.loc[active, 'position_gamma']
            port.loc[active, 'position_theta'] += temp.loc[active, 'position_theta']
            port.loc[active, 'position_rho'] += temp.loc[active, 'position_rho']
            port.loc[active, 'vega_pnl'] += temp.loc[active, 'vega_pnl'].fillna(0.0)
            port.loc[active, 'option_pnl'] += temp.loc[active, 'option_pnl'].fillna(0.0)
            port.loc[active, 'option_value'] += (temp.loc[active, 'option_value']) # need to make sure this accounts for long/short
            port.loc[active, 'active_positions'] += 1

        for col in ["log_return", "realized_vol", "total_return", 'rolling_rv']:
            if col in aggregate_df.columns:
                port[col] = aggregate_df[col]
        self.port_path = port
        return port

    def _compute_port_pnl(self, hedged_path:pd.DataFrame) -> pd.DataFrame:
        """
        attributes P&L on the portfolio level, important to remember vega is NOT additive across the portfolio
        due to its relation to time and the specific iv of a given contract
        """

        df = hedged_path.copy()
       
        dS = df["stock_price"].diff().fillna(0.0)
        df['stock_change'] = dS

        df['dt_seconds'] = df.index.to_series().diff().dt.total_seconds()
        df['dt_seconds'] = df['dt_seconds'].fillna(0.0)
        dtime = (df['dt_seconds']) / (86400*TRADING_DAYS)

        

        gamma_pnl = 0.5 * df["position_gamma"] * (dS ** 2)
        theta_pnl = (
            df["position_theta"] * (dtime) * TRADING_DAYS
        )
        vega_pnl = df['vega_pnl']

        df["gamma_pnl"] = gamma_pnl
        df["theta_pnl"] = theta_pnl

        df["realized_vol_pnl"] = gamma_pnl + theta_pnl
        df['implied_vol_pnl'] = vega_pnl
        df["theo_pnl"] = gamma_pnl + vega_pnl + theta_pnl

        df['stock_return'] = df['log_return']*100
        df['total_stock_return'] = df['total_return']*100

        df['stock_pnl'] = df['hedge_shares'].shift(1)*df['stock_change']

        df['actual_pnl'] = df['stock_pnl'] + df['option_pnl']
        df['position_pnl'] = df['actual_pnl'].cumsum()
        df['abs_actual_pnl'] = df['actual_pnl'].abs()
        df['residual_pnl'] = df['actual_pnl'] - df['theo_pnl']

        df['interest_rate'] = self.config.risk_free_rate
        df['interest_pnl'] = (df['interest_rate']*dtime) * -df['hedge_shares'] * df['stock_price']
        
        df["realized_vol"] = df["log_return"].expanding(min_periods=2).std() * np.sqrt(TRADING_DAYS)
        #df['abs_move_zscore'] = df['log_return']/(df['implied_vol']/math.sqrt(TRADING_DAYS)) # need to add a way to choose which contract implied vol to use for zscore
        columns_to_return = [
    # 1) Price Data
        'dte',
        'stock_price',
        'stock_change',
        'log_return',
        'total_return',
        'option_price',


    # 2) Volatility
        'rolling_rv',
        'realized_vol',
    #    'abs_move_zscore', # need to add a way for them to choose which contract iv to use for zscore and calc based on that one

    # 3) Position Weighted Greeks
    'position_delta',
    'position_gamma',
    'position_theta',
    #'position_vega',
    'position_rho',
    
    # 4) Actual Pnl
    'active_positions',
    "hedge_shares",
    "stock_pnl",
    "option_pnl",
    "interest_pnl",
    "actual_pnl",
    "abs_actual_pnl",
    'position_pnl',

    # 5) Theoretical Pnl
    'gamma_pnl',
    'realized_vol_pnl',
    'vega_pnl',
    'theo_pnl',
    'residual_pnl' # unexplained pnl
    
]
        self.portfolio_pnl = df
        existing_columns = [col for col in columns_to_return if col in df.columns]
        return df[existing_columns]
        
    
    