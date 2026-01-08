from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from datetime import timedelta

from contracts import OptionContract


TRADING_DAYS = 365.0
from pricing import iv_solve, compute_greeks
from config import PnLModelConfig
from hedging import compute_hedges, parse_hedge_freq
import math

class SingleContractPnL():
    """
    Framework for computing theoretical and realized PnL
    for one listed option contract on daily close data.

    The model is intentionally modular so it can be extended
    to a multi-contract portfolio without rewriting the core logic.
    """
    def __init__(self, stock_series: pd.Series, contract: OptionContract, 
                 config: PnLModelConfig = None, solver = None):
        self.contract = contract
        self.config = config or PnLModelConfig()
        self.solver = solver if solver is not None else iv_solve
        self.stock_df = self._normalize_stock_series(stock_series)

    def run(self, config=None) -> tuple[pd.DataFrame, pd.DataFrame]: #returns the path and pnl of a single contract
        if config is None:
            config = self.config
        path_df = self.build_path()
        hedged_df = compute_hedges(path_df, config)
        pnl_df = self.build_pnl(hedged_df, config)
        return path_df, pnl_df

    def build_path(self)->pd.DataFrame:
        side_sign = 1 if self.contract.side == 'long' else -1
        notional = self.contract.contracts * self.contract.contract_multiplier
        df = self.stock_df.copy()                     # stock_price, log_return, total_return

        # 1) attach sparse option series (option_price or iv)
        df = self._attach_option_price(df)       # LEFT join, no dropping stock rows

        # 2) time to expiry
        df["time_to_expiry"] = self._time_to_expiry(df.index)
        df['dte'] = self._days_to_expiry(df.index)

        # 3) resolve IV only where we have info (iv or option_price)
        iv = self._resolve_implied_vol(df)       # NaN where no quote
        iv = iv.ffill()
        iv = iv.dropna()
        df = df.loc[iv.index]
        df["implied_vol"] = iv

        # 4) synthetic option prices from S, T, IV (BS)
        df["option_price"] = self._solve_price_from_iv(df)

        # 5) greeks on full path
        greek_df = self._compute_greeks(df)
        self.greek_df = greek_df
        df = pd.concat([df, greek_df], axis=1)

        df['delta_signed'] = side_sign * df['delta']
        df['gamma_signed'] = side_sign * df['gamma']
        df['vega_signed'] = side_sign * df['vega']
        df['theta_signed'] = side_sign * df['theta']
        df['rho_signed'] = side_sign * df['rho']

        # 6) realized vol, entry/exit clipping, etc.
        df["rolling_rv"] = self._rolling_realized_vol(df["log_return"])
        df['option_value'] = df['option_price'] * notional*side_sign
        option_pnl = df["option_value"].diff().fillna(0.0)
        option_pnl.iloc[0] = 0.0
        df['option_pnl'] = option_pnl
        df['position_delta'] = notional * df['delta_signed']
        df['position_gamma'] = notional * df['gamma_signed']
        df['position_vega'] = notional * df['vega_signed']
        df['position_theta'] = notional * df['theta_signed']
        df['position_rho'] = notional * df['rho_signed']
        df = self._clip_to_entry_exit(df)        
        self.path_df = df
        return df

    def _clip_to_entry_exit(self, path_df:pd.DataFrame) -> pd.DataFrame:
        start = self.contract.entry_timestamp
        end = self.contract.exit_timestamp
        clipped = path_df.loc[start:end].copy()

        entry = self._compute_entry_state()
        first_idx = clipped.index[0]

        key_map = {
            "stock_price": "stock_price",
            "implied_vol": "implied_vol",
            "delta": "delta",
            "gamma": "gamma",
            "theta": "theta",
            "vega": "vega",
            "rho": "rho",
            "starting_dte": "dte",
            "starting_time_to_expiry": "time_to_expiry",
        }
        
        for key, col in key_map.items():
            if key in entry and col in clipped.columns:
                clipped.loc[first_idx, col] = entry[key]
        notional = self.contract.contracts * self.contract.contract_multiplier
        side_sign = 1 if self.contract.side =='long' else -1
        totals = {
            "delta": "position_delta",
            "gamma": "position_gamma",
            "theta": "position_theta",
            "vega": "position_vega",
            "rho": "position_rho",
        }
        
        for greek, col in totals.items():
            if greek in entry and col in clipped.columns:
                clipped.loc[first_idx, col] = entry[greek]*notional*side_sign
        
        if "option_price" in clipped.columns:
        # mark at the states you're storing (here: entry state on first row)
            clipped["option_value"] = clipped["option_price"] * notional * side_sign

            option_pnl = clipped["option_value"].diff().fillna(0.0)
            option_pnl.iloc[0] = 0.0    # no MTM between "prior" and entry, by construction
            clipped["option_pnl"] = option_pnl    
        #clipped.loc[first_idx, 'option_pnl'] =  clipped.loc[first_idx, 'option_value'] - entry['option_value']
        return clipped
    
    def build_pnl(self, hedge_df=None, config = None) -> pd.DataFrame: #takes the path dataframe and runs hedging sim from entry to exit
        if config is None:
            config = self.config
        if hedge_df is None:
            path_df = self.build_path()
            hedge_df = compute_hedges(path_df, config)
        entry_state = self._compute_entry_state()
        
        pnl_df = self._compute_pnl(hedge_df, entry_state)
        self.pnl_df = pnl_df
        return pnl_df

    def _resolve_delta_bands(self, hedge_df) ->pd.DataFrame:
        # Add hedge flags and new hedges when total_delta band is breached
        return

    def _normalize_stock_series(self, stock_series:pd.Series) ->pd.DataFrame: 
        # confirms stock prices are in order and include the entry and exit of the option

        contract = self.contract

        if contract.entry_timestamp is None:
            raise ValueError("OptionContract.entry_timestamp must be set.")
        effective_exit = contract.exit_timestamp or contract.expiration
        if effective_exit is None:
            raise ValueError("OptionContract.expiration or exit_timestamp must be set.")

        entry_ts = pd.Timestamp(contract.entry_timestamp)
        exit_ts = pd.Timestamp(effective_exit)
        if exit_ts < entry_ts:
            raise ValueError(
                f"exit_timestamp ({exit_ts}) is before entry_timestamp ({entry_ts})."
            )

        # --- basic Series checks / normalization ---
        if not isinstance(stock_series, pd.Series):
            raise TypeError(f"stock_series must be a pandas Series, got {type(stock_series)}")

        idx = stock_series.index
        if not isinstance(idx, pd.DatetimeIndex):
            try:
                idx = pd.to_datetime(idx)
            except Exception as e:
                raise ValueError("Index could not be converted to DatetimeIndex") from e
        if idx.isna().any():
            raise ValueError("DatetimeIndex contains NaT values after conversion")

        stock_series = stock_series.copy()
        stock_series.index = idx

        stock_series = stock_series.sort_index()
        if stock_series.index.has_duplicates:
            dupes = stock_series.index[stock_series.index.duplicated()].unique()
            raise ValueError(f"Duplicate timestamps in stock_series: {dupes}")

        stock_series = pd.to_numeric(stock_series, errors="coerce")
        if stock_series.isna().any():
            raise ValueError("stock_series contains non-numeric values (became NaN)")
        if (stock_series <= 0).any():
            bad_ts = stock_series[stock_series <= 0].index
            raise ValueError(f"stock_series contains non-positive prices at {bad_ts}")

        idx = stock_series.index
        idx_min = idx[0]
        idx_max = idx[-1]

        if idx_min > entry_ts or idx_max < exit_ts:
            raise ValueError(
                "stock_series does not cover the full contract window: "
                f"index range [{idx_min}, {idx_max}], "
                f"required at least [{entry_ts}, {exit_ts}]."
            )
        df = pd.DataFrame(index=stock_series.index)
        df["stock_price"] = stock_series

        df["log_return"] = np.log(df["stock_price"]).diff().fillna(0.0)
        base_price = df["stock_price"].iloc[0]
        df["total_return"] = df["stock_price"] / base_price - 1.0

        return df
    
    def _days_to_expiry(self, index: pd.Index) -> pd.Series:
        expiry = self.contract.expiration
        timedeltas = expiry - index
        return pd.Series([int(td.days) for td in timedeltas], index=index, dtype=int)
    
    def _time_to_expiry(self, index: pd.Index) -> pd.Series:
        expiry = self.contract.expiration
        timedeltas = expiry - index
        return pd.Series([td.total_seconds() / (TRADING_DAYS * 86400) for td in timedeltas], index=index)

    def _rolling_realized_vol(self, log_returns: pd.Series) -> pd.Series:
        window = self.config.realized_vol_window
        vol = log_returns.rolling(window).std() * np.sqrt(252)
        return vol.rename("rolling_realized_vol")

    def _tolerance_from_freq(self) -> pd.Timedelta:
        freq, unit = parse_hedge_freq(self.config.hedge_config['hedge_frequency'])
        if unit =='s':
            return pd.Timedelta(seconds=1)
        if unit =='m':
            minutes = int(freq.total_seconds() // 60)
            return pd.Timedelta(seconds=minutes)
        if unit =='h':
            return pd.Timedelta(minutes=1)
        if unit == 'd':
            return pd.Timedelta(minutes=5)
        raise ValueError(f'Unsupported hedge_frequency unit: {unit}')

    def _attach_option_price(self, stock_df: pd.DataFrame) -> pd.DataFrame: #allow them to set their own tolerance if so desire
        """
        Attaches the option series to the stock series
        """
        
        option_series = self.contract._option_price_series(stock_df.index)
        if option_series is None:
            raise ValueError(
                "Option price data missing. Provide 'option_price' or 'iv' in price_df "
                "or attach option_price_df to the OptionContract."
            )

        opt = option_series.copy().rename("option_price")
        if not isinstance(opt.index, pd.DatetimeIndex):
            opt.index = pd.to_datetime(opt.index)
        opt = opt.sort_index()

        stock = stock_df.sort_index().copy()
        if not isinstance(stock.index, pd.DatetimeIndex):
            stock.index = pd.to_datetime(stock.index)

        tol = self._tolerance_from_freq()

        stock_tmp = stock.reset_index()
        stock_tmp.columns = ["timestamp"] + list(stock_tmp.columns[1:])
        stock_tmp["date"] = stock_tmp["timestamp"].dt.normalize()

        opt_tmp = opt.reset_index()

        opt_tmp.columns = ["timestamp", "option_price"]
        opt_tmp["date"] = opt_tmp["timestamp"].dt.normalize()

        merged = pd.merge_asof(
            stock_tmp.sort_values(["date", "timestamp"]),
            opt_tmp.sort_values(["date", "timestamp"]),
            on="timestamp",
            by="date",
            direction="nearest",
            tolerance=tol,
        )

        merged = merged.set_index("timestamp").drop(columns=["date"])

        merged["true_option_price"] = merged.index.isin(opt.index)
        
        if self.contract.option_mode == 'iv':
            merged["implied_vol"] = merged["option_price"].astype(float)

            if "stock_price" not in merged.columns:
                raise KeyError("stock_df must have 'stock_price' column for IV pricing.")
            
            iv_input = merged[["stock_price", "implied_vol"]].copy()

            solved_prices = self._solve_price_from_iv(iv_input)

            merged["option_price"] = solved_prices




        return merged

    def _resolve_implied_vol(self, df: pd.DataFrame) -> pd.Series:

        iv = pd.Series(np.nan, index=df.index, dtype=float)

        needs_iv = iv.isna()
        if needs_iv.any():
            iv.loc[needs_iv] = self._solve_iv_from_price(df.loc[needs_iv])
        return iv.rename("implied_vol")

    def _solve_price_from_iv(self, df: pd.DataFrame) -> pd.Series:
        """
        to be implented when passing iv series is supported
        """
        from pricing import pricer
        
        prices = []
        expiry = self.contract.expiration
        
        for date, row in df.iterrows():

            if "time_to_expiry" in row.index and not pd.isna(row["time_to_expiry"]):
                time_to_expiry_years = max(row["time_to_expiry"], 0.0)
            else:
                timedelta = expiry - date
                time_to_expiry_years = max(timedelta.total_seconds() / (TRADING_DAYS*86400), 0.0)
            
            iv_value = row.get("iv", row.get("implied_vol", np.nan))
            if pd.isna(iv_value) or iv_value <= 0:
                prices.append((date, np.nan))
                continue
            
            option_price = pricer(
                S=row["stock_price"],
                K=self.contract.strike,
                T=time_to_expiry_years,
                r=self.config.risk_free_rate,
                q=0.0,
                sig=iv_value,
                is_call=self.contract.option_type == "call",
            )
            prices.append((date, option_price))
        
        return pd.Series(dict(prices), name="option_price")
    
    def _solve_iv_from_price(self, df_slice: pd.DataFrame) -> pd.Series:
        """
        solves iv and adds it to the dataframe
        """
        solved = []
        for date, row in df_slice.iterrows():
            time_to_expiry_years = max(row["time_to_expiry"], 0.0)
            solved_iv = self.solver(
                target_price=row["option_price"],
                S=row["stock_price"],
                K=self.contract.strike,
                time=time_to_expiry_years,
                q = 0.0,
                r=self.config.risk_free_rate,
                is_call=self.contract.option_type == "call",
            )
            solved.append((date, solved_iv))
        return pd.Series(dict(solved))


    def _compute_greeks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        gets all of our greeks
        """

        rows = []
        for date, row in df.iterrows():
            time_to_expiry_years = max(row["time_to_expiry"], 0.0)
            greeks = compute_greeks(
                S=row["stock_price"],
                K=self.contract.strike,
                time=time_to_expiry_years,
                vol=max(row["implied_vol"], 1e-8),
                r=self.config.risk_free_rate,
                is_call=self.contract.option_type == "call",
            )
            rows.append(greeks)
        greeks_df = pd.DataFrame(rows, index=df.index)
        return greeks_df

    def _extract_scalar(self, value):
        """Extract scalar value from Series or return value if already scalar.
        """
        if isinstance(value, pd.Series):
            if len(value) == 1:
                return value.iloc[0]
            elif len(value) > 1:
                return value.iloc[0]
            else:
                raise ValueError("Cannot extract scalar from empty Series")
        return value

    def _compute_entry_state(self) -> dict:
        """
        determines entry state of the contract in case the correct entry price is not represented in the option series data
        """

        side_sign = 1 if self.contract.side =='long' else -1
        notional = self.contract.contracts * self.contract.contract_multiplier

        entry_stock = self._extract_scalar(self.contract.stock_price_at_entry)
        if entry_stock is None:
            raise ValueError(
                "OptionContract must include stock_price_at_entry "
                "to compute immediate hedge positions."
            )
        
        entry_date = self.contract.entry_timestamp
        if entry_date is None:
            option_series = self.contract.option_price_series()
            if option_series is not None and len(option_series) > 0:
                entry_date = pd.Timestamp(option_series.index[0]).normalize()
            else:
                raise ValueError("Cannot determine entry_date: contract has no entry_date and no option_price_df")

        timedeltas = self.contract.expiration - entry_date
        days_to_expiry = max((timedeltas).days, 0.0)
    
        time_to_expiry_years = timedeltas.total_seconds() / (TRADING_DAYS*86400)
        
        entry_price = self._extract_scalar(self.contract.entry_price)
        if entry_price is None:
            raise ValueError(
                "OptionContract must include entry_price "
                "to compute immediate hedge positions."
            )
        
        entry_iv = self.solver(
            target_price=entry_price,
            S=entry_stock,
            K=self.contract.strike,
            time=time_to_expiry_years,
            r=self.config.risk_free_rate,
            q = 0.0,
            is_call=self.contract.option_type == "call",
        )
        entry_greeks = compute_greeks(
            S=entry_stock,
            K=self.contract.strike,
            time=time_to_expiry_years,
            vol=max(entry_iv, 1e-8),
            r=self.config.risk_free_rate,
            is_call=self.contract.option_type == "call",
        )

        entry_greeks["option_price"] = entry_price
        entry_greeks["implied_vol"] = entry_iv
        
        entry_greeks["stock_price"] = entry_stock
        entry_greeks['starting_dte'] = days_to_expiry
        entry_greeks['starting_time_to_expiry'] = time_to_expiry_years

        entry_greeks['delta_signed'] = side_sign * entry_greeks['delta']
        entry_greeks['gamma_signed'] = side_sign * entry_greeks['gamma']
        entry_greeks['vega_signed'] = side_sign * entry_greeks['vega']
        entry_greeks['theta_signed'] = side_sign * entry_greeks['theta']
        entry_greeks['rho_signed'] = side_sign * entry_greeks['rho']

        entry_greeks['position_delta'] = notional*entry_greeks['delta_signed']
        entry_greeks['position_gamma'] = notional*entry_greeks['gamma_signed']
        entry_greeks['position_vega'] = notional*entry_greeks['vega_signed']
        entry_greeks['position_theta'] = notional*entry_greeks['theta_signed']
        entry_greeks['position_rho'] = notional*entry_greeks['rho_signed']

        entry_greeks['option_value'] = entry_price*notional*side_sign
        entry_greeks['signed_notional'] = side_sign*notional
        return entry_greeks
    
    def _compute_pnl(self, hedge_df, entry_state) -> pd.DataFrame:
        #computes pnl components and returns the pnl df
        pnl_df = hedge_df

        pnl_df['entry_hedge_shares'] = np.round(-entry_state["position_delta"])
        
        dS = pnl_df["stock_price"].diff().fillna(0.0)
        pnl_df['stock_change'] = dS
        dIV = pnl_df["implied_vol"].diff().fillna(0.0) * 100
        dtime = (-1 * (pnl_df['time_to_expiry'])).diff().fillna(0.0)

        gamma_pnl = 0.5 * pnl_df["position_gamma"] * (dS ** 2)
        theta_pnl = (
            pnl_df["position_theta"] * (dtime*TRADING_DAYS)
        )
        vega_pnl = (
            0.5*(pnl_df["position_vega"] + pnl_df['position_vega'].shift(1)) * dIV
        )

        pnl_df["gamma_pnl"] = gamma_pnl
        pnl_df["theta_pnl"] = theta_pnl
        pnl_df["vega_pnl"] = vega_pnl

        pnl_df["realized_vol_pnl"] = gamma_pnl + theta_pnl
        pnl_df['implied_vol_pnl'] = vega_pnl
        pnl_df["theo_pnl"] = gamma_pnl + vega_pnl + theta_pnl

        pnl_df['stock_return'] = pnl_df['log_return']*100
        pnl_df['total_stock_return'] = pnl_df['total_return']*100

        pnl_df['stock_pnl'] = pnl_df['hedge_shares'].shift(1)*dS

        pnl_df['hedged_pnl'] = pnl_df['stock_pnl'] + pnl_df['option_pnl']
        pnl_df['position_pnl'] = pnl_df['hedged_pnl'].cumsum()
        pnl_df['abs_hedged_pnl'] = abs(pnl_df['hedged_pnl'])

        pnl_df['residual_pnl'] = pnl_df['hedged_pnl'] - pnl_df['theo_pnl']

        pnl_df['interest_rate'] = self.config.risk_free_rate
        pnl_df['interest_pnl'] = (pnl_df['interest_rate']*dtime) * -pnl_df['hedge_shares'] * pnl_df['stock_price']
        
        pnl_df["realized_vol"] = pnl_df["log_return"].expanding(min_periods=2).std() * np.sqrt(TRADING_DAYS)
        pnl_df['abs_move_zscore'] = pnl_df['log_return']/(pnl_df['implied_vol']/math.sqrt(TRADING_DAYS))

        columns_to_return = [
    # 1) Price Data
        'dte',
        'stock_price',
        'stock_change',
        'log_return',
        'total_return',
        'option_price',
        'option_value',

    # 2) Volatility
        'rolling_rv',
        'implied_vol',
        'realized_vol',
        'abs_move_zscore',


    # 3) Option Greeks
    "delta",
    "gamma",
    "theta",
    "vega",
    'rho',

    # 4) Position Weighted Greeks
    'position_delta',
    'position_gamma',
    'position_theta',
    'position_vega',
    'position_rho',
    
    # 5) Actual Pnl
    "hedge_shares",
    "stock_pnl",
    "option_pnl",
    "interest_pnl",
    "hedged_pnl",
    "abs_hedged_pnl",
    'position_pnl',

    # 6) Theoretical Pnl
    'gamma_pnl',
    'realized_vol_pnl',
    'vega_pnl',
    'theo_pnl',
    'residual_pnl' # unexplained pnl
    
]
        existing_columns = [col for col in columns_to_return if col in pnl_df.columns]
        return pnl_df[existing_columns]
    
