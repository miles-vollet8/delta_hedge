from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import webbrowser
import os
import subprocess
import sys
import re
from datetime import timedelta

from config import PnLModelConfig


############################
# Unused until intraday hedging is supported
############################
def parse_hedge_freq(freq: str) -> tuple[timedelta, str]:
    """
    Parse strings like '10s', '5m', '2h', '1d' into a timedelta.
    Enforces:
    - 1–59s
    - 1–59m
    - 1–4h
    - 1–30d
    """
    if not isinstance(freq, str):
        raise ValueError(f"hedge_frequency must be a string, got {type(freq)}")

    m = re.fullmatch(r"(\d+)([smhd])", freq.strip())
    if not m:
        raise ValueError(f"Invalid hedge_frequency format: {freq!r}. Expected e.g. '5m', '1h', '1d'.")

    value = int(m.group(1))
    unit = m.group(2)

    if unit == "s" and not (1 <= value <= 59):
        raise ValueError("Seconds hedge_frequency must be between 1s and 59s.")
    if unit == "m" and not (1 <= value <= 59):
        raise ValueError("Minutes hedge_frequency must be between 1m and 59m.")
    if unit == "h" and not (1 <= value <= 4):
        raise ValueError("Hours hedge_frequency must be between 1h and 4h.")
    #if unit == "d" and not (1 <= value <= 30): 
    if unit == "d" and not (1 <= value <= 1): # only doing 1 day for now
        raise ValueError("Days hedge_frequency must be between 1d and 30d.")

    # map to timedelta
    if unit == "s":
        return timedelta(seconds=value), unit
    if unit == "m":
        return timedelta(minutes=value), unit
    if unit == "h":
        return timedelta(hours=value), unit
    if unit == "d":
        return timedelta(days=value), unit
    
    raise ValueError(f"Unsupported hedge_frequency unit: {unit!r}")

def flag_hedge_timestamps(index: pd.DatetimeIndex, hedge_config: dict) -> pd.Series:
    """
    Takes in the path index and adds appropriate hedge flags
    Again not really used until intraday hedging is supported
    """


    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("index must be a DatetimeIndex")
    freq, unit = parse_hedge_freq(hedge_config['hedge_frequency'])

    hedge_flag = pd.Series(False, index=index)

    if unit in ('s', 'm', 'h'):
        for day, idx_day in index.to_series().groupby(index.normalize()):
            # idx_day is a Series of timestamps for that day
            ts_list = idx_day.index
            if len(ts_list) == 0:
                continue

            current_target = ts_list[0]  # hedge at first tick of the day
            if hedge_config['hedge_at_open'] ==True:
                hedge_flag.loc[current_target] = True

            # Step forward in calendar time, but align to actual ticks
            while True:
                next_time = current_target + freq
                # first timestamp >= next_time
                candidates = ts_list[ts_list >= next_time]
                if len(candidates) == 0:
                    break
                current_target = candidates[0]
                hedge_flag.loc[current_target] = True
            close_ts = ts_list[-1]
            hedge_flag.loc[close_ts] = True
    elif unit =='d':
        dates = index.normalize()
        unique_dates = dates.unique().sort_values()
        if len(unique_dates) ==0:
            return hedge_flag
        step = freq.days
        if step <= 0:
            raise ValueError(f"Invalid frequency {freq!r} for hedge_frequency={hedge_config['hedge_frequency']!r}")
        hedge_days = unique_dates[::step]

        for day in hedge_days:
            intraday = (dates == day)
            day_ts = index[intraday]
            if len(day_ts) == 0:
                continue
            close_ts = day_ts[-1]
            if hedge_config.get('hedge_at_open', False):
                hedge_flag.loc[day_ts[0]] = True
            hedge_flag.loc[close_ts] = True
    return hedge_flag


        
def compute_hedges(path_df, config=None) -> pd.DataFrame: #want a column with total shares and one with changes at each step
    """
    Creates the hedge_df by
    adding the appropriate number of hedge shares based on the target_delta
    """

    if config is None:
        config = PnLModelConfig()
    hedge_config = config.hedge_config
    hedge_flag = flag_hedge_timestamps(path_df.index, hedge_config)
    path_df['hedge_flag'] = hedge_flag.reindex(path_df.index, fill_value=False)
    path_df['hedge_shares'] = np.where(
        path_df['hedge_flag'],
        -np.round((path_df['position_delta']-hedge_config['delta_target'])), #rounds to nearest whole because who's out here using perfect fractionals
        np.nan
    )
    path_df['hedge_shares'] = path_df['hedge_shares'].ffill().fillna(0)
    path_df['hedge_change'] = path_df['hedge_shares'].diff()
    return path_df