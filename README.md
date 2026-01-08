# Delta-Hedged Option P&L Attribution Engine

A Python engine for backtesting delta-hedged options strategies and decomposing P&L into theoretical and realized volatility components under explicit hedging rules.

---

## Overview

This project implements a systematic framework for analyzing option strategies with discrete delta hedging and Greek-based P&L attribution.

It supports:

- Single-option and multi-contract portfolios
- Explicit delta targets and discrete hedging intervals
- Per-contract option price time series
- Transparent P&L decomposition consistent with common sell-side and buy-side conventions

The engine prioritizes **correctness, transparency, and extensibility** over execution speed.

---

## Core Features

- Delta-hedged P&L simulation under configurable hedging rules
- Greek-based P&L decomposition:
  - Delta (hedged stock P&L)
  - Gamma (curvature exposure)
  - Theta (time decay)
  - Vega (implied volatility exposure)
  - Realized vs implied volatility P&L
- Correct sign handling for long and short option positions
- Portfolio-level aggregation across multiple contracts on a single underlying
- Clean separation between contract definition, pricing, and P&L logic

---

## Architecture

### Core Components

- **`OptionContract`**
  - Defines option metadata (type, strike, expiry)
  - Stores entry/exit timestamps and entry prices
  - Encodes position side (long / short)

- **`SingleContractPnL`**
  - Aligns option data to stock data
  - Computes Greeks, hedge positions, and realized P&L paths
  - Produces full attribution for a single contract

- **`PortfolioPnL`**
  - Aggregates multiple contract paths
  - Computes portfolio-level Greeks and P&L attribution

- **`pricing/`**
  - Black–Scholes pricing and Greek utilities
  - Custom pricing functions may be substituted if they follow the same interface

### Data Flow

1. Contracts are defined and passed into the engine
2. Option data is aligned to the stock series
3. Option paths are constructed with full Greeks (IV forward-filled when missing)
4. Delta hedges are computed per interval
5. Realized P&L is computed from option prices and stock changes
6. Theoretical P&L is computed from Greeks
7. Residual P&L captures the difference

---

## API Overview

### `OptionContract`
Defines:
- Option type, strike, and expiration
- Entry and exit timestamps
- Entry option price and stock price
- Position side (long / short)

### `SingleContractPnL`
Inputs:
- Stock price series (`pd.Series` with `DatetimeIndex`)
- `OptionContract` instance

Outputs:
- DataFrame containing:
  - Greeks
  - Hedge positions
  - Realized P&L
  - Theoretical P&L components
  - Residual P&L

### `PortfolioPnL`
- Aggregates multiple `SingleContractPnL` paths
- Assumes a **single underlying asset**
- Produces portfolio-level Greeks and P&L attribution

---

## P&L Decomposition Methodology

P&L is decomposed using low-order Greek approximations per hedge interval:

- **Delta P&L**  
  Stock P&L from the delta-hedged position

- **Gamma P&L**  
    $$
    \frac{1}{2}\,\Gamma\,(\Delta S)^2
    $$

- **Theta P&L**  
  Time decay accrued over the hedge interval

- **Vega P&L**  
  Change in option value due to implied volatility moves

- **Realized Volatility P&L**  
  Defined as:
    Gamma P&L + Theta P&L:

  This follows the discrete-time replication identity under Black–Scholes assumptions; alternative decompositions may differ across practitioners.

- **Residual P&L**  
  Captures discretization error, higher-order effects, and model mismatch arising from discrete hedging and pricing assumptions.

### Conventions

- Vega P&L is computed **per contract** before portfolio aggregation
- Time-to-expiry uses **365 days per year**
- Position Greeks are expressed in units of the underlying  
  (e.g. `position_delta = 1` corresponds to one share)
- All theoretical calculations use the Black–Scholes model
- Hedging occurs at discrete timestamps; continuous rebalancing is not assumed

### Assumptions

- Stock and option prices are pandas Series indexed by timestamps
- The stock series contains **all timestamps** used by every option series
- Option prices are required as inputs  
  (native IV-series ingestion is planned but not yet supported)

---

## Input Data Format

### Stock Series
Requirements:
- Index — pandas Timestamp
- `stock_price` — float

The stock series must span the full lifetime of every option contract.

### Option Series
Requirements:
- Index — pandas Timestamp
- `option_price` — float 

Notes:
- Option prices may be sparse and are forward-filled after alignment
- Option prices must be the true value divided by contract multiplier ex 10.00 option_price is valued at $1000 for an equities contract
- All timestamps must fall within the stock series range

---

## Installation & Quickstart

Requires Python 3.10+.

```bash
git clone https://github.com/miles-vollet8/delta_hedge
cd delta_hedge
pip install -r requirements.txt
python examples/example.py
```
- The example script loads in stock and option data from examples/data and creates full P&L attribution DataFrame for a long call, put, and straddle
## Limitations
- No transaction costs or bid-ask spread modeling
- Constant interest rates and no dividends
- Assumes Black-Scholes dynamics
- Designed for research and attribution, not execution or live trading

## Roadmap
- Native implied volatility series handling
- Intraday hedging with delta bands and transaction costs
- Higher-order volatility Greeks (vanna, volga)
- Volatility surface support

## Disclaimer
- This project is for research and educational purposes. It is NOT intended as investment advice

