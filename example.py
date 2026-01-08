import config
import contracts
import single_contract
import portfolio

import pandas as pd

#####################################
# Reading in example data 
# from csv and converted to pd.Series
#####################################
call_df = pd.read_csv("example_data/tsla_call.csv")
call_series = pd.Series(call_df["option_price"].values, index=call_df["date"])

put_df = pd.read_csv("example_data/tsla_put.csv")
put_series = pd.Series(put_df["option_price"].values, index=put_df["date"])

stock_df = pd.read_csv("example_data/tsla_stock.csv")
stock_series = pd.Series(stock_df["stock_price"].values, index=stock_df["timestamp"])

####################################
# Initializing option contracts
####################################
strike = 380
expiry = pd.Timestamp('2025-12-19').normalize()
configuration = config.PnLModelConfig(risk_free_rate=0.0, realized_vol_window=5)

put = contracts.OptionContract(
    id = 'tsla_put',
    strike=380,
    expiration=expiry,
    entry_timestamp=pd.Timestamp('2025-02-03').normalize(),
    exit_timestamp=pd.Timestamp('2025-11-25').normalize(),
    entry_price=put_series.iloc[4], # must account for entry price not being first value of series
    stock_price_at_entry=stock_series.iloc[4], # same process so vega pnl doesn't blow up
    option_price=put_series,

    option_type='put',
    contracts=1,
    side='long',
    contract_multiplier=100.0
)

call = contracts.OptionContract(
    id = 'tsla_call',
    strike=380,
    expiration=expiry,
    entry_timestamp=pd.Timestamp('2025-01-28').normalize(),
    exit_timestamp=pd.Timestamp('2025-11-20').normalize(),
    entry_price=call_series.iloc[0], 
    stock_price_at_entry=stock_series.iloc[0],
    option_price=call_series,

    option_type='call',
    contracts=1,
    side='long',
    contract_multiplier=100.0
)
#####################################
# Calling and running single contract engines
#####################################
"""
Note: I find it much easier to review decomps in an excel file
"""
call_engine = single_contract.SingleContractPnL(stock_series, call, configuration)
call_path, call_pnl = call_engine.run()
print(call_pnl)
#call_pnl.to_excel('call_decomp.xlsx')

put_engine = single_contract.SingleContractPnL(stock_series, put, configuration)
put_path, put_pnl = put_engine.run()
print(put_pnl)
#put_pnl.to_excel('put_decomp.xlsx')

#####################################
# Calling and running aggregate portfolio engine
#####################################
port_engine = portfolio.PortfolioPnL(stock_series, [call, put], configuration)
port_path, port_pnl = port_engine.run()
print(port_pnl)
#port_pnl.to_excel('port_decomp.xlsx')
