from __future__ import annotations

from dataclasses import dataclass, field

##########################
# Configuration Class
##########################
@dataclass(frozen=True)
class PnLModelConfig:
    risk_free_rate: float = 0.0
    realized_vol_window: int = 5

    hedge_config: dict = field(
        default_factory=lambda: {
            'hedge_frequency': '1d',
#            'hedge_at_open': False,
            'delta_target': 0, # Weighted by the underlying so delta_target:100 means target is 100 shares of underlying worth of directional exposure
#            'use_delta_band': False,
#           'delta_band': 0.4, # will need to add high and low band as seperate inputs
#            'entry_exit_mode': 'synthetic', # can also be quote_only where we only trade based on the true data
        }
    )
    # transaction_costs: float = 0.0 #per trade transaction cost in basis points of notional