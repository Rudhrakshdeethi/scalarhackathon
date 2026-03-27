from pydantic import BaseModel, Field
from typing import Dict, List, Literal

class Action(BaseModel):
    action_type: Literal["buy", "sell", "hold"] = Field(..., description="Trade operation")
    asset_id: str = Field(..., description="Ticker symbol (BTC, ETH, GOLD, SPY)")
    quantity: float = Field(default=0.0, description="Amount to trade")

class Observation(BaseModel):
    prices: Dict[str, float]
    holdings: Dict[str, float]
    cash: float
    portfolio_value: float
    step_count: int

class Reward(BaseModel):
    value: float = Field(..., ge=-1.0, le=1.0)
    reason: str