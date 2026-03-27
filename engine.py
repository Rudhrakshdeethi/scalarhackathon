import random
from models import Action, Observation

class FinanceEngine:
    def __init__(self):
        self.assets = ["BTC", "ETH", "GOLD", "SPY"]
        self.done = False
        self.current_task_id = "task_01"
        self.reset()

    def reset(self, task_id: str = "task_01"):
        self.current_task_id = task_id
        self.prices = {asset: random.uniform(100, 500) for asset in self.assets}
        self.holdings = {asset: 0.0 for asset in self.assets}
        self.cash = 10000.0
        self.step_count = 0
        self.max_steps = 20
        self.done = False
        
        if task_id == "task_02":
            self.holdings["BTC"] = 20.0 # Start with drift
            self.cash = 2000.0
            
        return self._get_obs()

    def _get_obs(self) -> Observation:
        total_val = self.cash + sum(self.holdings[a] * self.prices[a] for a in self.assets)
        return Observation(
            prices=self.prices.copy(),
            holdings=self.holdings.copy(),
            cash=round(self.cash, 2),
            portfolio_value=round(total_val, 2),
            step_count=self.step_count
        )

    def step(self, action: Action):
        for asset in self.assets:
            self.prices[asset] *= (1 + random.uniform(-0.02, 0.02))

        price = self.prices.get(action.asset_id, 0)
        cost = price * action.quantity
        
        if action.action_type == "buy" and self.cash >= cost:
            self.holdings[action.asset_id] += action.quantity
            self.cash -= cost
        elif action.action_type == "sell" and self.holdings.get(action.asset_id, 0) >= action.quantity:
            self.holdings[action.asset_id] -= action.quantity
            self.cash += cost

        obs = self._get_obs()
        asset_ratio = (obs.portfolio_value - obs.cash) / obs.portfolio_value if obs.portfolio_value > 0 else 0
        reward = 1.0 - abs(asset_ratio - 0.5)

        self.step_count += 1
        if self.step_count >= self.max_steps or obs.portfolio_value < 7000:
            self.done = True
        
        return obs, reward, self.done