# FinEnv-Rebalancer: AI-Driven Portfolio Management

## 📊 Overview

FinEnv-Rebalancer is a high-fidelity OpenEnv simulation designed to train and evaluate AI agents on **Multi-Asset Portfolio Rebalancing**. In the real world, institutional investors and hedge funds lose millions to "drift"—when market movements push a portfolio away from its target risk profile.

This environment challenges agents to maintain a specific 50/50 Cash-to-Asset ratio across volatile assets (BTC, ETH, GOLD, SPY) using real-world trading mechanics.

## 🛠️ Action & Observation Spaces

### Action Space (Pydantic: `Action`)

Agents interact with the brokerage engine using a structured JSON schema:

- `action_type`: One of `buy`, `sell`, or `hold`.
- `asset_id`: The ticker symbol of the asset.
- `quantity`: The amount to trade (must be non-negative).

### Observation Space (Pydantic: `Observation`)

At each step, the agent receives:

- `prices`: Live market tickers for all assets.
- `holdings`: Current units owned per asset.
- `cash`: Available USD liquidity.
- `portfolio_value`: Total Net Liquidation Value (NLV).
- `step_count`: Current progress in the trading day.

## 🎯 Tasks & Difficulty

| Task ID   | Name             | Difficulty | Description                                                            |
| :-------- | :--------------- | :--------- | :--------------------------------------------------------------------- |
| `task_01` | Static Rebalance | **Easy**   | Start with 100% cash. Allocate into a 50/50 split.                     |
| `task_02` | Drift Correction | **Medium** | Start with a heavily skewed portfolio and rebalance during volatility. |
| `task_03` | Crash Survival   | **Hard**   | Maintain ratio during a simulated 20% market drawdown.                 |

## 📈 Reward Design

The environment uses **Dense Reward Shaping** to provide a constant signal:

- **Alignment Reward:** $1.0 - |(\text{Actual Ratio} - \text{Target Ratio})|$.
- **Success:** Rewards are maximized (1.0) when the agent hits the perfect 0.5 balance.
- **Fail-Safe:** Episodes terminate early if the portfolio value drops below 70% of initial capital.

## 🚀 Setup & Usage

### Local Testing

1. Install dependencies: `pip install -r requirements.txt`
2. Run the server: `uvicorn app:app --host 0.0.0.0 --port 7860`
3. Test health: `curl http://localhost:7860/health`

### Deployment

This environment is containerized via **Docker** and ready for **Hugging Face Spaces**.

- Tag: `openenv`
- Port: `7860`
- Secret required: `GEMINI_API_KEY` (for baseline inference).
