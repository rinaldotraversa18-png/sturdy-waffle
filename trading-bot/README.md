# Trading Bot

**A private, AI-powered trading bot for completing Tradeify funded account evaluations.**  
Connects to Tradovate for hands-free execution with built-in risk management that
enforces Tradeify $50k Growth Funded Account rules: $3,000 profit target, $2,000
trailing max EOD drawdown, and $1,250 daily loss limit.

---

## Quick Start

```bash
# 1. Clone and enter the project
cd trading-bot

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux / macOS
# .venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure credentials
cp .env.example .env
# Edit .env — add your Tradovate demo credentials (free account)

# 5. Run!
python -m src.main
```

The bot defaults to **paper trading** on `demo.tradovate.com` — no real orders
are submitted until you deliberately switch to live mode.

---

## How It Works

```
main.py (CLI entry point)
  └─→ BotOrchestrator.run()
       ├── _initialize()      → TradovateClient, RiskEngine, StrategyEngine, StateManager
       ├── _main_loop()       → evaluate stops → generate signals → risk-check → execute
       │    ├── callbacks     → _on_account_update, _on_order_update, _on_quote
       │    └── end-conditions → PASSED (profit target) / FAILED (drawdown) / LOCKED (daily loss)
       └── shutdown()         → persist state, disconnect, release PID lock
```

### Architecture

| Component | Module | Responsibility |
|---|---|---|
| **TradovateClient** | `src/client/` | REST auth + dual WebSocket (trading + market data). Sends orders, streams quotes and account updates. |
| **RiskEngine** | `src/risk/` | Enforces all Tradeify evaluation rules: daily loss limit, trailing EOD drawdown, contract caps. Every order passes through `check_bracket()` before submission. |
| **StrategyEngine** | `src/strategy/` | Generates trade signals from real-time quotes using mean-reversion and breakout logic. Handles position sizing via `calculate_size()`. |
| **Orchestrator** | `src/orchestrator/` | Wires everything together: main loop, WebSocket callback routing, stop-condition evaluation, graceful shutdown. |

### Risk Rules Enforced

1. **Daily Loss Limit ($1,250)** — realised P&L only. Breach → LOCKED for the day.
2. **EOD Trailing Drawdown ($2,000)** — trails peak `net_liq` upward only. Breach → FAILED (permanent).
3. **Profit Target ($3,000)** — cumulative realised P&L. Hit → PASSED.
4. **Contract Limits** — 4 mini contracts or 40 micros per instrument.

---

## Paper Trading (Default)

Out of the box the bot connects to `demo.tradovate.com`.  
**No real orders are placed.**

To switch to **live trading**:

```bash
python -m src.main --live
```

This connects to `live.tradovate.com` and submits real orders against your
live evaluation account. **Use with caution.**

---

## CLI Reference

```
python -m src.main [options]

Options:
  --config CONFIG       Path to YAML config file (default: config/config.yaml)
  --env {demo,live}     Tradovate environment (default: demo)
  --paper               Paper trading mode (default)
  --live                Live evaluation mode — real orders
  --log-level {DEBUG,INFO,WARNING,ERROR}  Log verbosity (default: INFO)
```

---

## Configuration

All configuration lives in `config/config.yaml`. Key parameters:

```yaml
bot:
  loop_interval: 1.0       # seconds between main-loop ticks
  state_path: state.json   # persisted risk state (survives restarts)
  lock_path: bot.lock      # PID lock prevents concurrent instances
  symbols: [MBT, MET]      # contract symbols to trade
```

Strategy and instrument defaults (tick sizes, contract caps) are in
`config/instruments.yaml`.

Secrets (API credentials) must go in `.env` — never commit them:

```
TRADOVATE_USERNAME=your_username
TRADOVATE_PASSWORD=your_password
TRADOVATE_APP_ID=your_app_id
TRADOVATE_DEVICE_ID=your_device_id
```

---

## Testing

```bash
# Run the full test suite
pytest tests/ -q

# Run only integration tests
pytest tests/test_integration.py -v

# Run only unit tests (exclude integration)
pytest tests/ -q --ignore=tests/test_integration.py
```

All tests use mocks — no network access required.

The **integration tests** in `tests/test_integration.py` serve as living
documentation. Read them to understand the bot's behaviour end to end:
- `test_full_evaluation_lifecycle` — startup → profitable trades → PASSED
- `test_daily_loss_stops_bot` — losing trades → LOCKED
- `test_drawdown_stops_bot` — equity drop → FAILED
- `test_contract_limit_rejects_oversized_order` — oversized orders blocked
- `test_state_persistence_roundtrip` — save/load produces identical state
- `test_session_reset_on_new_day` — daily counters reset, cumulative kept

---

## Requirements

- **Python 3.11+**
- **Tradovate demo account** (free — [sign up here](https://www.tradovate.com/))
- Dependencies listed in `requirements.txt`

---

## Project Structure

```
src/
├── client/         Tradovate REST + WebSocket API client
├── risk/           Risk engine (daily loss, drawdown, contract limits)
├── strategy/       Signal generation (mean-reversion + breakout) and position sizing
├── orchestrator/   Main bot loop, callback routing, lifecycle management
└── utils/          Structured logging, market clock, helpers

config/
├── config.yaml         Bot loop intervals, state paths, log config
└── instruments.yaml    Per-instrument specs (tick size, point value)

tests/
├── conftest.py             Shared fixtures and MockTradovateClient
├── test_models.py          Pydantic model validation
├── test_risk_engine.py     Risk engine rule enforcement
├── test_strategy_engine.py Signal generation and sizing
├── test_tradovate_client.py Client REST + WebSocket
├── test_orchestrator.py    BotOrchestrator lifecycle (22 tests)
├── test_integration.py     End-to-end pipeline (9 tests)
└── test_ws_manager.py      WebSocket connection management

logs/               Structured JSON logs (rotated daily)
```

---

## License

Proprietary. All rights reserved.
