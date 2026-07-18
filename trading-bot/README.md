# Trading Bot

Private AI-powered crypto trading bot for completing Tradeify funded account evaluations. Connects to Tradovate for hands-free execution with built-in risk management that enforces evaluation rules (daily loss limit, trailing drawdown, consistency rule).

## Prerequisites

- Python 3.11 or later
- A Tradovate account (demo or live)

## Setup

1. Clone the repository and navigate to the project directory:
   ```bash
   cd trading-bot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # .venv\Scripts\activate   # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env` and fill in your Tradovate credentials:
   ```bash
   cp .env.example .env
   ```

## Usage

Start the bot:
```bash
   python -m src.main
   ```

## Configuration

Configuration is managed through environment variables (see `.env.example`). Additional strategy and instrument settings live in `config/`.

## Project Structure

```
src/
├── client/         Tradovate REST + WebSocket API client
├── risk/           Evaluation rule enforcement engine
├── strategy/       Signal generation and position sizing
├── orchestrator/   Main bot loop and component coordination
└── utils/          Logging, clock, and helpers
config/             Strategy and instrument configuration
tests/              Test suite
logs/               Structured JSON logs
```

## License

Proprietary. All rights reserved.
