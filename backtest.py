"""Simple backtest driver enhanced with historical sentiment features."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from random import Random
from typing import Dict, Iterable, List, Optional, Sequence

from features.news_loader_contract import NewsArticle, NewsLoader
from features.sentiment_features import (
    SentimentFeatureRow,
    get_default_sentiment_scorer,
    prepare_sentiment_features,
)
from plots.compare_returns import plot_strategy_vs_buy_and_hold

DATA_DIR = Path("data")
PRICE_FILE = DATA_DIR / "prices.csv"
NEWS_FILE = DATA_DIR / "news.csv"
DEFAULT_MARKET_CLOSE = time(16, 0)


@dataclass
class PriceBar:
    date: datetime
    close: float


@dataclass
class BacktestResults:
    prices: List[PriceBar]
    features: Dict[date, SentimentFeatureRow]
    signal: List[float]
    strategy_returns: List[float]
    equity_curve: List[float]
    buy_and_hold_equity: List[float]


def _business_days(start: datetime, end: datetime) -> List[datetime]:
    days: List[datetime] = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(datetime.combine(current.date(), time(0, 0)))
        current += timedelta(days=1)
    return days


def load_prices(
    ticker: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> List[PriceBar]:
    """Load historical prices for a ticker.

    The loader expects a CSV file with columns ``date``, ``ticker`` and ``close``.
    When no file is present a deterministic synthetic series is produced to keep
    the example self-contained.
    """

    if PRICE_FILE.exists():
        prices: List[PriceBar] = []
        with PRICE_FILE.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("ticker", "").upper() != ticker.upper():
                    continue
                close = float(row["close"])
                date_value = datetime.fromisoformat(row["date"])  # type: ignore[arg-type]
                prices.append(PriceBar(date=date_value, close=close))
        prices.sort(key=lambda bar: bar.date)
    else:  # pragma: no cover - convenience path for interactive use
        start_dt = start or datetime(2020, 1, 1)
        end_dt = end or datetime(2020, 3, 31)
        days = _business_days(start_dt, end_dt)
        rng = Random(0)
        price = 100.0
        prices = []
        for day in days:
            price *= 1.0 + rng.gauss(0.0005, 0.02)
            prices.append(PriceBar(date=day, close=price))

    if start is not None:
        prices = [bar for bar in prices if bar.date >= start]
    if end is not None:
        prices = [bar for bar in prices if bar.date <= end]

    return prices


def load_historical_news(
    ticker: str,
    start: datetime,
    end: datetime,
) -> Iterable[NewsArticle]:
    """Load historical news articles adhering to the no-lookahead contract."""

    if not NEWS_FILE.exists():  # pragma: no cover - optional convenience
        return []

    articles: List[NewsArticle] = []
    with NEWS_FILE.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("ticker", "").upper() != ticker.upper():
                continue
            published = datetime.fromisoformat(row["published"])
            if published > end:
                continue
            if published < start - timedelta(days=30):
                continue
            articles.append(
                NewsArticle(
                    ticker=row["ticker"],
                    published=published,
                    headline=row.get("headline", ""),
                    content=row.get("content", ""),
                    source=row.get("source"),
                )
            )
    articles.sort(key=lambda article: article.published)
    return articles


def _moving_average(values: Sequence[float], window: int) -> List[float]:
    result: List[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        window_values = values[start : idx + 1]
        result.append(sum(window_values) / len(window_values))
    return result


def _compute_equity_curve(initial_capital: float, returns: Sequence[float]) -> List[float]:
    equity = initial_capital
    curve: List[float] = []
    for value in returns:
        equity *= 1.0 + value
        curve.append(equity)
    return curve


def run_backtest(
    ticker: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    *,
    initial_capital: float = 100_000.0,
    market_close: time = DEFAULT_MARKET_CLOSE,
    plot_path: Optional[Path | str] = None,
    news_loader: NewsLoader = load_historical_news,
) -> BacktestResults:
    """Run the backtest pipeline and return intermediate artefacts."""

    prices = load_prices(ticker, start=start, end=end)
    if not prices:
        raise ValueError("No price data available for the requested window.")

    sentiment_features = prepare_sentiment_features(
        ticker,
        [bar.date for bar in prices],
        news_loader,
        start=prices[0].date,
        end=prices[-1].date,
        market_close=market_close,
        lag_bars=1,
        scorer=get_default_sentiment_scorer(),
    )

    closes = [bar.close for bar in prices]
    returns: List[float] = [0.0]
    for idx in range(1, len(closes)):
        prev = closes[idx - 1]
        curr = closes[idx]
        returns.append(curr / prev - 1.0)

    short_ma = _moving_average(closes, window=10)
    long_ma = _moving_average(closes, window=30)
    signal = [1.0 if s > l else 0.0 for s, l in zip(short_ma, long_ma)]

    strategy_returns: List[float] = []
    prev_signal = 0.0
    for daily_return, daily_signal in zip(returns, signal):
        strategy_returns.append(daily_return * prev_signal)
        prev_signal = daily_signal

    equity_curve = _compute_equity_curve(initial_capital, strategy_returns)
    buy_and_hold_equity = _compute_equity_curve(initial_capital, returns)

    plot_strategy_vs_buy_and_hold(
        stock_returns=returns,
        strategy_returns=strategy_returns,
        title=f"{ticker} Strategy vs Buy & Hold",
        save_path=plot_path,
        show=False,
    )

    strategy_net_return = equity_curve[-1] / initial_capital - 1.0
    buy_hold_net_return = buy_and_hold_equity[-1] / initial_capital - 1.0
    print(f"Strategy net return: {strategy_net_return:.2%}")
    print(f"Buy & hold net return: {buy_hold_net_return:.2%}")

    return BacktestResults(
        prices=prices,
        features=sentiment_features,
        signal=signal,
        strategy_returns=strategy_returns,
        equity_curve=equity_curve,
        buy_and_hold_equity=buy_and_hold_equity,
    )


if __name__ == "__main__":  # pragma: no cover - manual execution entry-point
    run_backtest("AAPL")
