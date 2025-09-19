from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import math
from datetime import date, datetime, time, timedelta
from typing import Optional

import pytest

from features.news_loader_contract import NewsArticle
from features.sentiment_features import (
    SimpleSentimentIntensityAnalyzer,
    build_daily_sentiment_features,
    prepare_sentiment_features,
)


class DummyScorer(SimpleSentimentIntensityAnalyzer):
    """Deterministic scorer mapping keywords to fixed polarity values."""

    def __init__(self) -> None:
        super().__init__()
        self.mapping = {
            "good": 0.8,
            "great": 0.9,
            "bad": -0.5,
            "neutral": 0.0,
        }

    def polarity_scores(self, text: str) -> dict[str, float]:  # type: ignore[override]
        text_lower = text.lower()
        for keyword, score in self.mapping.items():
            if keyword in text_lower:
                return {"compound": score}
        return {"compound": 0.0}


@pytest.fixture
def trading_days() -> list[datetime]:
    start = datetime(2021, 1, 4)
    return [start + timedelta(days=idx) for idx in range(4) if (start + timedelta(days=idx)).weekday() < 5]


def test_articles_bucketed_and_lagged(trading_days: list[datetime]) -> None:
    scorer = DummyScorer()
    articles = [
        NewsArticle("TEST", datetime(2021, 1, 4, 15, 30), headline="good news"),
        NewsArticle("TEST", datetime(2021, 1, 4, 16, 30), headline="bad result"),
        NewsArticle("TEST", datetime(2021, 1, 5, 10, 0), headline="great update"),
    ]

    features = build_daily_sentiment_features(
        articles,
        trading_days,
        market_close=time(16, 0),
        lag_bars=1,
        scorer=scorer,
    )

    day_list = list(features.keys())
    assert features[day_list[0]].sent_mean is None
    assert features[day_list[1]].sent_count == 1
    assert math.isclose(features[day_list[1]].sent_mean or 0.0, 0.8, rel_tol=1e-9)
    assert features[day_list[2]].sent_count == 2
    assert math.isclose(features[day_list[2]].sent_mean or 0.0, 0.2, rel_tol=1e-9)


def _compute_expected_ema(values: list[Optional[float]], span: int) -> list[Optional[float]]:
    alpha = 2.0 / (span + 1.0)
    ema: Optional[float] = None
    result: list[Optional[float]] = []
    for value in values:
        if value is None:
            result.append(ema)
            continue
        ema = value if ema is None else alpha * value + (1 - alpha) * ema
        result.append(ema)
    return result


def _compute_expected_z(values: list[Optional[float]], window: int, min_periods: int) -> list[Optional[float]]:
    window_values: list[float] = []
    result: list[Optional[float]] = []
    for value in values:
        if value is None:
            result.append(None)
            continue
        window_values.append(value)
        if len(window_values) > window:
            window_values.pop(0)
        if len(window_values) < min_periods:
            result.append(None)
            continue
        mean = sum(window_values) / len(window_values)
        variance = sum((val - mean) ** 2 for val in window_values) / len(window_values)
        std = math.sqrt(variance)
        if std == 0:
            result.append(0.0)
        else:
            result.append((value - mean) / std)
    return result


def test_ema_and_zscore_follow_expected_formulas() -> None:
    scorer = DummyScorer()
    trading_days = [datetime(2021, 1, d) for d in range(4, 9)]
    articles = [
        NewsArticle("TEST", datetime(2021, 1, 4, 10, 0), headline="good"),
        NewsArticle("TEST", datetime(2021, 1, 5, 10, 0), headline="bad"),
        NewsArticle("TEST", datetime(2021, 1, 6, 10, 0), headline="good"),
        NewsArticle("TEST", datetime(2021, 1, 7, 10, 0), headline="great"),
        NewsArticle("TEST", datetime(2021, 1, 8, 10, 0), headline="neutral"),
    ]

    raw = build_daily_sentiment_features(
        articles,
        trading_days,
        market_close=time(16, 0),
        lag_bars=0,
        scorer=scorer,
    )
    shifted = build_daily_sentiment_features(
        articles,
        trading_days,
        market_close=time(16, 0),
        lag_bars=1,
        scorer=scorer,
    )

    raw_means = [row.sent_mean for row in raw.values()]
    expected_ema = _compute_expected_ema(raw_means, span=5)
    expected_ema_shifted = [expected_ema[idx - 1] if idx - 1 >= 0 else None for idx in range(len(expected_ema))]
    for row, expected in zip(shifted.values(), expected_ema_shifted):
        if expected is None:
            assert row.sent_ema_5 is None
        else:
            assert math.isclose(row.sent_ema_5 or 0.0, expected, rel_tol=1e-9)

    expected_z = _compute_expected_z(raw_means, window=20, min_periods=5)
    expected_z_shifted = [expected_z[idx - 1] if idx - 1 >= 0 else None for idx in range(len(expected_z))]
    for row, expected in zip(shifted.values(), expected_z_shifted):
        if expected is None:
            assert row.sent_z_20 is None
        else:
            assert math.isclose(row.sent_z_20 or 0.0, expected, rel_tol=1e-9)


def test_prepare_sentiment_features_filters_future_articles() -> None:
    scorer = DummyScorer()
    trading_days = [datetime(2021, 1, 4) + timedelta(days=idx) for idx in range(3)]

    def loader(ticker: str, start: datetime, end: datetime):
        assert end.date() == trading_days[-1].date()
        return [
            NewsArticle(ticker, datetime(2021, 1, 1, 12, 0), headline="good"),
            NewsArticle(ticker, datetime(2021, 1, 10, 12, 0), headline="great"),
        ]

    features = prepare_sentiment_features(
        "TEST",
        trading_days,
        loader,
        start=trading_days[0],
        end=trading_days[-1],
        market_close=time(16, 0),
        lag_bars=1,
        scorer=scorer,
    )

    day_list = list(features.keys())
    assert features[day_list[1]].sent_count == 1
    assert features[day_list[2]].sent_count == 0


def test_recency_weighting_respects_market_close() -> None:
    scorer = DummyScorer()
    trading_days = [datetime(2021, 1, 4) + timedelta(days=idx) for idx in range(2)]
    articles = [
        NewsArticle("TEST", datetime(2021, 1, 3, 18, 0), headline="bad"),
        NewsArticle("TEST", datetime(2021, 1, 4, 9, 0), headline="good"),
    ]

    features = build_daily_sentiment_features(
        articles,
        trading_days,
        market_close=time(16, 0),
        lag_bars=0,
        scorer=scorer,
        decay_half_life=1.0,
    )

    session = trading_days[0].date()
    row = features[session]
    assert row.sent_weighted is not None

    session_anchor = datetime.combine(session, time(16, 0))
    age_bad = (session_anchor - datetime(2021, 1, 3, 18, 0)).total_seconds() / 86400.0
    age_good = (session_anchor - datetime(2021, 1, 4, 9, 0)).total_seconds() / 86400.0
    weight_bad = math.exp(-math.log(2.0) * age_bad)
    weight_good = math.exp(-math.log(2.0) * age_good)
    expected = (weight_bad * (-0.5) + weight_good * 0.8) / (weight_bad + weight_good)

    assert math.isclose(row.sent_weighted, expected, rel_tol=1e-9)
