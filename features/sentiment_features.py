"""Build daily sentiment features from historical news without lookahead."""
from __future__ import annotations

import math
import re
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Protocol, Sequence

from .news_loader_contract import NewsArticle

try:  # pragma: no cover - optional dependency
    from nltk.sentiment import SentimentIntensityAnalyzer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    SentimentIntensityAnalyzer = None  # type: ignore


class SentimentScorer(Protocol):
    """Protocol describing a polarity scoring object."""

    def polarity_scores(self, text: str) -> Mapping[str, float]:
        """Return sentiment polarity values."""


@dataclass
class SimpleSentimentIntensityAnalyzer:
    """Deterministic fallback sentiment scorer used when VADER is unavailable."""

    positive_words: Sequence[str] = (
        "beat",
        "bull",
        "gain",
        "good",
        "growth",
        "improve",
        "positive",
        "profit",
        "surge",
        "up",
    )
    negative_words: Sequence[str] = (
        "bad",
        "bear",
        "cut",
        "decline",
        "drop",
        "fall",
        "loss",
        "miss",
        "negative",
        "down",
    )
    _word_pattern: re.Pattern[str] = re.compile(r"[A-Za-z']+")

    def polarity_scores(self, text: str) -> Mapping[str, float]:  # type: ignore[override]
        tokens = self._word_pattern.findall(text.lower())
        if not tokens:
            return {"compound": 0.0}
        positive = sum(token in self.positive_words for token in tokens)
        negative = sum(token in self.negative_words for token in tokens)
        total = positive + negative
        if total == 0:
            return {"compound": 0.0}
        score = (positive - negative) / total
        return {"compound": float(max(min(score, 1.0), -1.0))}


_vader_scorer: Optional[SentimentScorer] = None


def get_default_sentiment_scorer() -> SentimentScorer:
    """Return a default sentiment scorer, preferring NLTK's VADER."""

    global _vader_scorer
    if _vader_scorer is not None:
        return _vader_scorer

    if SentimentIntensityAnalyzer is not None:  # pragma: no branch - optional dependency
        try:
            _vader_scorer = SentimentIntensityAnalyzer()
            return _vader_scorer
        except Exception:
            pass

    _vader_scorer = SimpleSentimentIntensityAnalyzer()
    return _vader_scorer


FEATURE_COLUMNS = (
    "sent_mean",
    "sent_weighted",
    "sent_count",
    "sent_ema_5",
    "sent_z_20",
)


@dataclass
class SentimentFeatureRow:
    """Row of computed sentiment features for a single trading session."""

    sent_mean: Optional[float]
    sent_weighted: Optional[float]
    sent_count: float
    sent_ema_5: Optional[float]
    sent_z_20: Optional[float]

    @classmethod
    def empty(cls) -> "SentimentFeatureRow":
        return cls(None, None, 0.0, None, None)

    def copy(self) -> "SentimentFeatureRow":
        return SentimentFeatureRow(**asdict(self))


def _ensure_datetime(value: datetime | date) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.combine(value, time(0, 0))


def _normalize_calendar(trading_calendar: Sequence[datetime | date]) -> List[date]:
    days = sorted({(_ensure_datetime(day)).date() for day in trading_calendar})
    return days


def _assign_session(
    published: datetime,
    sessions: Sequence[date],
    market_close: time,
) -> Optional[date]:
    published_dt = _ensure_datetime(published)
    session_date = published_dt.date()
    if published_dt.time() >= market_close:
        session_date += timedelta(days=1)
    for session in sessions:
        if session >= session_date:
            return session
    return None


def _score_article(article: NewsArticle, scorer: SentimentScorer) -> Optional[float]:
    text = article.combined_text()
    if not text:
        return None
    try:
        return float(scorer.polarity_scores(text).get("compound", 0.0))
    except Exception:
        return None


def _compute_ema(values: Sequence[Optional[float]], span: int) -> List[Optional[float]]:
    alpha = 2.0 / (span + 1.0)
    ema: Optional[float] = None
    result: List[Optional[float]] = []
    for value in values:
        if value is None:
            result.append(ema)
            continue
        ema = value if ema is None else alpha * value + (1 - alpha) * ema
        result.append(ema)
    return result


def _compute_rolling_z(
    values: Sequence[Optional[float]],
    window: int,
    min_periods: int,
) -> List[Optional[float]]:
    window_values: List[float] = []
    result: List[Optional[float]] = []
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


def build_daily_sentiment_features(
    articles: Iterable[NewsArticle],
    trading_calendar: Sequence[datetime | date],
    *,
    market_close: time = time(16, 0),
    lag_bars: int = 1,
    scorer: Optional[SentimentScorer] = None,
    decay_half_life: float = 1.0,
) -> "OrderedDict[date, SentimentFeatureRow]":
    """Aggregate raw articles into lagged daily sentiment features."""

    sessions = _normalize_calendar(trading_calendar)
    if not sessions:
        return OrderedDict()

    scorer = scorer or get_default_sentiment_scorer()
    articles_by_session: MutableMapping[date, List[tuple[float, datetime]]] = defaultdict(list)

    for article in articles:
        session = _assign_session(article.published, sessions, market_close)
        if session is None:
            continue
        score = _score_article(article, scorer)
        if score is None:
            continue
        articles_by_session[session].append((score, _ensure_datetime(article.published)))

    raw_rows: List[SentimentFeatureRow] = []
    raw_means: List[Optional[float]] = []

    for session in sessions:
        entries = articles_by_session.get(session, [])
        if not entries:
            row = SentimentFeatureRow.empty()
            raw_rows.append(row)
            raw_means.append(None)
            continue
        scores = [score for score, _ in entries]
        mean_score = sum(scores) / len(scores)
        session_dt = datetime.combine(session, time(0, 0))
        half_life = max(decay_half_life, 1e-9)
        weights = []
        weighted_scores = []
        for score, published_dt in entries:
            age_days = (session_dt - published_dt).total_seconds() / 86400.0
            weight = math.exp(-math.log(2.0) * age_days / half_life)
            weights.append(weight)
            weighted_scores.append(score * weight)
        weight_sum = sum(weights)
        weighted = sum(weighted_scores) / weight_sum if weight_sum > 0 else None
        row = SentimentFeatureRow(
            sent_mean=mean_score,
            sent_weighted=weighted,
            sent_count=float(len(entries)),
            sent_ema_5=None,
            sent_z_20=None,
        )
        raw_rows.append(row)
        raw_means.append(mean_score)

    ema_values = _compute_ema(raw_means, span=5)
    z_values = _compute_rolling_z(raw_means, window=20, min_periods=5)

    for row, ema, z in zip(raw_rows, ema_values, z_values):
        row.sent_ema_5 = ema
        row.sent_z_20 = z

    shifted_rows: List[SentimentFeatureRow] = []
    for idx, _session in enumerate(sessions):
        if lag_bars <= 0:
            shifted_rows.append(raw_rows[idx].copy())
            continue
        source_idx = idx - lag_bars
        if source_idx < 0:
            shifted_rows.append(SentimentFeatureRow.empty())
        else:
            shifted_rows.append(raw_rows[source_idx].copy())

    return OrderedDict(zip(sessions, shifted_rows))


def prepare_sentiment_features(
    ticker: str,
    price_index: Sequence[datetime | date],
    news_loader: Iterable[NewsArticle] | Callable[[str, datetime, datetime], Iterable[NewsArticle]],
    *,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    market_close: time = time(16, 0),
    lag_bars: int = 1,
    scorer: Optional[SentimentScorer] = None,
    decay_half_life: float = 1.0,
) -> "OrderedDict[date, SentimentFeatureRow]":
    """Helper that loads articles and builds daily sentiment features."""

    sessions = _normalize_calendar(price_index)
    if not sessions:
        return OrderedDict()

    start_dt = _ensure_datetime(start or sessions[0])
    end_dt = _ensure_datetime(end or sessions[-1])

    if callable(news_loader):
        loader_iter = news_loader(ticker, start_dt, end_dt)
    else:
        loader_iter = news_loader

    filtered_articles = [
        article
        for article in loader_iter
        if _ensure_datetime(article.published) <= end_dt
    ]

    return build_daily_sentiment_features(
        filtered_articles,
        sessions,
        market_close=market_close,
        lag_bars=lag_bars,
        scorer=scorer,
        decay_half_life=decay_half_life,
    )


__all__ = [
    "FEATURE_COLUMNS",
    "SentimentFeatureRow",
    "SentimentScorer",
    "SimpleSentimentIntensityAnalyzer",
    "build_daily_sentiment_features",
    "get_default_sentiment_scorer",
    "prepare_sentiment_features",
]
