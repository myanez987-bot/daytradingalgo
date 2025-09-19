"""Contracts and data structures for sourcing historical news."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterable, Mapping, Optional


@dataclass(frozen=True)
class NewsArticle:
    """Container for a single news article."""

    ticker: str
    published: datetime
    headline: str = ""
    content: str = ""
    source: Optional[str] = None
    metadata: Optional[Mapping[str, object]] = None

    def combined_text(self) -> str:
        parts = [self.headline.strip(), self.content.strip()]
        return " ".join(part for part in parts if part)


NewsLoader = Callable[[str, datetime, datetime], Iterable[NewsArticle]]

__all__ = ["NewsArticle", "NewsLoader"]
