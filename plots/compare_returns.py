"""Scatter plot comparing buy-and-hold returns to strategy returns."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

try:  # pragma: no cover - plotting backend is optional in tests
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - fallback for headless envs
    plt = None  # type: ignore


def _align_returns(
    stock_returns: Sequence[float],
    strategy_returns: Sequence[float],
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    length = min(len(stock_returns), len(strategy_returns))
    return tuple(stock_returns[:length]), tuple(strategy_returns[:length])


def plot_strategy_vs_buy_and_hold(
    stock_returns: Sequence[float],
    strategy_returns: Sequence[float],
    *,
    title: Optional[str] = None,
    save_path: Optional[Path | str] = None,
    show: bool = False,
) -> Optional["plt.Axes"]:
    """Plot strategy returns against the benchmark buy-and-hold returns."""

    if plt is None:  # pragma: no cover - graceful no-op when matplotlib missing
        return None

    stock, strat = _align_returns(stock_returns, strategy_returns)
    if not stock:
        return None

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(stock, strat, alpha=0.6, edgecolor="none")

    bound = max(max(abs(value) for value in stock), max(abs(value) for value in strat))
    ax.plot([-bound, bound], [-bound, bound], linestyle="--", color="black", linewidth=1)

    ax.set_xlabel("Buy & Hold Return")
    ax.set_ylabel("Strategy Return")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return ax


__all__ = ["plot_strategy_vs_buy_and_hold"]
