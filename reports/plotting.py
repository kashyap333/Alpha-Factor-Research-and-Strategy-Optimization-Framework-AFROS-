import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple
import re


def _max_drawdown_info(wealth: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """
    Compute maximum drawdown and the period (start -> trough -> recovery end).
    Returns: (max_dd, start_date, trough_date, end_date)
    max_dd is negative (e.g. -0.25 for -25%).
    """
    if wealth.empty:
        return 0.0, None, None, None

    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0  # negative or zero
    trough_idx = drawdown.idxmin()
    max_dd = drawdown.min()

    # start = last time before trough where wealth == running_max
    pre_trough = wealth.loc[:trough_idx]
    try:
        start_idx = pre_trough[pre_trough == pre_trough.cummax()].index[-1]
    except IndexError:
        start_idx = wealth.index[0]

    # recovery end = first index after trough where wealth >= previous running max
    post_trough = wealth.loc[trough_idx:]
    recovery_idx = None
    prev_peak_value = wealth.loc[start_idx]
    for idx, val in post_trough.items():
        if val >= prev_peak_value:
            recovery_idx = idx
            break

    return float(max_dd), start_idx, trough_idx, recovery_idx


def plot_performance(
    portfolio_returns: pd.Series,
    save_path: Optional[Path | str] = None,
    show: bool = True,
    dpi: int = 300,
) -> Optional[Path]:
    """
    Plot cumulative returns and drawdown, highlight the maximum drawdown period.
    If save_path is provided, save the figure and return the Path; otherwise optionally show it.

    Args:
        portfolio_returns (pd.Series): Daily portfolio returns indexed by date.
        save_path (Path|str|None): If provided, save image to this path.
        show (bool): Whether to call plt.show() (ignored if save_path provided; pass show=True to display too).
        dpi (int): DPI when saving image.

    Returns:
        Path or None: Saved path if saved, else None.
    """
    # Input checks
    if not isinstance(portfolio_returns, pd.Series):
        raise ValueError("portfolio_returns must be a pandas Series indexed by date.")

    # Ensure sorted index and drop NaNs
    portfolio_returns = portfolio_returns.sort_index().dropna()
    if portfolio_returns.empty:
        raise ValueError("portfolio_returns is empty after dropping NaNs.")

    # Compute wealth/time series
    wealth = (1.0 + portfolio_returns).cumprod()
    cumulative_returns = wealth - 1.0

    # Compute drawdown series
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0  # <= 0

    # Compute max drawdown info
    max_dd, dd_start, dd_trough, dd_recovery = _max_drawdown_info(wealth)

    # --- Plotting ---
    fig, (ax_ret, ax_dd) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    # Cumulative returns (top)
    ax_ret.plot(cumulative_returns.index, cumulative_returns.values, linewidth=2, color="#2b8cbe")
    ax_ret.set_ylabel("Cumulative Return")
    ax_ret.set_title("Strategy Performance")
    ax_ret.grid(alpha=0.4, linestyle="--")

    # Highlight max drawdown period on the cumulative returns chart
    if dd_start is not None and dd_trough is not None:
        # Shade drawdown period
        ax_ret.axvspan(dd_start, dd_recovery if dd_recovery is not None else dd_trough, color="#fde725", alpha=0.15)
        # Mark peak and trough
        ax_ret.scatter([dd_start], [cumulative_returns.loc[dd_start]], color="#e6550d", zorder=5, label="Drawdown start")
        ax_ret.scatter([dd_trough], [cumulative_returns.loc[dd_trough]], color="#d73027", zorder=6, label="Trough")
        if dd_recovery is not None:
            ax_ret.scatter([dd_recovery], [cumulative_returns.loc[dd_recovery]], color="#2ca25f", zorder=5, label="Recovery")

    ax_ret.legend(loc="upper left", fontsize=9)

    # Drawdown (bottom)
    ax_dd.fill_between(drawdown.index, drawdown.values, 0, color="#d73027", alpha=0.35)
    ax_dd.plot(drawdown.index, drawdown.values, color="#a50f15", linewidth=1)
    ax_dd.set_ylabel("Drawdown")
    ax_dd.set_xlabel("Date")
    ax_dd.grid(alpha=0.4, linestyle="--")

    # Annotate max drawdown value & period on the drawdown plot
    if dd_trough is not None:
        ann_x = dd_trough
        ann_y = drawdown.loc[dd_trough]
        # Format % and period text
        dd_pct = f"{(max_dd * 100):.2f}%"
        if dd_recovery is None:
            period_text = f"{dd_start.date()} → {dd_trough.date()}"
        else:
            period_text = f"{dd_start.date()} → {dd_trough.date()} → {dd_recovery.date()}"
        annotation = f"Max Drawdown: {dd_pct}\nPeriod: {period_text}"
        ax_dd.annotate(
            annotation,
            xy=(ann_x, ann_y),
            xytext=(20, -40),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.9),
            arrowprops=dict(arrowstyle="->", connectionstyle="angle3", color="#555555"),
            fontsize=9,
        )

    fig.tight_layout()

    saved_path = None
    if save_path:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        saved_path = out_path.resolve()
        plt.close(fig)
    else:
        if show:
            plt.show()
        else:
            plt.close(fig)

    return saved_path


def update_readme_with_image(readme_path: Path, image_rel_path: str, section_header: str = "## Strategy Performance"):
    """
    Insert or update a README section referencing the performance image.
    If the section_header exists, replace its content with the new image link.
    Otherwise append the section and image at the end of README.

    Args:
        readme_path (Path|str): Path to README.md
        image_rel_path (str): Relative path (from README) to the image file to embed, e.g. "assets/perf_2025-01-01.png"
        section_header (str): Markdown header to use for the section.
    """
    readme_path = Path(readme_path)
    if not readme_path.exists():
        # Create README if missing
        readme_text = ""
    else:
        readme_text = readme_path.read_text(encoding='utf-8')
 
    # Markdown image line
    md_image = f"![Strategy Performance]({image_rel_path})\n"

    # Pattern to find the section and everything until the next top-level header (## or #) or EOF
    pattern = rf"({re.escape(section_header)}\s*\n)(.*?)(?=(\n#|\n##|\Z))"
    replacement = rf"{section_header}\n\n{md_image}\n"

    if re.search(pattern, readme_text, flags=re.DOTALL):
        new_readme = re.sub(pattern, replacement, readme_text, flags=re.DOTALL)
    else:
        # Append the section
        if not readme_text.endswith("\n"):
            readme_text += "\n"
        new_readme = readme_text + "\n" + replacement

    readme_path.write_text(new_readme, encoding='utf-8')
    return readme_path.resolve()