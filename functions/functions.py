import numpy as np

def apply_signal_mask(weights_df, signal_df):
    """
    Zero out weights for symbols not in the signal.

    Args:
        weights_df: DataFrame of raw weights (dates x tickers).
        signal_df: Binary DataFrame (dates x tickers), 1 = investable.

    Returns:
        masked_weights_df: Filtered weights with non-signal symbols zeroed.
    """
    # Align signal_df index to weights_df
    signal_df = signal_df.reindex_like(weights_df).fillna(0).astype(int)

    # Element-wise multiplication to apply the mask
    masked_weights_df = weights_df * signal_df

    # Re-normalize weights so each row sums to 1 (or 0 if all-zero)
    row_sums = masked_weights_df.sum(axis=1)
    normalized_weights = masked_weights_df.div(row_sums.replace(0, np.nan), axis=0).fillna(0)

    return normalized_weights

