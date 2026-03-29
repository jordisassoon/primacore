import pandas as pd
from collections.abc import Callable, Sequence


Transform = Callable[[pd.DataFrame], pd.DataFrame]


def load_csv(filepath: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(filepath)


def apply_transforms(df: pd.DataFrame, transforms: Sequence[Transform]) -> pd.DataFrame:
    """Apply a list of transform functions to a DataFrame."""
    for transform in transforms:
        df = transform(df)
    return df


def load_csv_with_transforms(
    filepath: str,
    transforms: Sequence[Transform] | None = None,
) -> pd.DataFrame:
    """Load a CSV file and apply custom transforms."""
    df = load_csv(filepath)
    if transforms:
        df = apply_transforms(df, transforms)
    return df


# Example transform functions
def drop_rows_with_any_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with null values."""
    return df.dropna(how="any")


def drop_columns_with_all_zero(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that contain only zero values."""
    return df.loc[:, (df != 0).any(axis=0)]


def drop_rows_with_all_zero(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that contain only zero values."""
    return df[(df != 0).any(axis=1)]


def drop_rows_with_all_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that contain only NaN values."""
    return df.dropna(how="all")


def l1_normalize_rows(df: pd.DataFrame) -> pd.DataFrame:
    """L1 normalize rows of a DataFrame."""
    numeric = df.select_dtypes(include="number")
    normalized = numeric.div(numeric.abs().sum(axis=1), axis=0)
    return df.assign(**normalized)
