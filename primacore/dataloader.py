import pandas as pd
from typing import Callable, List, Optional


def load_csv(filepath: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(filepath)


def apply_transforms(
    df: pd.DataFrame, transforms: List[Callable[[pd.DataFrame], pd.DataFrame]]
) -> pd.DataFrame:
    """Apply a list of transform functions to a DataFrame."""
    for transform in transforms:
        df = transform(df)
    return df


def load_csv_with_transforms(
    filepath: str,
    transforms: Optional[List[Callable[[pd.DataFrame], pd.DataFrame]]] = None,
) -> pd.DataFrame:
    """Load a CSV file and apply custom transforms."""
    df = load_csv(filepath)
    if transforms:
        df = apply_transforms(df, transforms)
    return df


# Example transform functions
def drop_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with null values."""
    return df.dropna()


def lowercase_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all column names to lowercase."""
    df.columns = df.columns.str.lower()
    return df


def filter_by_column(column: str, value) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Return a transform function that filters by column value."""

    def transform(df: pd.DataFrame) -> pd.DataFrame:
        return df[df[column] == value]

    return transform
