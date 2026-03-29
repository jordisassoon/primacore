import pathlib

import pandas as pd
import numpy as np

from primacore.dataloader import (
    load_csv,
    apply_transforms,
    load_csv_with_transforms,
    drop_rows_with_any_nan,
    drop_columns_with_all_zero,
    drop_rows_with_all_zero,
    drop_rows_with_all_nan,
    l1_normalize_rows,
)


def test_load_csv(tmp_path: pathlib.Path) -> None:
    """Test loading a CSV file into a DataFrame."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("a,b,c\n1,2,3\n4,5,6")
    df = load_csv(str(csv_file))
    assert df.shape == (2, 3)
    assert list(df.columns) == ["a", "b", "c"]


def test_apply_transforms() -> None:
    """Test applying a list of transform functions to a DataFrame."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [0, 0, 0]})
    transforms = [drop_columns_with_all_zero]
    result = apply_transforms(df, transforms)
    assert list(result.columns) == ["a"]


def test_load_csv_with_transforms(tmp_path: pathlib.Path) -> None:
    """Test loading a CSV file and applying custom transforms."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("a,b\n1,0\n2,0\n3,0")
    transforms = [drop_columns_with_all_zero]
    df = load_csv_with_transforms(str(csv_file), transforms)
    assert list(df.columns) == ["a"]


def test_drop_rows_with_any_nan() -> None:
    """Test dropping rows with any NaN values."""
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0]})
    result = drop_rows_with_any_nan(df)
    assert result.equals(pd.DataFrame({"a": [1.0, 3.0], "b": [4.0, 6.0]}, index=[0, 2]))


def test_drop_columns_with_all_zero() -> None:
    """Test dropping columns that contain only zero values."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0.0, 0.0, 0.0]})
    result = drop_columns_with_all_zero(df)
    assert result.equals(pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=[0, 1, 2]))


def test_drop_rows_with_all_zero() -> None:
    """Test dropping rows that contain only zero values."""
    df = pd.DataFrame({"a": [0.0, 1.0, 0.0], "b": [0.0, 2.0, 0.0]})
    result = drop_rows_with_all_zero(df)
    assert result.equals(pd.DataFrame({"a": [1.0], "b": [2.0]}, index=[1]))


def test_drop_rows_with_all_nan() -> None:
    """Test dropping rows that contain only NaN values."""
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, np.nan, 6.0]})
    result = drop_rows_with_all_nan(df)
    assert result.equals(pd.DataFrame({"a": [1.0, 3.0], "b": [4.0, 6.0]}, index=[0, 2]))


def test_l1_normalize_rows() -> None:
    """Test L1 normalization of DataFrame rows."""
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 3.0]})
    result = l1_normalize_rows(df)
    assert result.equals(
        pd.DataFrame({"a": [1 / 3, 2 / 5], "b": [2 / 3, 3 / 5]}, index=[0, 1])
    )
