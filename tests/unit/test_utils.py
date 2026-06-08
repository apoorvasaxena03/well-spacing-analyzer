"""
tests/unit/test_utils.py

Unit tests for src/utils/utils.py
All tests are pure in-memory — no file I/O, no network, no database.
"""

import math
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.utils.utils import (
    add_bench_columns_to_spacing,
    clean_column_names,
    drop_uwi_duplicates_keep_max_last_prod,
    read_csv_with_mapper,
    reorder_columns,
    standardize_column_names,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# reorder_columns
# ---------------------------------------------------------------------------


class TestReorderColumns:
    def test_moves_columns_after_reference(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})
        result = reorder_columns(df, columns_to_move=["c", "d"], reference_column="a")
        # c, d should appear right after a
        cols = result.columns.tolist()
        assert cols.index("c") == cols.index("a") + 1
        assert cols.index("d") == cols.index("a") + 2

    def test_all_original_columns_preserved(self):
        df = pd.DataFrame({"x": [1], "y": [2], "z": [3]})
        result = reorder_columns(df, columns_to_move=["z"], reference_column="x")
        assert set(result.columns) == {"x", "y", "z"}

    def test_raises_on_missing_column(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="must exist"):
            reorder_columns(df, columns_to_move=["c"], reference_column="a")

    def test_raises_on_missing_reference(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="must exist"):
            reorder_columns(df, columns_to_move=["a"], reference_column="x")

    def test_data_unchanged(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        result = reorder_columns(df, columns_to_move=["c"], reference_column="a")
        assert result["a"].tolist() == [1, 2]
        assert result["c"].tolist() == [5, 6]


# ---------------------------------------------------------------------------
# clean_column_names
# ---------------------------------------------------------------------------


class TestCleanColumnNames:
    def test_lowercase_and_underscores(self):
        df = pd.DataFrame({"First Name": [1], "LAST NAME": [2]})
        result = clean_column_names(df)
        assert "first_name" in result.columns
        assert "last_name" in result.columns

    def test_special_characters_removed(self):
        df = pd.DataFrame({"Oil (BBL/D)": [1], "Gas %": [2]})
        result = clean_column_names(df)
        # special chars stripped; spaces → underscore; lowercase
        for col in result.columns:
            assert "(" not in col and ")" not in col and "%" not in col

    def test_returns_dataframe(self):
        df = pd.DataFrame({"A": [1]})
        assert isinstance(clean_column_names(df), pd.DataFrame)


# ---------------------------------------------------------------------------
# standardize_column_names
# ---------------------------------------------------------------------------


class TestStandardizeColumnNames:
    @pytest.mark.parametrize("input_col,expected", [
        ("WellName",        "well_name"),
        ("FirstProdDate",   "first_prod_date"),
        ("ENVOperator",     "env_operator"),
        ("API_UWI_14",      "api_uwi_14"),
        ("SurfaceLatitude", "surface_latitude"),
        ("HoleDirection",   "hole_direction"),
        ("GALPerFT",        "gal_per_ft"),
    ])
    def test_camel_and_pascal_case(self, input_col, expected):
        df = pd.DataFrame({input_col: [1]})
        result = standardize_column_names(df)
        assert expected in result.columns

    def test_no_duplicate_underscores(self):
        df = pd.DataFrame({"API__Well": [1]})
        result = standardize_column_names(df)
        for col in result.columns:
            assert "__" not in col

    def test_returns_copy_not_inplace(self):
        df = pd.DataFrame({"WellName": [1]})
        result = standardize_column_names(df)
        assert "WellName" in df.columns  # original unchanged
        assert "WellName" not in result.columns


# ---------------------------------------------------------------------------
# read_csv_with_mapper
# ---------------------------------------------------------------------------


class TestReadCsvWithMapper:
    def test_renames_specified_columns(self, tmp_path):
        csv = tmp_path / "test.csv"
        csv.write_text("API14,WellName,Lat\n42001,Alice,31.5\n")
        df = read_csv_with_mapper(csv, col_map={"API14": "uwi", "WellName": "well_name"})
        assert "uwi" in df.columns
        assert "well_name" in df.columns
        assert "Lat" in df.columns  # unmapped column kept as-is

    def test_dtype_cast_after_rename(self, tmp_path):
        csv = tmp_path / "test.csv"
        csv.write_text("ID,Value\n1,3.14\n2,2.71\n")
        df = read_csv_with_mapper(
            csv,
            col_map={"ID": "id"},
            dtype_map={"id": "string", "Value": "float64"},
        )
        assert df["id"].dtype == "string"
        assert df["Value"].dtype == "float64"

    def test_no_mapper_behaves_like_read_csv(self, tmp_path):
        csv = tmp_path / "test.csv"
        csv.write_text("a,b\n1,2\n")
        df = read_csv_with_mapper(csv)
        assert list(df.columns) == ["a", "b"]

    def test_missing_dtype_col_ignored_gracefully(self, tmp_path):
        csv = tmp_path / "test.csv"
        csv.write_text("a,b\n1,2\n")
        # dtype_map references column 'z' which doesn't exist — should not raise
        df = read_csv_with_mapper(csv, dtype_map={"z": "float64"})
        assert list(df.columns) == ["a", "b"]


# ---------------------------------------------------------------------------
# drop_uwi_duplicates_keep_max_last_prod
# ---------------------------------------------------------------------------


class TestDropUwiDuplicates:
    def test_keeps_max_last_prod_date(self):
        df = pd.DataFrame({
            "uwi":           ["A", "A", "B"],
            "last_prod_date": [pd.Timestamp("2021-01-01"),
                               pd.Timestamp("2022-06-01"),
                               pd.Timestamp("2020-01-01")],
            "value": [1, 2, 3],
        })
        result = drop_uwi_duplicates_keep_max_last_prod(df)
        assert len(result) == 2
        a_row = result[result["uwi"] == "A"]
        assert a_row["last_prod_date"].iloc[0] == pd.Timestamp("2022-06-01")
        assert a_row["value"].iloc[0] == 2

    def test_all_nat_keeps_first_occurrence(self):
        df = pd.DataFrame({
            "uwi":           ["C", "C"],
            "last_prod_date": [pd.NaT, pd.NaT],
            "value":         [10, 20],
        })
        result = drop_uwi_duplicates_keep_max_last_prod(df)
        assert len(result) == 1
        assert result["value"].iloc[0] == 10  # first occurrence

    def test_no_duplicates_unchanged(self):
        df = pd.DataFrame({
            "uwi":           ["X", "Y"],
            "last_prod_date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01")],
        })
        result = drop_uwi_duplicates_keep_max_last_prod(df)
        assert len(result) == 2

    def test_raises_on_missing_uwi_col(self):
        df = pd.DataFrame({"well": ["A"], "last_prod_date": [pd.NaT]})
        with pytest.raises(KeyError):
            drop_uwi_duplicates_keep_max_last_prod(df, uwi_col="uwi")

    def test_raises_on_missing_date_col(self):
        df = pd.DataFrame({"uwi": ["A"], "other": [1]})
        with pytest.raises(KeyError):
            drop_uwi_duplicates_keep_max_last_prod(df, last_prod_col="last_prod_date")

    def test_mixed_nat_and_dates(self):
        """UWI with some NaT rows and some real dates — keep the max date row."""
        df = pd.DataFrame({
            "uwi":           ["D", "D", "D"],
            "last_prod_date": [pd.NaT,
                               pd.Timestamp("2019-06-01"),
                               pd.Timestamp("2021-12-31")],
            "value":         [1, 2, 3],
        })
        result = drop_uwi_duplicates_keep_max_last_prod(df)
        assert len(result) == 1
        assert result["value"].iloc[0] == 3


# ---------------------------------------------------------------------------
# add_bench_columns_to_spacing
# ---------------------------------------------------------------------------


class TestAddBenchColumnsToSpacing:
    def test_adds_bench_i_and_bench_k(self):
        # add_bench_columns_to_spacing also reorders 'direction_to_k_from_i_axis',
        # 'overlap_pct_i/k', etc. relative to '3D_dist' — include those columns.
        spacing = pd.DataFrame({
            "well_i":                    ["A", "B"],
            "well_k":                    ["B", "C"],
            "horizontal_dist":           [700.0, 800.0],
            "3D_dist":                   [701.0, 802.0],
            "direction_to_k_from_i_axis": ["N", "N"],
            "overlap_pct_i":             [0.8, 0.9],
            "overlap_pct_k":             [0.8, 0.9],
            "overlap_len_common_ft":     [5000.0, 5200.0],
            "LL_i":                      [6000.0, 6000.0],
            "LL_k":                      [6000.0, 6000.0],
        })
        header = pd.DataFrame({
            "uwi":   ["A", "B", "C"],
            "bench": ["WOLFCAMP A", "WOLFCAMP A", "WOLFCAMP B"],
        })
        result = add_bench_columns_to_spacing(spacing, header, reorder_columns)
        assert "bench_i" in result.columns
        assert "bench_k" in result.columns
        assert result["bench_i"].tolist() == ["WOLFCAMP A", "WOLFCAMP A"]
        assert result["bench_k"].tolist() == ["WOLFCAMP A", "WOLFCAMP B"]
