"""B: schema validation. C: missing/invalid data handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from harbor.data.schema import ACCOUNTS, MONTHLY_USAGE
from harbor.data.validate import assert_valid, validate_table


def test_valid_table_passes(synthetic_tables):
    result = validate_table(synthetic_tables["accounts"], ACCOUNTS)
    assert result.is_valid
    assert_valid(result)  # must not raise


def test_missing_required_column_fails():
    df = pd.DataFrame({"account_id": [1, 2, 3]})  # missing everything else
    result = validate_table(df, ACCOUNTS)
    assert not result.is_valid
    assert any("missing required columns" in e for e in result.errors)


def test_duplicate_primary_key_fails(synthetic_tables):
    df = pd.concat([synthetic_tables["accounts"], synthetic_tables["accounts"].iloc[[0]]], ignore_index=True)
    result = validate_table(df, ACCOUNTS)
    assert not result.is_valid
    assert any("duplicate" in e for e in result.errors)


def test_negative_mrr_fails(synthetic_tables):
    df = synthetic_tables["monthly_usage"].copy()
    df.loc[0, "mrr"] = -100.0
    result = validate_table(df, MONTHLY_USAGE, {"accounts": synthetic_tables["accounts"]})
    assert not result.is_valid
    assert any("negative" in e for e in result.errors)


def test_dangling_foreign_key_fails(synthetic_tables):
    df = synthetic_tables["monthly_usage"].copy()
    df.loc[0, "account_id"] = 999999  # does not exist in accounts
    result = validate_table(df, MONTHLY_USAGE, {"accounts": synthetic_tables["accounts"]})
    assert not result.is_valid
    assert any("not present in" in e for e in result.errors)


def test_excessive_missing_values_fails():
    df = pd.DataFrame(
        {
            "account_id": np.arange(100),
            "plan_tier": ["starter"] * 100,
            "industry": ["retail"] * 100,
            "employee_size_band": ["small"] * 100,
            "region": ["na"] * 100,
            "signup_month_offset": np.arange(100),
            "base_mrr": [np.nan] * 20 + [49.0] * 80,  # 20% missing
        }
    )
    result = validate_table(df, ACCOUNTS)
    assert not result.is_valid
    assert any("missing" in e for e in result.errors)


def test_assert_valid_raises_on_invalid_table():
    df = pd.DataFrame({"account_id": [1, 1]})
    result = validate_table(df, ACCOUNTS)
    with pytest.raises(ValueError):
        assert_valid(result)
