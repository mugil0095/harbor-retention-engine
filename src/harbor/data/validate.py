"""Schema and data-quality validation.

Every raw table must pass `validate_table` before it is allowed into the
feature engineering pipeline. This is deliberately simple (no external
validation-framework dependency) but checks the things that actually break
downstream code: missing columns, wrong dtypes, duplicate primary keys,
dangling foreign keys, and out-of-range values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from harbor.data.schema import TableSchema

_DTYPE_KIND_NAME = {"i": "integer", "f": "float", "O": "object/string", "b": "boolean", "M": "datetime"}


@dataclass
class ValidationResult:
    table_name: str
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def add(self, msg: str) -> None:
        self.errors.append(msg)


def validate_table(
    df: pd.DataFrame,
    schema: TableSchema,
    reference_tables: dict[str, pd.DataFrame] | None = None,
) -> ValidationResult:
    result = ValidationResult(table_name=schema.name)
    reference_tables = reference_tables or {}

    missing_cols = [c for c in schema.required_columns if c not in df.columns]
    if missing_cols:
        result.add(f"missing required columns: {missing_cols}")
        return result  # further checks would be meaningless

    for col, expected_kind in schema.required_columns.items():
        actual_kind = df[col].dtype.kind
        # treat pandas' small-int/float variations as compatible
        compatible = {
            "i": {"i", "u"},
            "f": {"f", "i", "u"},
            "b": {"b"},
            "O": {"O"},
            "M": {"M"},
        }[expected_kind]
        if actual_kind not in compatible:
            result.add(
                f"column '{col}' has dtype kind '{actual_kind}' "
                f"({df[col].dtype}), expected {_DTYPE_KIND_NAME[expected_kind]}"
            )

    if schema.primary_key:
        dupe_count = int(df.duplicated(subset=list(schema.primary_key)).sum())
        if dupe_count > 0:
            result.add(
                f"primary key {schema.primary_key} has {dupe_count} duplicate row(s)"
            )

    for col in schema.required_columns:
        if col not in df.columns:
            continue
        null_frac = float(df[col].isna().mean())
        if null_frac > 0.05 and col not in schema.primary_key:
            result.add(f"column '{col}' has {null_frac:.1%} missing values (> 5% threshold)")

    for col, ref_table_name in schema.foreign_keys.items():
        if ref_table_name not in reference_tables:
            continue
        ref_ids = set(reference_tables[ref_table_name]["account_id"].unique())
        dangling = ~df[col].isin(ref_ids)
        n_dangling = int(dangling.sum())
        if n_dangling > 0:
            result.add(
                f"column '{col}' has {n_dangling} value(s) not present in '{ref_table_name}'"
            )

    if "mrr" in df.columns:
        n_negative = int((df["mrr"] < 0).sum())
        if n_negative > 0:
            result.add(f"'mrr' has {n_negative} negative value(s)")

    if "month_index" in df.columns and (df["month_index"] < 0).any():
        result.add("'month_index' contains negative values")

    return result


def assert_valid(result: ValidationResult) -> None:
    if not result.is_valid:
        raise ValueError(
            f"Validation failed for table '{result.table_name}':\n"
            + "\n".join(f"  - {e}" for e in result.errors)
        )
