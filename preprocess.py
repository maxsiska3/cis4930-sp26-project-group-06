#!/usr/bin/env python3
"""
All shared settings come from config.py
  * drop the ID column
  * keep the binary label (attack vs normal) and the attack_cat family column
  * group rare protocol values into a single "other" category
  * treat the service "-" (missing) as its own category
  * one-hot encode proto / service / state
  * log(1 + x) transform the skewed columns listed in config.LOG_COLUMNS
  * check for duplicates and for overlap between the official train/test
    partitions (contamination check)

Features not standardized here, make_pipeline() in config.py
starts with a StandardScaler, fitted inside each training fold, so
scaling here would be redundant and would leak information across folds.
Resampling (SMOTE etc.) also happens only inside the pipeline

Everything learned from data is fitted on official TRAINING partition only then applied
to test partition

Outputs (written to --out-dir, default "processed"):
    train_processed.csv     features + label + attack_cat
    test_processed.csv      features + label + attack_cat
    preprocessor.joblib     fitted Preprocessor (reuse it in later scripts)
    preprocess_report.txt   data-quality summary

Usage:
    python preprocess.py
    python preprocess.py --keep-duplicates --drop-overlap

Library use (X can go straight into make_pipeline(...).fit):
    from preprocess import load_processed
    X_tr, yb_tr, yf_tr, X_te, yb_te, yf_te = load_processed()
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from config import (
    ATTACK_FAMILIES,
    BINARY_LABEL,
    CATEGORICAL_COLUMNS,
    FAMILY_LABEL,
    ID_COLUMN,
    LOG_COLUMNS,
    PROTO_MIN_COUNT,
    SEED,
    TEST_PATH,
    TRAIN_PATH,
)

EXPECTED_ROWS = {"train": 175_341, "test": 82_332}
FAMILIES = ["Normal"] + ATTACK_FAMILIES
FAMILY_FIXES = {"Backdoors": "Backdoor"}


# Loading and basic cleaning
def load_raw(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        sys.exit(f"ERROR: could not find {path}. Put the official CSVs in data/ (see config.py).")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    for col in (BINARY_LABEL, FAMILY_LABEL):
        if col not in df.columns:
            sys.exit(f"ERROR: expected column '{col}' not found in {path.name}.")

    df[FAMILY_LABEL] = df[FAMILY_LABEL].astype(str).str.strip().replace(FAMILY_FIXES)
    df[BINARY_LABEL] = df[BINARY_LABEL].astype(int)

    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype(str).str.strip().str.lower()
        
    # "-" means "no identified service". Keep as own category
    df["service"] = df["service"].replace({"-": "no_service"})

    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in (BINARY_LABEL, FAMILY_LABEL)]


def check_consistency(df: pd.DataFrame, name: str, report: list[str]) -> None:
    #Sanity checks - row count, label/family agreement, missing vals
    exp = EXPECTED_ROWS[name]
    if len(df) != exp:
        report.append(f"[warn] {name}: {len(df):,} rows (expected {exp:,}).")

    unknown = sorted(set(df[FAMILY_LABEL]) - set(FAMILIES))
    if unknown:
        report.append(f"[warn] {name}: unexpected family names {unknown}")

    # label should be 0 exactly when family Normal
    mismatch = int(((df[BINARY_LABEL] == 0) != (df[FAMILY_LABEL] == "Normal")).sum())
    if mismatch:
        report.append(f"[warn] {name}: {mismatch} rows where label and attack_cat disagree.")

    n_missing = int(df.isna().sum().sum())
    if n_missing:
        report.append(f"[warn] {name}: {n_missing} missing cells (numeric NaNs filled with train median).")

    missing_log = [c for c in LOG_COLUMNS if c not in df.columns]
    if missing_log:
        report.append(f"[warn] {name}: LOG_COLUMNS not found in data: {missing_log}")


# Duplicate and train/test overlap checks
def handle_duplicates(df: pd.DataFrame, name: str, drop: bool, report: list[str]) -> pd.DataFrame:
    n_dup = int(df.duplicated().sum())
    report.append(f"{name}: {n_dup:,} exact duplicate rows ({n_dup / len(df):.2%}).")
    if n_dup:
        by_family = df[df.duplicated(keep="first")][FAMILY_LABEL].value_counts()
        report.append("  duplicates by family: " + ", ".join(f"{k}={v}" for k, v in by_family.items()))
    if drop and n_dup:
        df = df.drop_duplicates().reset_index(drop=True)
        report.append(f"  -> dropped; {name} now has {len(df):,} rows.")
    return df


def handle_overlap(train: pd.DataFrame, test: pd.DataFrame, drop: bool,
                   report: list[str]) -> pd.DataFrame:
    # Find test rows that also appear in training
    feats = feature_columns(train)
    train_keys = set(pd.util.hash_pandas_object(train[feats], index=False).values)
    test_keys = pd.util.hash_pandas_object(test[feats], index=False)
    in_train = test_keys.isin(train_keys).to_numpy()
    n_overlap = int(in_train.sum())
    report.append(
        f"train/test overlap: {n_overlap:,} test rows ({n_overlap / len(test):.2%}) "
        "also appear in the training partition."
    )
    if n_overlap:
        by_family = test[in_train][FAMILY_LABEL].value_counts()
        report.append("  overlap by family: " + ", ".join(f"{k}={v}" for k, v in by_family.items()))
    if drop and n_overlap:
        test = test[~in_train].reset_index(drop=True)
        report.append(f"  -> removed from test; test now has {len(test):,} rows.")
    elif n_overlap:
        report.append("  -> kept (pass --drop-overlap to remove them from the test set).")
    return test


# Preprocessor (fit on train only)
class Preprocessor:
    # Rare-proto grouping -> log1p on config.LOG_COLUMNS -> one-hot. No scaling

    def __init__(self, min_proto_count: int = PROTO_MIN_COUNT):
        self.min_proto_count = min_proto_count

    def _group_proto(self, s: pd.Series) -> pd.Series:
        return s.where(s.isin(self.keep_protos_), "other")

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        self.num_cols_ = [c for c in feature_columns(df) if c not in CATEGORICAL_COLUMNS]
        self.log_cols_ = [c for c in LOG_COLUMNS if c in self.num_cols_]
        self.medians_ = df[self.num_cols_].median()

        counts = df["proto"].value_counts()
        self.keep_protos_ = set(counts[counts >= self.min_proto_count].index)

        cats = df[CATEGORICAL_COLUMNS].copy()
        cats["proto"] = self._group_proto(cats["proto"])
        self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cats)

        self.feature_names_ = list(self.num_cols_) + list(
            self.encoder_.get_feature_names_out(CATEGORICAL_COLUMNS)
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        num = df[self.num_cols_].fillna(self.medians_).copy()
        for c in self.log_cols_:
            num[c] = np.log1p(num[c].clip(lower=0))

        cats = df[CATEGORICAL_COLUMNS].copy()
        cats["proto"] = self._group_proto(cats["proto"])
        onehot = pd.DataFrame(
            self.encoder_.transform(cats),
            columns=self.encoder_.get_feature_names_out(CATEGORICAL_COLUMNS),
            index=df.index,
        )

        out = pd.concat([num, onehot], axis=1).astype("float32")
        out[BINARY_LABEL] = df[BINARY_LABEL].to_numpy()
        out[FAMILY_LABEL] = df[FAMILY_LABEL].to_numpy()
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


# Reporting
def class_distribution(df: pd.DataFrame, name: str) -> list[str]:
    counts = df[FAMILY_LABEL].value_counts()
    lines = [f"{name} class distribution ({len(df):,} rows):"]
    for fam in FAMILIES:
        n = int(counts.get(fam, 0))
        lines.append(f"  {fam:<15}{n:>9,}  {n / len(df):7.3%}")
    return lines


# Public helpers for the experiment scripts
def split_xy(df: pd.DataFrame):
    # Return (X, binary label, family label) from processed frame
    X = df.drop(columns=[BINARY_LABEL, FAMILY_LABEL])
    return X, df[BINARY_LABEL], df[FAMILY_LABEL]


def load_processed(out_dir: str | Path = "processed"):
    # Load saved processed CSVs as X / y_binary / y_family for train then test
    out_dir = Path(out_dir)
    train = pd.read_csv(out_dir / "train_processed.csv")
    test = pd.read_csv(out_dir / "test_processed.csv")
    return (*split_xy(train), *split_xy(test))


# Main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="processed", help="where to write outputs")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="do not drop exact duplicate rows (default: drop within each partition)")
    ap.add_argument("--drop-overlap", action="store_true",
                    help="remove test rows that also appear in the training partition")
    args = ap.parse_args()

    np.random.seed(SEED)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: list[str] = []

    train = load_raw(TRAIN_PATH)
    test = load_raw(TEST_PATH)
    report.append(f"Loaded train {train.shape}, test {test.shape} (after dropping '{ID_COLUMN}').")
    check_consistency(train, "train", report)
    check_consistency(test, "test", report)

    report.append("")
    report.append("--- Duplicate / contamination checks ---")
    train = handle_duplicates(train, "train", not args.keep_duplicates, report)
    test = handle_duplicates(test, "test", not args.keep_duplicates, report)
    test = handle_overlap(train, test, args.drop_overlap, report)

    report.append("")
    report += class_distribution(train, "train")
    report += class_distribution(test, "test")

    pre = Preprocessor()
    train_p = pre.fit_transform(train)
    test_p = pre.transform(test)

    report.append("")
    report.append("--- Transform summary ---")
    report.append(f"protocols kept (>= {pre.min_proto_count} train rows): {sorted(pre.keep_protos_)}; rest -> 'other'")
    report.append(f"log1p applied to {len(pre.log_cols_)} columns: {pre.log_cols_}")
    report.append(f"final feature count: {len(pre.feature_names_)}")
    report.append(f"service categories: {list(pre.encoder_.categories_[1])}")
    report.append(f"state categories: {list(pre.encoder_.categories_[2])}")
    report.append("features are NOT standardized; make_pipeline() scales inside each fold.")

    train_p.to_csv(out_dir / "train_processed.csv", index=False)
    test_p.to_csv(out_dir / "test_processed.csv", index=False)
    joblib.dump(pre, out_dir / "preprocessor.joblib")
    (out_dir / "preprocess_report.txt").write_text("\n".join(report) + "\n")

    print("\n".join(report))
    print(f"\nWrote outputs to {out_dir.resolve()}")


if __name__ == "__main__":
    main()