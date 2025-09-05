# cluster_value_profiles.py
# Build VALUE-LEVEL profiles per cluster for all categorical columns
# and for a forced list of important categoricals (emotion/zodiac/job/etc).

import os
import sys
import glob
import time
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# ---------------- Config ----------------
DEFAULT_GLOB = "clusters_personas*.csv"       # fallback search pattern
OUT_PREFIX   = "cluster_profiles"             # output file prefix

# Force these columns to be treated as categorical WHEN PRESENT
FORCED_CATEGORICALS = [
    "top_emotion", "emotion_sentiment", "reddit_sentiment",
    "horoscope", "job",
    "emo1_label", "emo2_label", "emo3_label", "emo4_label", "emo5_label",
    "color_name"
]
CLUSTER_COL = "cluster"
# ----------------------------------------


def find_input_csv() -> str:
    env_path = os.environ.get("CLUSTER_CSV")
    if env_path and os.path.isfile(env_path):
        print(f"📦 Using CLUSTER_CSV from env: {env_path}")
        return env_path

    print(f"🔎 CLUSTER_CSV not set or file missing. Searching newest '{DEFAULT_GLOB}'...")
    candidates = [p for p in glob.glob(DEFAULT_GLOB) if os.path.isfile(p)]
    if not candidates:
        raise FileNotFoundError(
            f"No input found. Set CLUSTER_CSV or place a file matching {DEFAULT_GLOB}."
        )
    newest = max(candidates, key=lambda p: os.path.getmtime(p))
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(newest)))
    print(f"   • Found {len(candidates)} file(s). Picking newest: {newest} (modified {ts})")
    return newest


def cramers_v(contingency: pd.DataFrame) -> float:
    chi2, _, _, _ = chi2_contingency(contingency)
    n = contingency.to_numpy().sum()
    if n == 0:
        return np.nan
    r, k = contingency.shape
    phi2 = chi2 / n
    # bias-corrected
    phi2corr = max(0.0, phi2 - (k-1)*(r-1)/(n-1))
    rcorr = r - (r-1)**2/(n-1)
    kcorr = k - (k-1)**2/(n-1)
    denom = min((kcorr-1), (rcorr-1))
    if denom <= 0:
        return np.nan
    return np.sqrt(phi2corr / denom)


def prepare_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return list of categorical columns to analyze, after forcing key columns to category dtype."""
    present_forced = [c for c in FORCED_CATEGORICALS if c in df.columns]
    # coerce forced columns to string category (robust to NaNs)
    for c in present_forced:
        df[c] = df[c].astype("string").fillna(pd.NA).astype("category")

    # add any other object/category columns (except the cluster column)
    auto_obj = [
        c for c in df.columns
        if c != CLUSTER_COL and (df[c].dtype == "object" or str(df[c].dtype).startswith("category"))
    ]
    # union while preserving order: forced first, then the extras not already in list
    cats = present_forced + [c for c in auto_obj if c not in present_forced]
    print(f"🧩 Categorical features selected ({len(cats)}): {', '.join(cats) if cats else '—'}")
    return cats


def categorical_profiles(df: pd.DataFrame, cat_cols: list[str]):
    long_rows, test_rows = [], []

    for c in cat_cols:
        # Build contingency & stats
        try:
            cont = pd.crosstab(df[CLUSTER_COL], df[c])
            chi2, pval, dof, _ = chi2_contingency(cont)
            v = cramers_v(cont)
            test_rows.append({"feature": c, "chi2": chi2, "dof": dof, "p_value": pval, "cramers_v": v})
            print(f"   • {c:<18} chi2={chi2:.2f}  p={pval:.3g}  V={v:.3f}")
        except Exception as e:
            print(f"   • {c:<18} (warn) chi-square failed: {e}")
            test_rows.append({"feature": c, "chi2": np.nan, "dof": np.nan, "p_value": np.nan, "cramers_v": np.nan})

        # Row-normalized % per cluster
        ct_pct = pd.crosstab(df[CLUSTER_COL], df[c], normalize="index") * 100.0
        ct_pct = ct_pct.fillna(0.0).round(2)
        for clus, row in ct_pct.iterrows():
            for val, pct in row.items():
                long_rows.append({"cluster": clus, "feature": c, "value": val, "pct": pct})

    prof_long = pd.DataFrame(long_rows).sort_values(["feature", "cluster", "pct"], ascending=[True, True, False])
    tests = pd.DataFrame(test_rows).sort_values("p_value")
    return prof_long, tests


def numeric_profiles(df: pd.DataFrame):
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != CLUSTER_COL]
    if not num_cols:
        return pd.DataFrame()
    print(f"🔢 Numeric features to summarize: {len(num_cols)}")
    return df.groupby(CLUSTER_COL)[num_cols].agg(["mean", "std", "median", "min", "max"])


def main():
    try:
        input_csv = find_input_csv()
        print(f"📥 Loading: {input_csv}")
        df = pd.read_csv(input_csv)
        print(f"   • Rows={len(df):,}  Cols={len(df.columns)}")

        if CLUSTER_COL not in df.columns:
            raise ValueError("Input CSV must contain a 'cluster' column.")
        print("\n📊 Cluster sizes:")
        print(df[CLUSTER_COL].value_counts(dropna=False).sort_index().to_string())

        # Prepare categorical list (forced + auto)
        print("\n🧪 Selecting categorical columns (forcing emotions/zodiac/job if present)…")
        cat_cols = prepare_categorical_columns(df)
        if not cat_cols:
            print("⚠️  No categorical columns found. Only numeric summary will be produced.")

        # Build categorical value profiles
        if cat_cols:
            print("\n📐 Computing categorical value profiles & chi-square/Cramér’s V…")
            prof_long, tests = categorical_profiles(df, cat_cols)
        else:
            prof_long = pd.DataFrame(columns=["cluster", "feature", "value", "pct"])
            tests = pd.DataFrame(columns=["feature", "chi2", "dof", "p_value", "cramers_v"])

        # Numeric summaries
        print("\n📈 Computing numeric summaries…")
        num_summary = numeric_profiles(df)

        # Save
        prof_path = f"{OUT_PREFIX}_categorical_profiles_long.csv"
        tests_path = f"{OUT_PREFIX}_categorical_tests.csv"
        num_path  = f"{OUT_PREFIX}_numeric_summary.csv"

        prof_long.to_csv(prof_path, index=False)
        tests.to_csv(tests_path, index=False)
        if not num_summary.empty:
            num_summary.to_csv(num_path)

        print("\n✅ Saved:")
        print(f"   • Categorical value profiles (long): {prof_path}")
        print(f"   • Chi-square & Cramér’s V per feature: {tests_path}")
        if not num_summary.empty:
            print(f"   • Numeric summaries by cluster: {num_path}")

        # Quick preview
        if not prof_long.empty:
            print("\n=== Preview: top rows of value profiles ===")
            print(prof_long.head(20).to_string(index=False))
        if not tests.empty:
            print("\n=== Strongest categorical associations (top 10 by Cramér’s V) ===")
            print(tests.sort_values("cramers_v", ascending=False).head(10).to_string(index=False))

    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
