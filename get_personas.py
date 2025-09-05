# get_personas.py
# ------------------------------------------------------------
# Discovers personas by clustering emotion confidences, sentiment,
# simple text features, and categories. Loads
# ./clusters_personas_genz_only.csv by default (or pass --csv).
#
# Outputs:
#   persona_assignments.csv
#   persona_cards.json
#   persona_pca_scatter.png
# ------------------------------------------------------------

import argparse, json, re, warnings
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_CSV = "clusters_personas_genz_only.csv"

# ---------------------------- utils ----------------------------

def resolve_csv(path_arg: Optional[str]) -> Path:
    if path_arg:
        p = Path(path_arg).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"{p} not found")
    for base in (Path.cwd(), Path("/mnt/data")):
        p = (base / DEFAULT_CSV).resolve()
        if p.exists():
            return p
    raise FileNotFoundError(
        f"{DEFAULT_CSV} not found in {Path.cwd()} or /mnt/data. "
        f"Pass --csv /full/path/to/file.csv"
    )

def build_emotion_table(df: pd.DataFrame) -> pd.DataFrame:
    label_cols = [c for c in df.columns if re.fullmatch(r"emo\d+_label", c)]
    pairs = []
    for lc in label_cols:
        i = re.findall(r"\d+", lc)[0]
        cc = f"emo{i}_conf"
        if cc in df.columns:
            pairs.append((lc, cc))
    tall_rows = []
    for idx, row in df.iterrows():
        for lc, cc in pairs:
            lab = row.get(lc); conf = row.get(cc)
            if isinstance(lab, str) and pd.notna(conf):
                tall_rows.append((idx, lab.strip(), float(conf)))
    if not tall_rows:
        return pd.DataFrame(index=df.index)
    tall = pd.DataFrame(tall_rows, columns=["row", "emotion", "conf"])
    wide = tall.pivot_table(index="row", columns="emotion", values="conf", aggfunc="max").fillna(0.0)
    wide = wide.reindex(df.index, fill_value=0.0)
    wide.columns = [f"emotion_{c}" for c in wide.columns]
    return wide

def sentiment_to_score(x) -> float:
    if isinstance(x, str):
        s = x.strip().lower()
        if s == "positive": return 1.0
        if s == "negative": return -1.0
    return 0.0

def try_kmeans_k(X: np.ndarray, ks=range(2, 11), random_state: int = 42) -> Tuple[Optional[int], float]:
    best_k, best_score = None, -1.0
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        try:
            score = silhouette_score(X, labels)
        except Exception:
            continue
        if score > best_score:
            best_k, best_score = k, score
    return best_k, best_score

def top_terms_per_cluster(tfidf_matrix, terms: np.ndarray, labels: np.ndarray, topn: int = 12) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = {}
    df_terms = pd.DataFrame(tfidf_matrix.toarray(), columns=terms)
    for c in sorted(np.unique(labels)):
        idx = labels == c
        mean_tfidf = df_terms[idx].mean(axis=0)
        out[int(c)] = list(mean_tfidf.sort_values(ascending=False).head(topn).index)
    return out

def make_ohe():
    """Create OneHotEncoder that works on all sklearn versions."""
    try:
        # Newer sklearn (>=1.4) uses sparse_output
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # Older sklearn expects 'sparse'
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

# ---------------------------- main ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Discover personas from rows.")
    ap.add_argument("--csv", default=None, help=f"Path to CSV (default: look for {DEFAULT_CSV})")
    ap.add_argument("--text-col", default="text", help="Text column name (default: text)")
    ap.add_argument("--kmax", type=int, default=10, help="Max k to try for KMeans (default: 10)")
    ap.add_argument("--min_df", type=int, default=1, help="TF-IDF min_df (raise to 3–5 for large data)")
    args = ap.parse_args()

    csv_path = resolve_csv(args.csv)
    print(f"[info] Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[info] Shape: {df.shape} | Columns: {list(df.columns)}")

    print("[info] Building emotion probability features from emo*_label/conf ...")
    emo_wide = build_emotion_table(df)

    work = df.copy()
    work["sentiment_score"] = work.get("reddit_sentiment", pd.Series([None]*len(work))).apply(sentiment_to_score)
    work = pd.concat([work, emo_wide], axis=1)

    text_col = args.text_col
    if text_col not in work.columns:
        raise ValueError(f"text column '{text_col}' not found in CSV")

    cat_cols = [c for c in ["top_emotion", "horoscope", "job"] if c in work.columns]
    emo_cols = list(emo_wide.columns)
    num_cols = ["sentiment_score"] + emo_cols

    # Text pipeline: fill NaNs -> TF-IDF -> SVD
    text_pipe = Pipeline([
        ("fillna", FunctionTransformer(lambda s: s.fillna(""), validate=False)),
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=args.min_df)),
        ("svd", TruncatedSVD(n_components=50, random_state=42)),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("text", text_pipe, text_col),
            ("cat", make_ohe(), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    pipe = Pipeline([
        ("pre", pre),
        ("scale", StandardScaler(with_mean=False)),
    ])

    print("[info] Vectorizing + scaling ...")
    X = pipe.fit_transform(work)

    print("[info] Trying different k for KMeans ...")
    best_k, best_s = try_kmeans_k(X, ks=range(2, max(3, args.kmax + 1)))
    if best_k is None:
        print("[warn] Silhouette failed; using k=2.")
        best_k, best_s = 2, -1.0
    print(f"[info] Best k = {best_k} (silhouette={best_s:.4f})")

    km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    work["persona_id"] = labels

    print("[info] Building cluster descriptors ...")
    tfidf_plain = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=args.min_df)
    text_m = tfidf_plain.fit_transform(work[text_col].fillna(""))
    terms = np.array(tfidf_plain.get_feature_names_out())
    top_words = top_terms_per_cluster(text_m, terms, labels, topn=12)

    persona_cards = {}
    for c in sorted(np.unique(labels)):
        idx = labels == c
        sub = work[idx]
        card = {
            "size": int(idx.sum()),
            "share": float(idx.mean()),
            "top_words": top_words.get(int(c), []),
            "avg_sentiment_score": float(sub["sentiment_score"].mean()),
            "top_emotions": [],
            "horoscope_top": [],
            "job_top": [],
        }
        if emo_cols:
            emo_means = sub[emo_cols].mean().sort_values(ascending=False)
            card["top_emotions"] = [
                {"emotion": e.replace("emotion_", ""), "avg_conf": float(v)}
                for e, v in emo_means.head(8).items()
            ]
        if "horoscope" in sub:
            card["horoscope_top"] = [k for k, _ in Counter(sub["horoscope"].astype(str)).most_common(5)]
        if "job" in sub:
            card["job_top"] = [k for k, _ in Counter(sub["job"].astype(str)).most_common(5)]
        persona_cards[int(c)] = card

    print("[info] Ranking feature influence ...")
    pre_fnames: List[str] = [f"text_svd_{i}" for i in range(50)]
    if cat_cols:
        ohe = pipe.named_steps["pre"].named_transformers_["cat"]
        # get_feature_names_out may not exist on very old versions:
        try:
            pre_fnames += list(ohe.get_feature_names_out(cat_cols))
        except Exception:
            pre_fnames += [f"cat_{i}" for i in range(ohe.categories_.__len__())]
    pre_fnames += num_cols

    X_dense = X if isinstance(X, np.ndarray) else X.toarray()
    try:
        mi = mutual_info_classif(X_dense, labels, random_state=42)
        mi_ranking = sorted([(pre_fnames[i], float(mi[i])) for i in range(len(pre_fnames))],
                            key=lambda x: x[1], reverse=True)[:30]
    except Exception:
        mi_ranking = []

    try:
        clf = RandomForestClassifier(n_estimators=250, random_state=42)
        clf.fit(X_dense, labels)
        perm = permutation_importance(clf, X_dense, labels, n_repeats=8, random_state=42)
        pi_ranking = sorted([(pre_fnames[i], float(perm.importances_mean[i])) for i in range(len(pre_fnames))],
                            key=lambda x: x[1], reverse=True)[:30]
    except Exception:
        pi_ranking = []

    persona_summary = {
        "source_csv": str(csv_path),
        "n_rows": int(len(work)),
        "n_personas": int(best_k),
        "silhouette": float(best_s),
        "feature_importance": {
            "mutual_information_top30": mi_ranking,
            "permutation_importance_top30": pi_ranking
        },
        "personas": persona_cards
    }

    print("[info] Writing persona_assignments.csv ...")
    out_assign = df.copy()
    out_assign["persona_id"] = labels
    out_assign.to_csv("persona_assignments.csv", index=False)

    print("[info] Writing persona_cards.json ...")
    with open("persona_cards.json", "w") as f:
        json.dump(persona_summary, f, indent=2)

    print("[info] Creating persona_pca_scatter.png ...")
    try:
        pca2 = PCA(n_components=2, random_state=42)
        X2 = pca2.fit_transform(X_dense)
        plt.figure(figsize=(6, 5))
        for c in range(best_k):
            m = labels == c
            plt.scatter(X2[m, 0], X2[m, 1], s=16, label=f"persona {c}", alpha=0.8)
        plt.legend(); plt.title("Personas (PCA 2D)")
        plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout()
        plt.savefig("persona_pca_scatter.png", dpi=160)
    except Exception as e:
        print(f"[warn] PCA plot skipped: {e!r}")

    print("[done] Files saved: persona_assignments.csv, persona_cards.json, persona_pca_scatter.png")

if __name__ == "__main__":
    main()
