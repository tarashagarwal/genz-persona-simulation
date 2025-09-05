# cluster_personas.py  (final with reveal_flag)
import os
import numpy as np
import pandas as pd

from datasets import load_dataset
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Optional libs
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

try:
    import hdbscan
    HAS_HDBSCAN = True
except Exception:
    HAS_HDBSCAN = False

import torch
from sentence_transformers import SentenceTransformer

# Plotting
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
try:
    import plotly.express as px
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

# ---------------- Config ----------------
HF_REPO_ID = os.environ.get("HF_REPO_ID", "tarashagarwal/genz-persona-simulation")
SPLIT = "train"  # or "validation"

TEXT_COL = "text"
NUM_COLS = ["age", "birth_year_est", "emotion_conf", "reddit_conf", "is_genz", "masking"]

# gender excluded; keep zodiac/job + emotions
CAT_COLS = [
    "job", "horoscope",
    "top_emotion", "emotion_sentiment", "reddit_sentiment",
    "emo1_label", "emo2_label", "emo3_label", "emo4_label", "emo5_label"
]

TEXT_WEIGHT = 1.0
CAT_WEIGHT  = 1.0
NUM_WEIGHT  = 1.0

EMB_MODEL = os.environ.get("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
BATCH_SIZE = 64
MAX_CHARS = 1000

REDUCE_DIM = 50
PROJ_2D = "umap"         # "umap" if available else PCA fallback
N_CLUSTERS = 8
NUMBER_OF_DATA_POINTS = 97962

OUT_CSV  = "clusters_personas_genz_only.csv"
OUT_PNG  = "clusters_personas.png"
OUT_HTML = "clusters_personas.html"
SEED = 42
# ----------------------------------------


def load_hf_df(repo_id: str, split: str) -> pd.DataFrame:
    print(f"📥 Loading dataset from HF Hub: {repo_id} [{split}]")
    ds = load_dataset(repo_id, split=split)
    ds = ds.select(range(min(NUMBER_OF_DATA_POINTS, len(ds))))
    df = ds.to_pandas()
    print(f"   • Loaded rows: {len(df):,} | Cols: {len(df.columns)}")

    # Gen Z only if present
    if "is_genz" in df.columns:
        before = len(df)
        df = df[df["is_genz"] == 1].reset_index(drop=True)
        print(f"   • Kept Gen Z only (is_genz==1): {len(df):,} (dropped {before - len(df):,})")
    else:
        print("   • Warning: 'is_genz' column not found; no row filtering applied.")

    # Create reveal_flag if possible (1 = reveal, 0 = masked)
    if "masking" in df.columns:
        df["reveal_flag"] = (df["masking"].astype("float") == 0).astype("int64")
    else:
        df["reveal_flag"] = np.nan  # keep NaN if masking not present

    return df


def embed_texts(texts, model_name=EMB_MODEL, batch_size=BATCH_SIZE):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🧠 Loading SBERT: {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device)
    texts = [str(t)[:MAX_CHARS] if isinstance(t, str) else "" for t in texts]
    print(f"   • Embedding {len(texts):,} texts (batch_size={batch_size})")
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embs


def prepare_blocks(df: pd.DataFrame):
    # TEXT
    X_text = embed_texts(df[TEXT_COL].fillna("").astype(str).tolist())
    X_text = normalize(X_text) * TEXT_WEIGHT

    # NUMERIC
    num_df = pd.DataFrame()
    for c in NUM_COLS:
        num_df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    X_num = np.zeros((len(df), 0))
    if len(num_df.columns):
        X_num = StandardScaler().fit_transform(num_df.values)
        X_num = normalize(X_num) * NUM_WEIGHT

    # CATEGORICAL
    cat_df = pd.DataFrame()
    for c in CAT_COLS:
        cat_df[c] = df[c].astype(str).replace({"nan": np.nan}).fillna("NA") if c in df.columns else "NA"
    X_cat = np.zeros((len(df), 0))
    if len(cat_df.columns):
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_cat = ohe.fit_transform(cat_df.values)
        X_cat = normalize(X_cat) * CAT_WEIGHT

    X = np.concatenate([X_text, X_num, X_cat], axis=1)
    print(f"🔗 Blocks: text={X_text.shape[1]}, num={X_num.shape[1]}, cat={X_cat.shape[1]} → total={X.shape[1]}")
    return X


def reduce_dimensions(X):
    if REDUCE_DIM and REDUCE_DIM < X.shape[1]:
        print(f"🔻 PCA → {REDUCE_DIM} dims")
        pca = PCA(n_components=REDUCE_DIM, random_state=SEED)
        return pca.fit_transform(X)
    return X


def project_2d(Xr):
    if PROJ_2D.lower() == "umap" and HAS_UMAP:
        print("🗺️  UMAP 2D projection…")
        reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=30, min_dist=0.1)
        return reducer.fit_transform(Xr)
    print("🗺️  PCA 2D projection…")
    return PCA(n_components=2, random_state=SEED).fit_transform(Xr)


def cluster_features(Xr):
    if HAS_HDBSCAN:
        print("🔎 HDBSCAN clustering…")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=30, metric="euclidean")
        labels = clusterer.fit_predict(Xr)  # -1 = noise
        algo = "hdbscan"
    else:
        print(f"🔎 KMeans clustering (k={N_CLUSTERS})…")
        km = KMeans(n_clusters=N_CLUSTERS, n_init="auto", random_state=SEED)
        labels = km.fit_predict(Xr)
        algo = "kmeans"
    return labels, algo


# --------- Plotting ---------
def _cluster_colors(labels):
    uniq = np.unique(labels)
    uniq_sorted = [l for l in uniq if l != -1] + ([-1] if -1 in uniq else [])
    cmap = get_cmap("tab20")
    color_map = {}
    for i, lab in enumerate(uniq_sorted):
        color_map[lab] = (0.65, 0.65, 0.65, 0.6) if lab == -1 else (*cmap(i % 20)[:3], 0.8)
    return np.array([color_map[l] for l in labels]), color_map


def plot_matplotlib(df_out: pd.DataFrame, png_path=OUT_PNG, circle_size=60):
    print(f"🖼️  Saving matplotlib scatter: {png_path}")
    labels = df_out["cluster"].values
    colors, _ = _cluster_colors(labels)

    fig, ax = plt.subplots(figsize=(9, 7), dpi=140)
    ax.set_title("Persona clusters (Gen Z only)", fontsize=14)
    ax.scatter(df_out["x2d"], df_out["y2d"], s=circle_size, c=colors, marker="o", linewidths=0, edgecolors="none")

    # legend with sizes
    counts = df_out["cluster"].value_counts().sort_index()
    for lab, cnt in counts.items():
        lbl = f"cluster {lab}" if lab != -1 else "noise (-1)"
        ax.scatter([], [], c=_cluster_colors(np.array([lab]))[0], s=80, label=f"{lbl}  n={cnt}")

    ax.legend(loc="best", fontsize=8, frameon=True)
    ax.set_xlabel("x2d"); ax.set_ylabel("y2d"); ax.grid(alpha=0.15)
    plt.tight_layout(); plt.savefig(png_path); plt.close(fig)
    print("   • done.")


def _preview_text(s: str, n=140):
    s = (s or "").replace("\n", " ").strip()
    return (s[:n] + "…") if len(s) > n else s


def plot_plotly(df_out: pd.DataFrame, html_path=OUT_HTML):
    if not HAS_PLOTLY:
        print("ℹ️ Plotly not installed; skipping interactive HTML.")
        return
    print(f"🌐 Saving interactive Plotly scatter: {html_path}")
    hover_df = df_out.copy()
    hover_df["text_preview"] = hover_df[TEXT_COL].astype(str).map(lambda s: _preview_text(s, 200))
    fig = px.scatter(
        hover_df,
        x="x2d", y="y2d", color="cluster",
        hover_data={
            "cluster": True,
            "job": True if "job" in hover_df.columns else False,
            "horoscope": True if "horoscope" in hover_df.columns else False,
            "top_emotion": True if "top_emotion" in hover_df.columns else False,
            "reveal_flag": True if "reveal_flag" in hover_df.columns else False,
            "text_preview": True,
        },
        opacity=0.88, template="plotly_white",
        title="Persona clusters (Gen Z only, interactive)"
    )
    fig.update_traces(marker=dict(size=8, line=dict(width=0)))
    fig.update_layout(legend_title_text='cluster')
    fig.write_html(html_path, include_plotlyjs="cdn")
    print("   • done.")


# ---------------- Main ----------------
def main():
    df = load_hf_df(HF_REPO_ID, SPLIT)

    if TEXT_COL not in df.columns:
        raise ValueError(f"Missing '{TEXT_COL}' column.")

    X  = prepare_blocks(df)
    Xr = reduce_dimensions(X)
    labels, algo = cluster_features(Xr)
    Z = project_2d(Xr)

    df_out = df.copy()
    df_out["cluster"] = labels
    df_out["x2d"] = Z[:, 0]
    df_out["y2d"] = Z[:, 1]

    counts = df_out["cluster"].value_counts(dropna=False).sort_index()
    print("📊 Cluster sizes (Gen Z only):")
    print(counts.to_string())

    # Plots
    plot_matplotlib(df_out, png_path=OUT_PNG, circle_size=70)
    plot_plotly(df_out, html_path=OUT_HTML)

    # Save PRUNED CSV with reveal_flag included
    desired_cols = [
        "text", "top_emotion", "emo1_label", "emo2_label",
        "horoscope", "job", "reveal_flag",
        "cluster", "x2d", "y2d"
    ]
    for c in desired_cols:
        if c not in df_out.columns:
            df_out[c] = np.nan
    df_out[desired_cols].to_csv(OUT_CSV, index=False)
    print(f"💾 Saved pruned CSV: {OUT_CSV}")

    print("\n🔧 Tune TEXT_WEIGHT / CAT_WEIGHT / NUM_WEIGHT to balance influence.")
    print(f"🖼️  Figures saved: {OUT_PNG} and {OUT_HTML if HAS_PLOTLY else '(Plotly not installed)'}")


if __name__ == "__main__":
    main()
