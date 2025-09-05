# cluster_personas.py
import os
import numpy as np
import pandas as pd

from datasets import load_dataset
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

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

# ---- plotting deps ----
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import to_hex, CSS4_COLORS, to_rgb
from matplotlib.patches import Patch
import math

try:
    import plotly.express as px
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False


# ---------------- Config ----------------
HF_REPO_ID = os.environ.get("HF_REPO_ID", "tarashagarwal/genz-persona-simulation")
SPLIT = "train"

TEXT_COL = "text"
NUM_COLS = ["age", "birth_year_est", "emotion_conf", "reddit_conf", "is_genz", "masking"]

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
PROJ_2D = "umap"
N_CLUSTERS = 8
NUMBER_OF_DATA_POINTS = 97962

OUT_CSV        = "clusters_personas_genz_only.csv"
OUT_LEGEND_CSV = "clusters_personas_cluster_colors.csv"
OUT_PNG        = "clusters_personas.png"
OUT_HTML       = "clusters_personas.html"

SEED = 42
# ----------------------------------------


def load_hf_df(repo_id: str, split: str) -> pd.DataFrame:
    ds = load_dataset(repo_id, split=split)
    ds = ds.select(range(min(NUMBER_OF_DATA_POINTS, len(ds))))
    df = ds.to_pandas()

    if "is_genz" in df.columns:
        df = df[df["is_genz"] == 1].reset_index(drop=True)
    return df


def embed_texts(texts, model_name=EMB_MODEL, batch_size=BATCH_SIZE):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    texts = [str(t)[:MAX_CHARS] if isinstance(t, str) else "" for t in texts]
    embs = model.encode(
        texts, batch_size=batch_size,
        show_progress_bar=True, convert_to_numpy=True,
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
    scaler = StandardScaler()
    X_num = scaler.fit_transform(num_df.values) if len(num_df.columns) else np.zeros((len(df), 0))
    if X_num.shape[1] > 0:
        X_num = normalize(X_num) * NUM_WEIGHT

    # CATEGORICAL
    cat_df = pd.DataFrame()
    for c in CAT_COLS:
        cat_df[c] = df[c].astype(str).replace({"nan": np.nan}).fillna("NA") if c in df.columns else "NA"
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat = ohe.fit_transform(cat_df.values) if len(cat_df.columns) else np.zeros((len(df), 0))
    if X_cat.shape[1] > 0:
        X_cat = normalize(X_cat) * CAT_WEIGHT

    return np.concatenate([X_text, X_num, X_cat], axis=1)


def reduce_dimensions(X):
    if REDUCE_DIM and REDUCE_DIM < X.shape[1]:
        pca = PCA(n_components=REDUCE_DIM, random_state=SEED)
        return pca.fit_transform(X)
    return X


def project_2d(Xr):
    if PROJ_2D.lower() == "umap" and HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=30, min_dist=0.1)
        return reducer.fit_transform(Xr)
    else:
        p2 = PCA(n_components=2, random_state=SEED)
        return p2.fit_transform(Xr)


def cluster_features(Xr):
    if HAS_HDBSCAN:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=30, metric="euclidean")
        return clusterer.fit_predict(Xr), "hdbscan"
    km = KMeans(n_clusters=N_CLUSTERS, n_init="auto", random_state=SEED)
    return km.fit_predict(Xr), "kmeans"


# ---- Color handling ----
def closest_color_name(hex_color):
    target_rgb = np.array(to_rgb(hex_color))
    min_dist = math.inf
    closest_name = None
    for name, hexval in CSS4_COLORS.items():
        rgb = np.array(to_rgb(hexval))
        dist = np.linalg.norm(target_rgb - rgb)
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name


def build_cluster_colors(labels):
    unique = np.unique(labels)
    unique_sorted = [l for l in unique if l != -1] + ([-1] if -1 in unique else [])
    cmap = get_cmap("tab20")

    color_map = {}
    for i, lab in enumerate(unique_sorted):
        rgba = (0.65, 0.65, 0.65, 0.8) if lab == -1 else (*cmap(i % 20)[:3], 0.85)
        hex_color = to_hex(rgba, keep_alpha=False)
        color_map[lab] = closest_color_name(hex_color)

    row_colors = np.array([color_map[l] for l in labels])
    counts = pd.Series(labels).value_counts().sort_index()
    legend_df = pd.DataFrame({
        "cluster": counts.index,
        "count": counts.values,
        "color_name": [color_map[c] for c in counts.index]
    })
    return row_colors, legend_df, color_map


# ---------------- PLOTTING ----------------
def plot_matplotlib(df_out: pd.DataFrame, png_path=OUT_PNG):
    labels = df_out["cluster"].values
    color_names, legend_df, _ = build_cluster_colors(labels)

    fig, ax = plt.subplots(figsize=(9, 7), dpi=140)
    ax.set_title("Persona clusters (Gen Z only)")
    ax.scatter(df_out["x2d"], df_out["y2d"], s=60, c=color_names)

    patches = [Patch(facecolor=name, label=f"cluster {lab}  n={cnt}  {name}")
               for lab, cnt, name in zip(legend_df["cluster"], legend_df["count"], legend_df["color_name"])]
    ax.legend(handles=patches, loc="best", fontsize=8)
    plt.savefig(png_path)
    plt.close(fig)


# ---------------- MAIN ----------------
def main():
    df = load_hf_df(HF_REPO_ID, SPLIT)
    X = prepare_blocks(df)
    Xr = reduce_dimensions(X)
    labels, algo = cluster_features(Xr)
    Z = project_2d(Xr)

    # attach results
    df_out = df.copy()
    df_out["cluster"] = labels
    df_out["x2d"], df_out["y2d"] = Z[:, 0], Z[:, 1]

    row_colors, legend_df, _ = build_cluster_colors(labels)
    df_out["cluster_color_name"] = row_colors

    # Save main CSV
    desired_cols = [
        "text", "top_emotion",
        "emo1_label", "emo1_conf",
        "emo2_label", "emo2_conf",
        "emo3_label", "emo3_conf",
        "emo4_label", "emo4_conf",
        "horoscope", "job",
        "cluster", "x2d", "y2d", "cluster_color_name",
    ]
    for c in desired_cols:
        if c not in df_out.columns:
            df_out[c] = np.nan
    df_out[desired_cols].to_csv(OUT_CSV, index=False)

    # Save legend
    legend_df.to_csv(OUT_LEGEND_CSV, index=False)

    # Plot
    plot_matplotlib(df_out, png_path=OUT_PNG)
    print(f"💾 Saved: {OUT_CSV}, {OUT_LEGEND_CSV}, {OUT_PNG}")


if __name__ == "__main__":
    main()
