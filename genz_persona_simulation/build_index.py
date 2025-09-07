# build_indexes.py
import pandas as pd
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

CSV_FILE = "persona_database.csv"
PARQUET_FILE = "personas.parquet"
INDEX_DIR = Path("indexes")
TEXT_COL = "text"
ID_COL = "id"
PERSONA_COL = "persona_id"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

INDEX_DIR.mkdir(parents=True, exist_ok=True)

def load_dataset():
    if Path(PARQUET_FILE).exists():
        print(f"[info] Loading {PARQUET_FILE}")
        return pd.read_parquet(PARQUET_FILE)
    else:
        print(f"[info] Loading {CSV_FILE}")
        df = pd.read_csv(CSV_FILE)
        df.to_parquet(PARQUET_FILE, index=False)
        print(f"[info] Saved Parquet {PARQUET_FILE}")
        return df

def build_and_save_indexes(df):
    model = SentenceTransformer(EMB_MODEL)

    for pid, sub in df.groupby(PERSONA_COL):
        texts = sub[TEXT_COL].fillna("").tolist()
        vecs = model.encode(texts, normalize_embeddings=True).astype("float32")

        index = faiss.IndexFlatIP(vecs.shape[1])  # cosine similarity
        index.add(vecs)

        faiss.write_index(index, str(INDEX_DIR / f"persona_{pid}.faiss"))
        sub.reset_index(drop=True).to_parquet(INDEX_DIR / f"persona_{pid}.meta.parquet")

        print(f"[info] Built index for persona {pid} with {len(sub)} rows")

if __name__ == "__main__":
    df = load_dataset()
    build_and_save_indexes(df)
    print("[done] All persona indexes saved in ./indexes/")
