# enrich_and_label_masking_push_safe.py
import os
from datetime import datetime
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import create_repo
from datasets import load_dataset

# ---------------- Config ----------------
INPUT_FILES = {"train": "data/train.csv", "validation": "data/validation.csv"}
OUTPUT_FILES = {"train": "data/train_final.csv", "validation": "data/validation_final.csv"}

DEVICE = 0           # 0 = first GPU, -1 = CPU
BATCH_SIZE = 32
CONF_MIN = 0.90      # both models must be >= this to keep the row

# Gen Z window
GENZ_START, GENZ_END = 1997, 2012
DATE_IS_BIRTHDATE = True     # True: 'date' is DOB like "23,November,2002"

# Optional row cap for quick testing (None for full)
ROWS_LIMIT = None

# Push to Hugging Face
PUSH_TO_HUB = True
HF_REPO_ID = os.environ.get("HF_REPO_ID", "tarashagarwal/genz-persona-simulation")
HF_PRIVATE = True
# ----------------------------------------


# ---------- helpers ----------
def parse_year(s: str) -> int | None:
    try: return datetime.strptime(str(s).strip(), "%d,%B,%Y").year
    except Exception: return None

def add_genz(df: pd.DataFrame) -> pd.DataFrame:
    if DATE_IS_BIRTHDATE:
        df["birth_year_est"] = df["date"].apply(parse_year)
    else:
        post_year = df["date"].apply(parse_year)
        age = pd.to_numeric(df.get("age", None), errors="coerce")
        df["birth_year_est"] = post_year - age
    df["is_genz"] = df["birth_year_est"].between(GENZ_START, GENZ_END, inclusive="both").astype(int)
    return df

POS_EMOS = {"admiration","amusement","gratitude","joy","love","optimism","pride","relief","caring","excitement"}
NEG_EMOS = {"anger","annoyance","disappointment","disapproval","disgust","embarrassment","fear","grief","nervousness","remorse","sadness","confusion"}
NEU_EMOS = {"neutral","realization","curiosity","surprise"}

def emo_to_sentiment(label: str) -> str:
    if label in POS_EMOS: return "positive"
    if label in NEG_EMOS: return "negative"
    return "neutral"

REDDIT_MAP = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}

# ---------- models ----------
emo_clf = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=None,
    device=DEVICE,
    truncation=True,
    padding=True,
)

reddit_model_name = "minh21/XLNet-Reddit-Sentiment-Analysis"
reddit_tokenizer = AutoTokenizer.from_pretrained(reddit_model_name)
reddit_model = AutoModelForSequenceClassification.from_pretrained(reddit_model_name)
sent_clf = pipeline(
    "sentiment-analysis",
    model=reddit_model,
    tokenizer=reddit_tokenizer,
    device=DEVICE,
    truncation=True,
    padding=True,
)

def safe_emotions_for_batch(batch_texts):
    """Return lists: top_label, conf, coarse_sentiment. Skips rows that error."""
    out_labels, out_confs, out_sent = [], [], []
    try:
        outputs = emo_clf(batch_texts)  # list[list[dict]]
        for scores in outputs:
            try:
                top = max(scores, key=lambda x: x["score"])
                lbl = str(top["label"])
                conf = float(top["score"])
                out_labels.append(lbl)
                out_confs.append(conf)
                out_sent.append(emo_to_sentiment(lbl))
            except Exception:
                out_labels.append(None); out_confs.append(None); out_sent.append(None)
    except Exception:
        # fallback: per-row
        for t in batch_texts:
            try:
                scores = emo_clf([t])[0]
                top = max(scores, key=lambda x: x["score"])
                lbl = str(top["label"])
                conf = float(top["score"])
                out_labels.append(lbl)
                out_confs.append(conf)
                out_sent.append(emo_to_sentiment(lbl))
            except Exception:
                out_labels.append(None); out_confs.append(None); out_sent.append(None)
    return out_labels, out_confs, out_sent

def safe_sentiment_for_batch(batch_texts):
    """Return lists: label_str, confidence. Skips rows that error."""
    out_lbls, out_confs = [], []
    try:
        outputs = sent_clf(batch_texts)  # list[dict]
        for o in outputs:
            try:
                lbl = REDDIT_MAP.get(str(o["label"]), "neutral")
                conf = float(o["score"])
                out_lbls.append(lbl); out_confs.append(conf)
            except Exception:
                out_lbls.append(None); out_confs.append(None)
    except Exception:
        for t in batch_texts:
            try:
                o = sent_clf([t])[0]
                lbl = REDDIT_MAP.get(str(o["label"]), "neutral")
                conf = float(o["score"])
                out_lbls.append(lbl); out_confs.append(conf)
            except Exception:
                out_lbls.append(None); out_confs.append(None)
    return out_lbls, out_confs

def process_split(in_path: str, out_path: str):
    # 1) Load
    df = pd.read_csv(in_path)
    if ROWS_LIMIT: df = df.head(ROWS_LIMIT)

    if "text" not in df.columns:
        raise ValueError(f"`text` column not found in {in_path}")
    if "date" not in df.columns:
        raise ValueError(f"`date` column not found in {in_path}")

    # 2) Gen Z flag
    df = add_genz(df)

    # 3) Emotions (batched, error-safe)
    texts = df["text"].astype(str).fillna("").tolist()
    top_emotion, emo_conf, emo_sent = [], [], []
    for i in range(0, len(texts), BATCH_SIZE):
        tl, tc, ts = safe_emotions_for_batch(texts[i:i+BATCH_SIZE])
        top_emotion.extend(tl); emo_conf.extend(tc); emo_sent.extend(ts)

    # 4) Reddit sentiment (batched, error-safe)
    red_sent, red_conf = [], []
    for i in range(0, len(texts), BATCH_SIZE):
        rl, rc = safe_sentiment_for_batch(texts[i:i+BATCH_SIZE])
        red_sent.extend(rl); red_conf.extend(rc)

    # 5) Attach columns
    df["top_emotion"] = top_emotion
    df["emotion_conf"] = emo_conf
    df["emotion_sentiment"] = emo_sent
    df["reddit_sentiment"] = red_sent
    df["reddit_conf"] = red_conf

    # 6) Drop rows that errored (None) or low confidence
    good = (
        df["top_emotion"].notna()
        & df["emotion_sentiment"].notna()
        & df["reddit_sentiment"].notna()
        & (pd.to_numeric(df["emotion_conf"], errors="coerce") >= CONF_MIN)
        & (pd.to_numeric(df["reddit_conf"], errors="coerce") >= CONF_MIN)
    )
    before = len(df)
    df = df.loc[good].reset_index(drop=True)
    dropped = before - len(df)
    print(f"[{os.path.basename(in_path)}] kept {len(df)} rows, dropped {dropped} (errors/low conf)")

    # 7) masking (1 = masked, sentiments differ; 0 = reveal)
    df["masking"] = (df["emotion_sentiment"] != df["reddit_sentiment"]).astype(int)

    # 8) Save FINAL file
    df.to_csv(out_path, index=False)
    print(f"✅ Saved {out_path}")

def main():
    for split in ("train","validation"):
        process_split(INPUT_FILES[split], OUTPUT_FILES[split])

    # Push both final CSVs to HF Hub as a dataset repo
    if PUSH_TO_HUB:
        create_repo(repo_id=HF_REPO_ID, repo_type="dataset", private=HF_PRIVATE, exist_ok=True)
        # Load the final CSVs as a DatasetDict and push
        ds = load_dataset("csv", data_files={
            "train": OUTPUT_FILES["train"],
            "validation": OUTPUT_FILES["validation"],
        })
        ds.push_to_hub(HF_REPO_ID)
        print(f"🚀 Pushed to https://huggingface.co/datasets/{HF_REPO_ID}")

if __name__ == "__main__":
    main()
