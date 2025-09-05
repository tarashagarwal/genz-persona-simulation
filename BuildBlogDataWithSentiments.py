# enrich_and_label_masking_push_safe_verbose_top5_range_named_rows.py
import os
from datetime import datetime
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import create_repo
from datasets import load_dataset
from tqdm import tqdm

# ---------------- Config ----------------
INPUT_FILES = {"train": "data/train.csv", "validation": "data/validation.csv"}
BASE_OUTPUT_FILES = {"train": "data/train_final.csv", "validation": "data/validation_final.csv"}

DEVICE = 0           # 0 = first GPU, -1 = CPU
BATCH_SIZE = 32
CONF_MIN = 0.90      # both models must be >= this to keep the row

GENZ_START, GENZ_END = 1997, 2012
DATE_IS_BIRTHDATE = True     # True: 'date' is DOB like "23,November,2002"

# Use RANGE or ROWS_LIMIT. If RANGE is provided, it takes precedence.
START_ROW = None        # 1-based inclusive (set None to disable)
END_ROW   = None      # 1-based inclusive (set None to go till end)
ROWS_LIMIT = None    # e.g., 200 when RANGE not used

PUSH_TO_HUB = True
HF_REPO_ID = os.environ.get("HF_REPO_ID", "tarashagarwal/genz-persona-simulation")
HF_PRIVATE = True
ROW_LOG_PREVIEW_CHARS = 60  # characters of text to print per-row in logs
# ----------------------------------------


# ---------- helpers ----------
def parse_year(s: str) -> int | None:
    try:
        return datetime.strptime(str(s).strip(), "%d,%B,%Y").year
    except Exception:
        return None

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

def top5_from_scores(scores):
    """scores -> (top1_label, top1_conf, top5_labels[5], top5_scores[5])"""
    if not scores:
        return None, None, [None]*5, [None]*5
    sorted_scores = sorted(scores, key=lambda x: x["score"], reverse=True)
    top1 = sorted_scores[0]
    top5 = sorted_scores[:5] if len(sorted_scores) >= 5 else (sorted_scores + [{"label":None,"score":None}]*(5-len(sorted_scores)))
    top5_labels = [e["label"] for e in top5]
    top5_scores = [float(e["score"]) if e["score"] is not None else None for e in top5]
    return str(top1["label"]), float(top1["score"]), top5_labels, top5_scores


# ---------- dynamic output naming ----------
OUTPUT_FILES = {}
for split, path in BASE_OUTPUT_FILES.items():
    base, ext = os.path.splitext(path)
    if START_ROW is not None or END_ROW is not None:
        start_str = START_ROW if START_ROW is not None else "start"
        end_str = END_ROW if END_ROW is not None else "end"
        path = f"{base}_rows{start_str}-{end_str}{ext}"
    OUTPUT_FILES[split] = path


# ---------- models ----------
print("🔧 Loading models...")
emo_clf = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=None,          # return all labels; we’ll sort for top-5
    device=DEVICE,
    truncation=True,
    padding=True,
)
print("  • GoEmotions pipeline ready on", emo_clf.device)

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
    max_length=256   # safe for Reddit-style short text
)
print("  • Reddit sentiment pipeline ready on", sent_clf.device)
print()


# ---------- batch-safe inference with per-row logs ----------
def safe_emotions_for_batch(batch_texts, start_idx=0):
    """
    Return lists per item:
      top1_label, top1_conf, coarse_sentiment_from_top1,
      top5_label_1..5, top5_conf_1..5
    Any error on an item yields Nones for that item.
    """
    out_top1_lbl, out_top1_conf, out_sent = [], [], []
    out_top5_lbls, out_top5_confs = [], []

    try:
        outputs = emo_clf(batch_texts)  # list[list[dict]]
        for j, scores in enumerate(outputs):
            row_num = start_idx + j
            preview = batch_texts[j][:ROW_LOG_PREVIEW_CHARS].replace("\n"," ")
            print(f"     📝 GoEmotions row {row_num}: {preview!r}")
            try:
                t1_lbl, t1_conf, t5_lbls, t5_confs = top5_from_scores(scores)
                out_top1_lbl.append(t1_lbl); out_top1_conf.append(t1_conf)
                out_sent.append(emo_to_sentiment(t1_lbl) if t1_lbl is not None else None)
                out_top5_lbls.append(t5_lbls); out_top5_confs.append(t5_confs)
            except Exception as e:
                print(f"       ⚠️ GoEmotions item error at row {row_num}: {e}")
                out_top1_lbl.append(None); out_top1_conf.append(None); out_sent.append(None)
                out_top5_lbls.append([None]*5); out_top5_confs.append([None]*5)
    except Exception as e:
        print(f"   ⚠️ GoEmotions batch-level error: {e}")
        for j, t in enumerate(batch_texts):
            row_num = start_idx + j
            preview = t[:ROW_LOG_PREVIEW_CHARS].replace("\n"," ")
            try:
                print(f"     📝 Fallback GoEmotions row {row_num}: {preview!r}")
                scores = emo_clf([t])[0]
                t1_lbl, t1_conf, t5_lbls, t5_confs = top5_from_scores(scores)
                out_top1_lbl.append(t1_lbl); out_top1_conf.append(t1_conf)
                out_sent.append(emo_to_sentiment(t1_lbl) if t1_lbl is not None else None)
                out_top5_lbls.append(t5_lbls); out_top5_confs.append(t5_confs)
            except Exception as e2:
                print(f"       ⚠️ GoEmotions per-row error at {row_num}: {e2}")
                out_top1_lbl.append(None); out_top1_conf.append(None); out_sent.append(None)
                out_top5_lbls.append([None]*5); out_top5_confs.append([None]*5)

    return out_top1_lbl, out_top1_conf, out_sent, out_top5_lbls, out_top5_confs


def safe_sentiment_for_batch(batch_texts, start_idx=0):
    """Return lists: label_str, confidence. Any error on an item yields None; includes per-row logs."""
    out_lbls, out_confs = [], []
    try:
        outputs = sent_clf(batch_texts)  # list[dict]
        for j, o in enumerate(outputs):
            row_num = start_idx + j
            preview = batch_texts[j][:ROW_LOG_PREVIEW_CHARS].replace("\n"," ")
            print(f"     💬 RedditSent row {row_num}: {preview!r}")
            try:
                lbl = REDDIT_MAP.get(str(o["label"]), "neutral")
                conf = float(o["score"])
                out_lbls.append(lbl); out_confs.append(conf)
            except Exception as e:
                print(f"       ⚠️ RedditSent item error at row {row_num}: {e}")
                out_lbls.append(None); out_confs.append(None)
    except Exception as e:
        print(f"   ⚠️ RedditSent batch-level error: {e}")
        for j, t in enumerate(batch_texts):
            row_num = start_idx + j
            preview = t[:ROW_LOG_PREVIEW_CHARS].replace("\n"," ")
            try:
                print(f"     💬 Fallback RedditSent row {row_num}: {preview!r}")
                o = sent_clf([t])[0]
                lbl = REDDIT_MAP.get(str(o["label"]), "neutral")
                conf = float(o["score"])
                out_lbls.append(lbl); out_confs.append(conf)
            except Exception as e2:
                print(f"       ⚠️ RedditSent per-row error at {row_num}: {e2}")
                out_lbls.append(None); out_confs.append(None)
    return out_lbls, out_confs


# ---------- range/limit ----------
def apply_row_range(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Apply 1-based inclusive [START_ROW, END_ROW] or fall back to ROWS_LIMIT."""
    if START_ROW is not None or END_ROW is not None:
        start_idx = max(0, (START_ROW or 1) - 1)
        end_idx = END_ROW if END_ROW is not None else None  # iloc end is exclusive
        before = len(df)
        df = df.iloc[start_idx:end_idx].reset_index(drop=True)
        print(f"  • Using RANGE [{START_ROW or 1}..{END_ROW or before}] on {split_name}: kept {len(df)} rows out of {before}")
    elif ROWS_LIMIT:
        before = len(df)
        df = df.head(ROWS_LIMIT).reset_index(drop=True)
        print(f"  • Limiting to first {ROWS_LIMIT} rows on {split_name}: kept {len(df)} rows out of {before}")
    return df


# ---------- processing ----------
def process_split(in_path: str, out_path: str, split_name: str):
    print(f"📥 Loading: {in_path}")
    df = pd.read_csv(in_path)
    print(f"  • Rows loaded: {len(df)}")

    # Apply range/limit
    df = apply_row_range(df, split_name)

    # schema checks
    if "text" not in df.columns:
        raise ValueError(f"`text` column not found in {in_path}")
    if "date" not in df.columns:
        raise ValueError(f"`date` column not found in {in_path}")

    print("🧮 Computing Gen Z flags...")
    df = add_genz(df)
    print("  • Example birth years:", df["birth_year_est"].head(min(3, len(df))).tolist())

    print("🧠 Running GoEmotions (batched with per-row logs)...")
    texts = df["text"].astype(str).fillna("").tolist()
    n = len(texts)

    top1_lbls, top1_confs, emo_sent_coarse = [], [], []
    top5_lbl_lists, top5_conf_lists = [], []

    for i in tqdm(range(0, n, BATCH_SIZE), desc=f"  GoEmotions {split_name}"):
        t1l, t1c, t_sent, t5l, t5c = safe_emotions_for_batch(texts[i:i+BATCH_SIZE], start_idx=i)
        top1_lbls.extend(t1l); top1_confs.extend(t1c); emo_sent_coarse.extend(t_sent)
        top5_lbl_lists.extend(t5l); top5_conf_lists.extend(t5c)

    print("💬 Running Reddit sentiment (batched with per-row logs)...")
    red_sent, red_conf = [], []
    for i in tqdm(range(0, n, BATCH_SIZE), desc=f"  RedditSent {split_name}"):
        rl, rc = safe_sentiment_for_batch(texts[i:i+BATCH_SIZE], start_idx=i)
        red_sent.extend(rl); red_conf.extend(rc)

    print("🧷 Attaching columns...")
    df["top_emotion"] = top1_lbls
    df["emotion_conf"] = top1_confs
    df["emotion_sentiment"] = emo_sent_coarse
    df["reddit_sentiment"] = red_sent
    df["reddit_conf"] = red_conf

    # Top-5 emotion columns
    for k in range(5):
        df[f"emo{k+1}_label"] = [lst[k] if lst and len(lst) > k else None for lst in top5_lbl_lists]
        df[f"emo{k+1}_conf"]  = [lst[k] if lst and len(lst) > k else None for lst in top5_conf_lists]

    print(f"🔎 Filtering by confidence >= {CONF_MIN} and valid predictions...")
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
    print(f"  • Kept {len(df)} rows, dropped {dropped} (errors/low conf)")

    print("🎭 Computing masking flag (1=masked, 0=reveal)...")
    df["masking"] = (df["emotion_sentiment"] != df["reddit_sentiment"]).astype(int)
    print("  • Masking rate:", df["masking"].mean() if len(df) else 0.0)

    print(f"💾 Saving final: {out_path}")
    df.to_csv(out_path, index=False)
    print(f"✅ Saved {out_path} (final rows: {len(df)})\n")


def main():
    print("====== START ======")
    for split in ("train", "validation"):
        print(f"\n=== Split: {split} ===")
        process_split(INPUT_FILES[split], OUTPUT_FILES[split], split)

    if PUSH_TO_HUB:
        print("☁️  Preparing push to Hugging Face Hub…")
        create_repo(repo_id=HF_REPO_ID, repo_type="dataset", private=HF_PRIVATE, exist_ok=True)
        print("  • Loading final CSVs into DatasetDict...")
        ds = load_dataset("csv", data_files=OUTPUT_FILES)
        print("  • Pushing to Hub:", HF_REPO_ID)
        ds.push_to_hub(HF_REPO_ID)
        print(f"🚀 Pushed to https://huggingface.co/datasets/{HF_REPO_ID}")

    print("====== DONE ======")


if __name__ == "__main__":
    main()
