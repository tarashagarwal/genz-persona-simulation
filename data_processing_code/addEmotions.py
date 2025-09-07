# add_emotion_and_genz_final.py
from datasets import load_dataset
from transformers import pipeline
from huggingface_hub import create_repo
import pandas as pd
from datetime import datetime
import os

# ---------------- Config ----------------
INPUT_FILES = {"train": "data/train.csv", "validation": "data/validation.csv"}
OUTPUT_FILES = {
    "train": "data/train_with_emotion_genz.csv",
    "validation": "data/validation_with_emotion_genz.csv",
}
DEVICE = 0
BATCH_SIZE = 32

GENZ_START, GENZ_END = 1997, 2012  # inclusive
DATE_IS_BIRTHDATE = True           # True = DOB like "23,November,2002", False = post date - age

LIMIT_ROWS = False   # set True for debugging
ROWS_LIMIT = 10

# Hugging Face push
PUSH_TO_HUB = True
HF_REPO_ID = os.environ.get("HF_REPO_ID", "tarashagarwal/genz-persona-simulation")
HF_PRIVATE = True
# ----------------------------------------


def parse_year(s: str) -> int | None:
    try:
        return datetime.strptime(str(s).strip(), "%d,%B,%Y").year
    except Exception:
        return None


def add_genz_flags(path: str) -> str:
    df = pd.read_csv(path)

    if LIMIT_ROWS:
        df = df.head(ROWS_LIMIT)
        print(f"⚡ Limiting {path} to {ROWS_LIMIT} rows")

    if DATE_IS_BIRTHDATE:
        df["birth_year_est"] = df["date"].apply(parse_year)
    else:
        post_year = df["date"].apply(parse_year)
        age = pd.to_numeric(df["age"], errors="coerce")
        df["birth_year_est"] = post_year - age

    # Boolean flag → int (0/1)
    df["is_genz"] = df["birth_year_est"].between(GENZ_START, GENZ_END, inclusive="both").astype(int)

    tmp_path = path.replace(".csv", "_with_genz.csv")
    df.to_csv(tmp_path, index=False)
    return tmp_path


# 1) Add Gen Z flags to CSVs
train_path = add_genz_flags(INPUT_FILES["train"])
val_path = add_genz_flags(INPUT_FILES["validation"])

# 2) Load with Hugging Face
dataset = load_dataset("csv", data_files={"train": train_path, "validation": val_path})

# 3) Build GoEmotions pipeline
clf = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=None,
    device=DEVICE,
    truncation=True,
    padding=True,
)

# 4) Add top emotion
def add_top_emotion(batch):
    outputs = clf(batch["text"])
    return {"top_emotion": [max(s, key=lambda x: x["score"])["label"] for s in outputs]}

dataset = dataset.map(add_top_emotion, batched=True, batch_size=BATCH_SIZE)

# 5) Save only the FINAL outputs
dataset["train"].to_csv(OUTPUT_FILES["train"], index=False)
dataset["validation"].to_csv(OUTPUT_FILES["validation"], index=False)

print("✅ Final CSVs saved:")
print(f" - {OUTPUT_FILES['train']}")
print(f" - {OUTPUT_FILES['validation']}")

# 6) Push to Hugging Face Hub
if PUSH_TO_HUB:
    create_repo(repo_id=HF_REPO_ID, repo_type="dataset", private=HF_PRIVATE, exist_ok=True)
    dataset.push_to_hub(HF_REPO_ID)
    print(f"✅ Final dataset pushed: https://huggingface.co/datasets/{HF_REPO_ID}")
