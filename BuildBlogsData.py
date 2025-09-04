import os
import glob
from typing import Iterator, Dict

from datasets import Dataset, DatasetDict
from tabulate import tabulate  # pip install tabulate


def iter_examples(files) -> Iterator[Dict]:
    def parse_date(line: str) -> str:
        return line.strip().split("<date>")[-1].split("</date>")[0]

    for file_path in files:
        file_name = os.path.basename(file_path)
        try:
            file_id, gender, age, job, horoscope = tuple(file_name.split(".")[:-1])
        except ValueError as e:
            raise ValueError(
                f"Unexpected filename format for '{file_name}'. "
                "Expected '<id>.<gender>.<age>.<job>.<horoscope>.xml'"
            ) from e

        with open(file_path, encoding="latin_1") as f:
            current_date = ""
            for raw_line in f:
                line = raw_line.strip()
                if "<date>" in line:
                    current_date = parse_date(line)
                elif line and not line.startswith("<"):
                    yield {
                        "text": line,
                        "date": current_date,
                        "gender": gender,
                        "age": int(age),
                        "job": job,
                        "horoscope": horoscope,
                    }


def build_dataset(data_dir: str = None) -> DatasetDict:
    data_dir = data_dir or os.environ.get("BLOGS_DIR", os.path.expanduser("~/Downloads/blogs"))
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Blogs directory not found: {data_dir}")

    files = sorted(glob.glob(os.path.join(data_dir, "*.xml")))
    if not files:
        raise FileNotFoundError(f"No .xml files found under {data_dir}")

    # 95/5 split by index (every 20th file -> validation)
    val_files = [fp for i, fp in enumerate(files) if i % 20 == 0]
    train_files = [fp for i, fp in enumerate(files) if i % 20 != 0]

    train_ds = Dataset.from_generator(iter_examples, gen_kwargs={"files": train_files})
    val_ds = Dataset.from_generator(iter_examples, gen_kwargs={"files": val_files})

    # Ensure a stable schema
    train_ds = train_ds.cast_column("age", train_ds.features["age"])
    val_ds = val_ds.cast_column("age", val_ds.features["age"])

    return DatasetDict(train=train_ds, validation=val_ds)


if __name__ == "__main__":
    ds = build_dataset()  # uses BLOGS_DIR env var or ~/Downloads/blogs

    # Save both splits to CSV
    print("💾 Saving train.csv and validation.csv...")
    ds["train"].to_csv("train.csv", index=False)
    ds["validation"].to_csv("validation.csv", index=False)

    print("✅ Done! train.csv and validation.csv created.")

