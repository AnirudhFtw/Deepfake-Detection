"""
Stage: video-level train/val/test split, shared by every domain. Splitting
once here — instead of each domain's train.py re-deriving its own random
split — is what makes spatial, frequency, and (later) temporal results
comparable, and what makes fusion training/evaluation valid.
"""
import csv
import json
import os
import random
from collections import defaultdict

from config import OUTPUT_DIR, SPLITS_DIR, VAL_RATIO, TEST_RATIO, SEED

MANIFEST_PATH = os.path.join(OUTPUT_DIR, "manifest.csv")


def load_manifest():
    with open(MANIFEST_PATH) as f:
        return list(csv.DictReader(f))


def split_rows(rows, val_ratio, test_ratio, seed):
    rng = random.Random(seed)
    rows = rows[:]
    rng.shuffle(rows)

    n = len(rows)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)

    val = rows[:n_val]
    test = rows[n_val:n_val + n_test]
    train = rows[n_val + n_test:]

    return train, val, test


def main():
    rows = load_manifest()

    by_label = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    splits = {"train": [], "val": [], "test": []}

    for label, label_rows in by_label.items():
        train, val, test = split_rows(label_rows, VAL_RATIO, TEST_RATIO, SEED)
        splits["train"].extend(train)
        splits["val"].extend(val)
        splits["test"].extend(test)

    os.makedirs(SPLITS_DIR, exist_ok=True)

    for name, split_rows_ in splits.items():
        entries = [
            {"label": r["label"], "video": r["video_name"]}
            for r in split_rows_
        ]
        path = os.path.join(SPLITS_DIR, f"{name}.json")
        with open(path, "w") as f:
            json.dump(entries, f, indent=2)
        print(f"{name}: {len(entries)} videos -> {path}")


if __name__ == "__main__":
    main()
