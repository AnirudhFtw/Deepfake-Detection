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

from config import OUTPUT_DIR, SPLITS_DIR, VAL_RATIO, TEST_RATIO, SEED, FFPP_SPLITS_DIR

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


def load_ffpp_splits(splits_dir):
    """Load the official FF++ pair lists and map each clip to its split."""
    clip_splits, original_splits = {}, {}
    for split in ("train", "val", "test"):
        with open(os.path.join(splits_dir, f"{split}.json")) as f:
            pairs = json.load(f)
        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError(f"Invalid FF++ pair in {split}.json: {pair!r}")
            clip_splits["_".join(pair)] = split
            original = pair[0]  # FF++ names manipulated clips target_source.
            original_splits.setdefault(original, set()).add(split)
    return clip_splits, original_splits


def split_ffpp_rows(rows, splits_dir):
    clip_splits, original_splits = load_ffpp_splits(splits_dir)
    splits = {"train": [], "val": [], "test": []}
    for row in rows:
        name = row["video_name"]
        split = clip_splits.get(name)
        if split is None:
            matches = original_splits.get(name, set())
            if len(matches) != 1:
                raise ValueError(
                    f"Cannot assign {name!r} to one FF++ split; check the official split files."
                )
            split = matches.pop()
        splits[split].append(row)
    return splits


def main():
    rows = load_manifest()

    by_label = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    if FFPP_SPLITS_DIR:
        splits = split_ffpp_rows(rows, FFPP_SPLITS_DIR)
    else:
        splits = {"train": [], "val": [], "test": []}
        for label_rows in by_label.values():
            train, val, test = split_rows(label_rows, VAL_RATIO, TEST_RATIO, SEED)
            splits["train"].extend(train)
            splits["val"].extend(val)
            splits["test"].extend(test)

    os.makedirs(SPLITS_DIR, exist_ok=True)

    for name, split_rows_ in splits.items():
        entries = [
            {"label": r["label"], "video": r["video_id"]}
            for r in split_rows_
        ]
        path = os.path.join(SPLITS_DIR, f"{name}.json")
        with open(path, "w") as f:
            json.dump(entries, f, indent=2)
        print(f"{name}: {len(entries)} videos -> {path}")


if __name__ == "__main__":
    main()
