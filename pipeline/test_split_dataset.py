"""Dependency-free checks for the pipeline's split safety rules."""
import json
import os
import tempfile

from select_videos import video_id
from split_dataset import split_ffpp_rows


def main():
    assert video_id("a/b", "000.mp4") == "a__b__000"

    with tempfile.TemporaryDirectory() as splits_dir:
        for split, pairs in {
            "train": [["000", "001"]],
            "val": [["002", "003"]],
            "test": [["004", "005"]],
        }.items():
            with open(os.path.join(splits_dir, f"{split}.json"), "w") as f:
                json.dump(pairs, f)

        rows = [
            {"video_name": "000", "label": "real"},
            {"video_name": "000_001", "label": "fake"},
            {"video_name": "002_003", "label": "fake"},
            {"video_name": "004_005", "label": "fake"},
        ]
        splits = split_ffpp_rows(rows, splits_dir)
        assert [row["video_name"] for row in splits["train"]] == ["000", "000_001"]
        assert [row["video_name"] for row in splits["val"]] == ["002_003"]
        assert [row["video_name"] for row in splits["test"]] == ["004_005"]

    print("split safety checks passed")


if __name__ == "__main__":
    main()
