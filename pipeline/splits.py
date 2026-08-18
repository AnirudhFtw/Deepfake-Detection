"""
Shared helper for loading the video-level splits produced by
split_dataset.py, so every domain (spatial, frequency, temporal) trains
and evaluates on the exact same set of videos.
"""
import json
import os


def load_split_keys(splits_dir, split_name):
    """Returns a list of (label, video_name) tuples for the given split."""
    path = os.path.join(splits_dir, f"{split_name}.json")
    with open(path) as f:
        entries = json.load(f)
    return [(entry["label"], entry["video"]) for entry in entries]


def indices_for_keys(video_indices, keys):
    """
    video_indices: dict as returned by DeepfakeDataset.get_video_indices(),
    keyed by (class_name, video_name).
    """
    indices = []
    for key in keys:
        indices.extend(video_indices.get(tuple(key), []))
    return indices
