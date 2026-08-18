import os
import numpy as np
import torch
from torch.utils.data import Dataset


class DeepfakeDataset(Dataset):
    """
    Dataset structure:

    root_dir/
    ├── real/
    │   ├── video1/
    │   │    frame_00000.npy
    │   │    frame_00001.npy
    │   │    ...
    │   ├── video2/
    │   └── ...
    │
    └── fake/
        ├── video1/
        ├── video2/
        └── ...
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.samples = []
        self.video_indices = {}

        class_map = {
            "real": 0,
            "fake": 1
        }

        for class_name, label in class_map.items():

            class_path = os.path.join(root_dir, class_name)

            if not os.path.exists(class_path):
                continue

            videos = sorted(os.listdir(class_path))

            for video in videos:

                video_path = os.path.join(class_path, video)

                if not os.path.isdir(video_path):
                    continue

                key = (class_name, video)
                self.video_indices[key] = []

                files = sorted(os.listdir(video_path))

                for file in files:

                    if not file.endswith(".npy"):
                        continue

                    sample = {
                        "path": os.path.join(video_path, file),
                        "label": label,
                        "class": class_name,
                        "video": video
                    }

                    idx = len(self.samples)

                    self.samples.append(sample)
                    self.video_indices[key].append(idx)

        print(f"\nLoaded {len(self.samples)} samples from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        image = np.load(sample["path"]).astype(np.float32)

        # HWC -> CHW
        image = np.transpose(image, (2, 0, 1))

        image = torch.from_numpy(image)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(sample["label"], dtype=torch.long)

        return image, label

    def get_video_indices(self):
        """
        Returns:
            {
                ('real','video1'): [0,1,2,...],
                ('fake','video7'): [203,204,...]
            }
        """
        return self.video_indices

    def get_num_videos(self):
        return len(self.video_indices)

    def get_num_frames(self):
        return len(self.samples)

    def summary(self):

        real_frames = 0
        fake_frames = 0

        real_videos = 0
        fake_videos = 0

        for (cls, _), indices in self.video_indices.items():

            if cls == "real":
                real_videos += 1
                real_frames += len(indices)
            else:
                fake_videos += 1
                fake_frames += len(indices)

        print("\n========== Dataset Summary ==========")
        print(f"Real Videos : {real_videos}")
        print(f"Fake Videos : {fake_videos}")
        print(f"Real Frames : {real_frames}")
        print(f"Fake Frames : {fake_frames}")
        print(f"Total Frames: {len(self.samples)}")
        print("=====================================\n")