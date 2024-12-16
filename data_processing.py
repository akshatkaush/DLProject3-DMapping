import torch
from torchvision import datasets, transforms
from pathlib import Path
from collections import defaultdict
import random
from torch.utils.data import DataLoader
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from PIL import Image


class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, num_nodes, overlap_pct=0.01, val_pct=0.2):
        self.path = Path(data_path)
        self.all_images = list(self.path.rglob("*.png"))
        self.overlap_pct = overlap_pct
        self.class_wise_dict = self.get_class_wise_dict()
        self.val_set = self.make_val_set(val_pct)
        classes_per_node = len(self.class_wise_dict) // num_nodes
        self.classes_for_node = {
            node: list(self.class_wise_dict.keys())[
                node * classes_per_node : (node + 1) * classes_per_node
            ]
            for node in range(num_nodes)
        }
        self.images_per_node = defaultdict(list)
        for node_idx in range(num_nodes):
            self.make_node_ds(node_idx)

    def label_func(self, x):
        return x.parent.name

    def get_class_wise_dict(self):
        class_wise_dict = defaultdict(list)
        for img in self.all_images:
            class_wise_dict[self.label_func(img)].append(img)
        return class_wise_dict

    def make_node_ds(self, node_idx):
        for cls in self.classes_for_node[node_idx]:
            self.images_per_node[node_idx].extend(self.class_wise_dict[cls])
        # get overlap_pct images from each other class
        overlap_images = []
        for cls in self.class_wise_dict.keys():
            if cls not in self.classes_for_node[node_idx]:
                overlap_images.extend(
                    random.sample(
                        self.class_wise_dict[cls],
                        int(len(self.class_wise_dict[cls]) * self.overlap_pct),
                    )
                )
        self.images_per_node[node_idx].extend(overlap_images)

    def make_val_set(self, val_pct):
        val_set = []
        for cls, imgs in self.class_wise_dict.items():
            val_imgs = random.sample(imgs, int(len(imgs) * val_pct))
            val_set.extend(val_imgs)
            # remove val_imgs from class_wise_dict
            for img in val_imgs:
                imgs.remove(img)

        return val_set

    def __len__(self, node_idx):
        return len(self.images_per_node[node_idx])

    def __getitem__(self, idx, node_idx):
        return self.images_per_node[node_idx][idx]


class NodeDataset(torch.utils.data.Dataset):
    def __init__(self, combined_dataset, node_idx, transform=None):
        self.combined_dataset = combined_dataset
        self.node_idx = node_idx
        self.transform = transform or transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return self.combined_dataset.__len__(self.node_idx)

    def __getitem__(self, idx):
        img_path = self.combined_dataset.__getitem__(idx, self.node_idx)
        # Load image using PIL
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)

        # Get label
        label = int(self.combined_dataset.label_func(img_path))
        return img, label


class ValDataset(torch.utils.data.Dataset):
    def __init__(self, combined_dataset, transform=None):
        self.combined_dataset = combined_dataset
        self.transform = transform or transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.combined_dataset.val_set)

    def __getitem__(self, idx):
        img_path = self.combined_dataset.val_set[idx]
        # Load image using PIL
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # Get label
        label = int(self.combined_dataset.label_func(img_path))
        return img, label
