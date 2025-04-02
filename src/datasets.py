import glob
import os
import xml.etree.ElementTree as ET
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, Subset


class CatDogDataset(Dataset):
    """
    Dataset code provided in assignment instructions.
    https://colab.research.google.com/drive/1oyRBec77YBynOWLmEPIXEzyO3R4BrIUF?usp=drive_link#scrollTo=6Vn-weRJRmrE
    """
    def __init__(
        self, img_dir: str, ann_dir: str, target_image_size: tuple[int, int], 
        transform: Optional[callable] = None, augmentations: Optional[list] = None
    ):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transform = transform
        self.augmentations = augmentations
        self.img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.ann_files = sorted(glob.glob(os.path.join(ann_dir, "*.xml")))
        self.label_map = {"cat": 0, "dog": 1}  # Label mapping
        self.target_image_size = target_image_size

    def parse_annotation(self, ann_path: str) -> tuple[int, int, list[dict]]:
        """
        Parses an annotation XML file and extracts object information.
        """
        tree = ET.parse(ann_path)
        root = tree.getroot()
        width = int(root.find("size/width").text)
        height = int(root.find("size/height").text)
        objects = []

        for obj in root.findall("object"):
            name = obj.find("name").text
            xmin = int(obj.find("bndbox/xmin").text)
            ymin = int(obj.find("bndbox/ymin").text)
            xmax = int(obj.find("bndbox/xmax").text)
            ymax = int(obj.find("bndbox/ymax").text)

            label = self.label_map.get(name, -1)  # Default to -1 if unknown label
            objects.append({"label": label, "bbox": [xmin, ymin, xmax, ymax]})

        return width, height, objects

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Loads an image and its corresponding annotation.
        """
        img_path = self.img_files[idx]
        ann_path = self.ann_files[idx]

        image = Image.open(img_path).convert("RGB")
        _, _, objects = self.parse_annotation(ann_path)

        bboxes = [obj['bbox'] for obj in objects[:1]]  # Take only first bbox
        labels = [obj['label'] for obj in objects[:1]]  # Take only first label

        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        if self.transform:
            image, bboxes = self.transform(image, bboxes)

        image = T.ToTensor()(image)

        return image, bboxes, labels


def prepare_datasets(
    img_dir: str, ann_dir: str, target_image_size: tuple[int, int],
    transform_train: Optional[callable], transform_test: Optional[callable]
) -> tuple[Subset, Subset, Subset]:
    """
    Prepares train, validation, and test datasets by splitting data stratified by labels.
    """
    train_dataset = CatDogDataset(img_dir, ann_dir,
                                  target_image_size, transform=transform_train)
    valid_test_dataset = CatDogDataset(img_dir, ann_dir, target_image_size,
                                       transform=transform_test)

    labels = np.array([
        int(valid_test_dataset[i][2]) if valid_test_dataset[i][2].shape[0] == 1
        else int(valid_test_dataset[i][2][0])
        for i in range(len(valid_test_dataset))
    ])

    train_val_indices = np.arange(3000)
    test_indices = np.arange(3000, len(valid_test_dataset))

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=42)
    train_idx, valid_idx = next(splitter.split(
        np.zeros(len(train_val_indices)), labels[train_val_indices]))

    train_dataset = Subset(train_dataset, train_val_indices[train_idx])
    valid_dataset = Subset(valid_test_dataset, train_val_indices[valid_idx])
    test_dataset = Subset(valid_test_dataset, test_indices)

    return train_dataset, valid_dataset, test_dataset
