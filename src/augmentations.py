import random
from typing import Optional

import torch
import torchvision.transforms.functional as F
from PIL import Image


class RandomScale:
    """
    Apply random scaling to an image and its bounding boxes.
    """
    def __init__(self, scale_range: tuple[float, float] = (0.8, 1.2)):
        self.scale_range = scale_range

    def __call__(self, img: Image.Image, bboxs: torch.Tensor
                 ) -> tuple[Image.Image, torch.Tensor]:
        w, h = img.size
        scale_factor = random.uniform(*self.scale_range)
        center_x, center_y = w / 2, h / 2

        T = torch.tensor([
            [scale_factor, 0, (1 - scale_factor) * center_x],
            [0, scale_factor, (1 - scale_factor) * center_y],
            [0, 0, 1]
        ], dtype=torch.float)

        img = F.affine(img, angle=0, translate=(0, 0), scale=scale_factor,
                       shear=0, center=(center_x, center_y))

        N = bboxs.shape[0]
        ones = torch.ones((N, 1), dtype=bboxs.dtype, device=bboxs.device)

        tl = torch.cat([bboxs[:, 0:1], bboxs[:, 1:2], ones], dim=1)  # (N, 3)
        br = torch.cat([bboxs[:, 2:3], bboxs[:, 3:4], ones], dim=1)  # (N, 3)

        tl_transformed = (T @ tl.T).T  # (N, 3)
        br_transformed = (T @ br.T).T  # (N, 3)

        new_bboxs = torch.cat([
            tl_transformed[:, 0:1],
            tl_transformed[:, 1:2],
            br_transformed[:, 0:1],
            br_transformed[:, 1:2]
        ], dim=1)

        new_bboxs[:, [0, 2]] = torch.clamp(new_bboxs[:, [0, 2]], 0, w)
        new_bboxs[:, [1, 3]] = torch.clamp(new_bboxs[:, [1, 3]], 0, h)

        return img, new_bboxs


class RandomTranslate:
    """
    Apply random translation to an image and its bounding boxes.
    """
    def __init__(self, translate_range: tuple[float, float] = (-0.2, 0.2)):
        self.translate_range = translate_range

    def __call__(self, img: Image.Image, bboxs: torch.Tensor
                 ) -> tuple[Image.Image, torch.Tensor]:
        w, h = img.size
        tx = int(random.uniform(*self.translate_range) * w)
        ty = int(random.uniform(*self.translate_range) * h)

        img = F.affine(img, angle=0, translate=(tx, ty), scale=1.0, shear=0)

        bboxs[:, [0, 2]] += tx
        bboxs[:, [1, 3]] += ty

        bboxs[:, [0, 2]] = torch.clamp(bboxs[:, [0, 2]], 0, w)
        bboxs[:, [1, 3]] = torch.clamp(bboxs[:, [1, 3]], 0, h)

        return img, bboxs


class RandomBrightness:
    """
    Apply random brightness adjustment to an image.
    """
    def __init__(self, exposure_factor: float = 1.5):
        self.exposure_factor = exposure_factor

    def __call__(self, img: Image.Image, bboxs: torch.Tensor
                 ) -> tuple[Image.Image, torch.Tensor]:
        img = F.adjust_brightness(img, random.uniform(1 / self.exposure_factor,
                                                      self.exposure_factor))
        return img, bboxs


class RandomSaturation:
    """
    Apply random saturation adjustment to an image.
    """
    def __init__(self, saturation_factor: float = 1.5):
        self.saturation_factor = saturation_factor

    def __call__(self, img: Image.Image, bboxs: torch.Tensor
                 ) -> tuple[Image.Image, torch.Tensor]:
        img = F.adjust_saturation(
            img, random.uniform(1 / self.saturation_factor,
                                self.saturation_factor))
        return img, bboxs


class Resize:
    """
    Resize an image and adjust its bounding boxes accordingly.
    """
    def __init__(self, target_size: tuple[int, int],
                 interpolation: int = Image.BILINEAR):
        self.target_size = target_size
        self.interpolation = interpolation

    def __call__(self, img: Image.Image, bboxs: torch.Tensor
                 ) -> tuple[Image.Image, torch.Tensor]:
        orig_w, orig_h = img.size
        new_w, new_h = self.target_size

        scale_x = new_w / orig_w
        scale_y = new_h / orig_h

        img = img.resize((new_w, new_h), self.interpolation)

        bboxs[:, [0, 2]] *= scale_x
        bboxs[:, [1, 3]] *= scale_y

        return img, bboxs


class Augmentations:
    """
    Apply a series of augmentations to an image and its bounding boxes.
    """
    def __init__(self, transforms: Optional[list] = None):
        self.transforms = transforms if transforms is not None else []

    def __call__(self, img: Image.Image, bboxs: torch.Tensor
                 ) -> tuple[Image.Image, torch.Tensor]:
        for transform in self.transforms:
            img, bboxs = transform(img, bboxs)
        return img, bboxs
