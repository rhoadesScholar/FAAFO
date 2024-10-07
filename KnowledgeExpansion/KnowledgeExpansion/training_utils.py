# %%
import os
import numpy as np
from glob import glob
import torch
import torchvision
import tifffile

from data.download import mitolab_prefix


# Define your augmentation
class RandomSpatialAugmentation:
    def __init__(self, transforms, p=0.13):
        self.transform = torchvision.transforms.RandomApply(transforms, p=p)

    def __call__(self, image, label):
        # Apply the same transformation to both image and label
        seed = torch.randint(0, 10000, (1,)).item()  # generate random seed
        torch.manual_seed(seed)
        image = self.transform(image)
        torch.manual_seed(seed)
        label = self.transform(label)
        return {"image": image, "label": label}


# %%
# Defines the hyperparameters
seeds = [3, 4, 13, 42, 69]
lr = 1e-4
batch_size = 16
n_epochs = 100
steps = [50, 75]

# Define augmentations for both the input image and the ground truth
spatial_augment = RandomSpatialAugmentation(
    [
        torchvision.transforms.RandomAffine(
            degrees=180,
            scale=(0.8, 1.2),
            shear=5,
        ),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomApply(
            [
                torchvision.transforms.RandomResizedCrop(
                    (224, 224), scale=(0.8, 1.2), interpolation=0
                )
            ],
            p=0.13,
        ),
    ],
    p=0.75,
)
"""
Example usage:
    augmented = spatial_augment(image=image, label=label)
    augmented_image = augmented["image"]
    augmented_label = augmented["label"]
"""

# Define augmentations for the ground truth only
augment_gt = torchvision.transforms.RandomApply(
    [
        torchvision.transforms.RandomAffine(
            degrees=10,
            scale=(0.9, 1.0),
            shear=5,
        ),
        torchvision.transforms.RandomErasing(p=0.13),
        torchvision.transforms.RandomResizedCrop(
            (224, 224), scale=(0.8, 1.2), interpolation=0
        ),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomPerspective(),
    ]
)
augment_raw = torchvision.transforms.RandomApply(
    [
        torchvision.transforms.GaussianBlur(3, sigma=(0.05, 0.13)),
        torchvision.transforms.ColorJitter(brightness=0.13, contrast=0.13),
    ]
)


# Define the command to launch subprocesses
# launch_command = "bsub -n 8 -gpu num=1 -q gpu_l4 -o logs/{seed}.out -e logs/{seed}.err python -m {script} {seed}"
launch_command = "python -m {script} {seed}"

get_optimizer = lambda models, lr: torch.optim.RAdam(
    [model.parameters() for model in models], lr=lr, decoupled_weight_decay=True
)

get_scheduler = lambda optimizer, steps: torch.optim.lr_scheduler.MultiStepLR(
    optimizer, steps, gamma=0.1
)


class MitolabDataset(torch.utils.data.Dataset):
    def __init__(
        self, root, spatial_transform=None, gt_transform=None, raw_transform=None
    ):
        self.root = root
        self.images = glob(os.path.join(root, "images", "*.tiff"))
        self.masks = glob(os.path.join(root, "masks", "*.tiff"))
        self.spatial_transform = spatial_transform
        self.gt_transform = gt_transform
        self.raw_transform = raw_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = tifffile.imread(self.images[idx])
        mask = tifffile.imread(self.masks[idx]) > 0

        if self.spatial_transform:
            augmented = self.spatial_transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        if self.gt_transform:
            mask = self.gt_transform(mask)
        if self.raw_transform:
            image = self.raw_transform(image)

        image = torchvision.transforms.functional.to_tensor(image)
        mask = torchvision.transforms.functional.to_tensor(mask)

        return image, mask


def get_data_loaders(batch_size=4, num_workers=16):
    train_dataset = MitolabDataset(
        os.path.join(mitolab_prefix, "train"),
        spatial_transform=spatial_augment,
        gt_transform=augment_gt,
        raw_transform=augment_raw,
    )
    val_dataset = MitolabDataset(os.path.join(mitolab_prefix, "val"))
    test_dataset = MitolabDataset(os.path.join(mitolab_prefix, "test"))

    pin_memory = torch.cuda.is_available()

    # Load the dataset
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return {"train": train_loader, "val": val_loader, "test": test_loader}


# %%
