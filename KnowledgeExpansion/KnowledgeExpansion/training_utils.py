# %%
import os
import numpy as np
from glob import glob
import torch
import torchvision
import tifffile

import os

from KnowledgeExpansion.data.download import mitolab_prefix


# Define your augmentation
class RandomSpatialAugmentation(torch.nn.Module):
    def __init__(self, transforms, p=0.13):
        super().__init__()
        self.transform = torchvision.transforms.RandomApply(transforms, p=p)

    def __call__(self, image, mask):
        # Apply the same transformation to both image and mask
        seed = torch.randint(0, 10000, (1,)).item()  # generate random seed
        torch.manual_seed(seed)
        image = self.transform(image)
        torch.manual_seed(seed)
        mask = self.transform(mask)
        return {"image": image, "mask": mask}


# %%
# Defines the hyperparameters
seeds = [3, 4, 13, 42, 69]
lr = 1e-3
batch_size = 128
num_workers = 12
n_epochs = 100
steps = [25, 50, 75]

# Define the command to launch subprocesses
launch_command = 'bsub -n 12 -gpu "num=1" -q gpu_h100 -o logs/{seed}.out -e logs/{seed}.err python {script} {seed}'
# launch_command = "python {script} {seed} &"

# Define augmentations for both the input image and the ground truth
spatial_transform = RandomSpatialAugmentation(
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

# Define augmentations for the ground truth only
gt_transform = torchvision.transforms.RandomApply(
    [
        torchvision.transforms.RandomAffine(
            degrees=10,
            scale=(0.9, 1.1),
            shear=3,
        ),
        torchvision.transforms.RandomErasing(p=0.13),
        torchvision.transforms.RandomResizedCrop((224, 224), scale=(0.9, 1.1)),
        torchvision.transforms.RandomHorizontalFlip(p=0.03),
        torchvision.transforms.RandomVerticalFlip(p=0.03),
        torchvision.transforms.RandomPerspective(distortion_scale=0.13, p=0.13),
        torchvision.transforms.GaussianBlur(11, sigma=(0.1, 2)),
    ],
    p=0.13,
)
raw_transform = torchvision.transforms.RandomApply(
    [
        torchvision.transforms.GaussianBlur(3, sigma=(0.05, 0.13)),
        torchvision.transforms.ColorJitter(brightness=0.13, contrast=0.13),
    ]
)

get_optimizer = lambda models, lr: torch.optim.RAdam(
    *[model.parameters() for model in models], lr=lr, decoupled_weight_decay=True
)

get_scheduler = lambda optimizer, steps: torch.optim.lr_scheduler.MultiStepLR(
    optimizer, steps, gamma=0.1
)


class MitolabDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        spatial_transform=None,
        gt_transform=None,
        raw_transform=None,
        size=(224, 224),
        **kwargs,
    ):
        super().__init__()
        self.root = root
        self.images = glob(os.path.join(root, "images", "*.tiff"))
        self.masks = glob(os.path.join(root, "masks", "*.tiff"))
        self.spatial_transform = spatial_transform
        self.gt_transform = gt_transform
        self.raw_transform = raw_transform
        self.size = np.array(size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = tifffile.imread(self.images[idx])
        mask = tifffile.imread(self.masks[idx]) > 0

        image = torchvision.transforms.functional.to_tensor(image).float()
        mask = torchvision.transforms.functional.to_tensor(mask).float()

        if self.size is not None and any(self.size != image.shape[-2:]):
            image = torchvision.transforms.functional.resize(image, self.size)
            mask = torchvision.transforms.functional.resize(
                mask, self.size, interpolation=0
            )

        batch = {}
        if self.spatial_transform:
            augmented = self.spatial_transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        if self.gt_transform:
            fake_mask = (self.gt_transform(mask) > 0.5).float()
            score = torch.nn.functional.binary_cross_entropy(fake_mask, mask)
            batch["fake_mask"] = fake_mask
            batch["score"] = score

        if self.raw_transform:
            image = self.raw_transform(image)

        batch["image"] = image
        batch["mask"] = mask

        return batch


def get_dataloaders(
    batch_size=4,
    num_workers=16,
    spatial_transform=None,
    gt_transform=None,
    raw_transform=None,
):
    print(f"Loading data from {mitolab_prefix}")
    train_dataset = MitolabDataset(
        os.path.join(mitolab_prefix, "train"),
        spatial_transform=spatial_transform,
        gt_transform=gt_transform,
        raw_transform=raw_transform,
    )
    val_dataset = MitolabDataset(os.path.join(mitolab_prefix, "val"))
    test_dataset = MitolabDataset(os.path.join(mitolab_prefix, "test"))

    # Load the dataset
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    print("Data loaded")
    return {"train": train_loader, "val": val_loader, "test": test_loader}


# %%
