# %%
import os
import numpy as np
from glob import glob
import torch
import torchvision
import tifffile


import os

from Binary_V_Distance.data.download import mitolab_prefix, cem_prefix


# Define your augmentation
class RandomSpatialAugmentation(torch.nn.Module):
    def __init__(self, transforms, p=0.13):
        super().__init__()
        self.transform = torchvision.transforms.RandomApply(transforms, p=p)

    def __call__(self, image=None, mask=None):
        # Apply the same transformation to both image and mask
        seed = torch.randint(0, 10000, (1,)).item()  # generate random seed
        if image is not None:
            torch.manual_seed(seed)
            image = self.transform(image)
        if mask is not None:
            torch.manual_seed(seed)
            mask = self.transform(mask)
        return {"image": image, "mask": mask}


# %%
# Defines the hyperparameters
seeds = [3, 4, 13, 42, 1337]
lr = 1e-2
batch_size = 128
num_workers = 12
n_epochs = 300
steps = [25, 50, 75]
pred_weight = 1.0

use_scheduler = False

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

raw_transform = torchvision.transforms.RandomApply(
    [
        torchvision.transforms.GaussianBlur(3, sigma=(0.05, 0.13)),
        torchvision.transforms.ColorJitter(brightness=0.13, contrast=0.13),
    ]
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

        if self.gt_transform:
            mask = self.gt_transform(mask).float()

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

        if self.raw_transform:
            image = self.raw_transform(image)

        batch["image"] = image
        batch["mask"] = mask

        return batch


class CEMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        spatial_transform=None,
        raw_transform=None,
        size=(224, 224),
        **kwargs,
    ):
        super().__init__()
        self.root = root
        self.images = glob(os.path.join(root, "*.tiff"))
        self.spatial_transform = spatial_transform
        self.raw_transform = raw_transform
        self.size = np.array(size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = tifffile.imread(self.images[idx])

        image = torchvision.transforms.functional.to_tensor(image).float()

        if self.size is not None and any(self.size != image.shape[-2:]):
            image = torchvision.transforms.functional.resize(image, self.size)

        batch = {}
        if self.spatial_transform:
            augmented = self.spatial_transform(image=image)
            image = augmented["image"]

        if self.raw_transform:
            image = self.raw_transform(image)

        batch["image"] = image

        return batch


def get_dataloaders(
    batch_size=4,
    num_workers=16,
    spatial_transform=None,
    gt_transform=None,
    raw_transform=None,
    datasets=["train", "val", "test"],  # "unlabeled"
):
    print(f"Loading data from {mitolab_prefix}\n\tor {cem_prefix}")
    loaders = {}
    for dataset in datasets:
        print(f"Loading {dataset} dataset")
        if dataset == "train":
            loaders[dataset] = torch.utils.data.DataLoader(
                MitolabDataset(
                    os.path.join(mitolab_prefix, dataset),
                    spatial_transform=spatial_transform,
                    gt_transform=gt_transform,
                    raw_transform=raw_transform,
                ),
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )
        elif dataset == "unlabeled":
            loaders[dataset] = torch.utils.data.DataLoader(
                CEMDataset(
                    os.path.join(cem_prefix, "cem1.5M", "*"),
                    spatial_transform=spatial_transform,
                    raw_transform=raw_transform,
                ),
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )
        else:
            loaders[dataset] = torch.utils.data.DataLoader(
                MitolabDataset(os.path.join(mitolab_prefix, dataset)),
                batch_size=batch_size,
                num_workers=num_workers,
            )
    print("Data loaded")
    return loaders


def get_optimizer(models, lr):
    params = list()
    for model in models:
        params += list(model.parameters())
    return torch.optim.RAdam(params, lr=lr, decoupled_weight_decay=True)


class fake_scheduler:
    def step(self):
        pass


class multischeduler:
    def __init__(self, schedulers):
        self.schedulers = schedulers

    def step(self):
        for scheduler in self.schedulers:
            scheduler.step()


def get_scheduler(optimizer, steps):
    if not use_scheduler:
        return fake_scheduler()
    if len(optimizer) == 1:
        return torch.optim.lr_scheduler.MultiStepLR(optimizer[0], steps, gamma=0.1)
    else:
        schedulers = []
        for opt in optimizer:
            schedulers.append(
                torch.optim.lr_scheduler.MultiStepLR(opt, steps, gamma=0.1)
            )
        return multischeduler(schedulers)


def log_dict(writer, dict, step):
    for key, value in dict.items():
        writer.add_scalar(key, value, step)


def toggle_grad(model, requires_grad=None):
    for param in model.parameters():
        if requires_grad is None:
            param.requires_grad = not param.requires_grad
        else:
            param.requires_grad = requires_grad


# %%
