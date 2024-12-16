# %%
import os
import sys
import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from Binary_V_Distance.training_utils import (
    get_dataloaders,
    num_workers,
)
from cellmap_data.transforms.targets import DistanceTransform, SignedDistanceTransform

from Binary_V_Distance.calibration import get_student_type, STUDENT_TYPE_COLORS
import torch

#%%
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load the dataset
loaders = get_dataloaders(1, num_workers, datasets=["test"])

# Get the student model checkpoint paths
student_paths = glob("models/checkpoints/*_42.pth")

models = {}
for student_path in student_paths:
    _, student_type = get_student_type(student_path, STUDENT_TYPE_COLORS.keys())
    if not student_type:
        print(f"Student type not found for {student_path}")
        continue
    print(f"Calculating calibration error for {student_path}")
    # Load the student model
    student = torch.load(student_path, weights_only=False)
    student.eval()
    student.to(device)
    student_name = (
        os.path.basename(student_path).split(".")[0].removeprefix("student_")
    )
    models[student_type] = student

# Iterate over the test dataset
with torch.no_grad():
    for batch in loaders["test"]:
        # Extract the image and mask from the batch
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        break

target_func = {"BCE": lambda x: x, "distance": DistanceTransform(), "signed_distance": SignedDistanceTransform()}
post_func = {"BCE": torch.sigmoid, "distance": lambda x: x, "signed_distance": lambda x: x}
# with thresholding
# post_func = {"BCE": lambda x: torch.sigmoid(x) > 0.5, "distance": lambda x: x > 1, "signed_distance": lambda x: x > 1}
fig, ax = plt.subplots(len(student_paths), 3, figsize=(5 * len(student_paths), 15))
for i, (student_type, student) in enumerate(models.items()):
    # Forward pass
    output = student(image)

    output = post_func[student_type](output)

    # Calculate the target
    target = target_func[student_type](mask)
    target = post_func[student_type](target)

    # Plot the image and the distance
    ax[i, 0].imshow(image[0, 0].detach().cpu().numpy(), cmap="gray")
    tax = ax[i, 1].imshow(target[0, 0].detach().cpu().numpy(), cmap="viridis")
    ax[i, 2].imshow(output[0, 0].detach().cpu().numpy(), cmap="viridis", clim=tax.get_clim())
    ax[i, 0].set_title(f"Input")
    ax[i, 1].set_title(f"Target: {student_type}")
    ax[i, 2].set_title(f"Output: {student_type}")
    plt.colorbar(ax[i, 1].imshow(target[0, 0].detach().cpu().numpy(), cmap="viridis"), ax=ax[i, 1])
    plt.colorbar(ax[i, 2].imshow(output[0, 0].detach().cpu().numpy(), cmap="viridis"), ax=ax[i, 2])
# %%
