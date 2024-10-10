# %%
import numpy as np
import random
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

from tensorboardX import SummaryWriter
from tqdm import tqdm

import os
import sys

from KnowledgeExpansion.training_utils import (
    get_dataloaders,
    get_optimizer,
    get_scheduler,
    spatial_transform,
    raw_transform,
    launch_command,
    seeds,
    lr,
    batch_size,
    num_workers,
    n_epochs,
    steps,
    log_dict,
    cem_prefix,
)


# %%
# Define the baseline student training function
def expansion(seed: int):

    print(f"Knowledge expansion for student using teacher with seed {seed}")

    # Set the random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Save path for the models
    save_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "models", "checkpoints"
    )
    save_path = save_path + os.path.sep + "{model}_" + f"{seed}.pth"

    # Load the models
    student = torch.load(save_path.format(model="student_joint"), weights_only=False)
    teacher = torch.load(save_path.format(model="teacher_joint"), weights_only=False)
    student.train()
    teacher.eval()

    if torch.cuda.is_available():
        student = student.cuda()
        teacher = teacher.cuda()

    # Load the dataset
    loaders = get_dataloaders(
        batch_size,
        num_workers,
        spatial_transform,
        raw_transform,
        datasets=["unlabeled"],
    )

    # Define the optimizer and scheduler
    optimizer = get_optimizer([student], lr)
    scheduler = get_scheduler(optimizer, steps)

    # Make the tensorboard writer
    writer = SummaryWriter(f"logs/expansion_{seed}")

    # Train the student model
    epoch_bar = tqdm(range(n_epochs))
    for epoch in epoch_bar:
        train_bar = tqdm(loaders["unlabeled"])
        for i, batch in enumerate(train_bar):
            if torch.cuda.is_available():
                for key in batch:
                    batch[key] = batch[key].cuda()

            optimizer.zero_grad()

            # Student forward pass and loss calculation
            student_output = student(batch["image"])
            student_loss = teacher(batch["image"], student_output).mean()

            student_loss.backward()
            optimizer.step()
            train_bar.set_description(f"Loss: {student_loss.item()}")
            scalars = {"Predicted loss": student_loss.item()}
            log_dict(writer, scalars, epoch * len(loaders["unlabeled"]) + i)
        scheduler.step()

        # # Save the student model if it is the best one so far
        # if total_loss < best_val:
        #     if os.path.exists(save_path):
        #         os.remove(save_path)
        torch.save(student, save_path.format(model="student_expansion"))
        # best_val = total_loss


if __name__ == "__main__":
    print("Starting student knowledge expansion")
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
        expansion(seed)
    else:
        # Launch subprocesses for each seed
        for seed in seeds:
            print(f"Launching training for seed {seed}")
            # joint_train(seed)
            os.system(launch_command.format(script=__file__, seed=seed))


# %%
