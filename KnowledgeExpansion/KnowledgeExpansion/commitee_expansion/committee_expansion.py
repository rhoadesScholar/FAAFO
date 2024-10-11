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
    save_path = save_path + os.path.sep + "{model}_{seed}.pth"

    # Load the models
    student = (
        torch.load(
            save_path.format(model="student_joint", seed=seed), weights_only=False
        )
        .train()
        .to("cuda" if torch.cuda.is_available() else "cpu")
    )
    # Load all the teacher models
    teachers = []
    for teacher_seed in seeds:
        teachers.append(
            torch.load(
                save_path.format(model="teacher_joint", seed=teacher_seed),
                weights_only=False,
            )
            .eval()
            .to("cuda" if torch.cuda.is_available() else "cpu")
        )
    student.train()

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
    writer = SummaryWriter(f"logs/committee_expansion_{seed}")

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
            student_loss = torch.tensor(0.0, requires_grad=True).cuda()
            losses = []
            for teacher in teachers:
                loss = teacher(batch["image"], student_output).mean()
                losses.append(loss.item())
                student_loss += loss

            student_loss.backward()
            optimizer.step()
            train_bar.set_description(f"Loss: {student_loss.item()} ± {np.std(losses)}")
            writer.add_histogram(
                "Student loss", losses, epoch * len(loaders["unlabeled"]) + i
            )
            scalars = {
                "Predicted loss": student_loss.item(),
                "Loss std": np.std(losses),
            }
            log_dict(writer, scalars, epoch * len(loaders["unlabeled"]) + i)
        scheduler.step()

        torch.save(
            student, save_path.format(model="student_committee_expansion", seed=seed)
        )


if __name__ == "__main__":
    print("Starting committee knowledge expansion")
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
        expansion(seed)
    else:
        # Launch subprocesses for each seed
        for seed in seeds:
            print(f"Launching training for seed {seed}")
            os.system(launch_command.format(script=__file__, seed=seed))


# %%
