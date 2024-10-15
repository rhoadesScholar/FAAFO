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

batch_size /= 2  # Divide the batch size by 2 so that all the teachers can fit in memory
batch_size = int(batch_size)


# %%
# Define the baseline student training function
def ensemble_expanded(seed: int):

    print(f"Knowledge expansion for student with seed {seed} using all teachers.")

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
        datasets=["val", "unlabeled"],
    )

    # Define the optimizer and scheduler
    student_criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = get_optimizer([student], lr)
    scheduler = get_scheduler(optimizer, steps)

    # Make the tensorboard writer
    writer = SummaryWriter(f"logs/ensemble_expanded_{seed}")

    # Train the student model
    epoch_bar = tqdm(range(n_epochs))
    for epoch in epoch_bar:
        student.train()
        train_bar = tqdm(loaders["unlabeled"])
        for i, batch in enumerate(train_bar):
            if torch.cuda.is_available():
                for key in batch:
                    batch[key] = batch[key].cuda()

            # Student forward pass and loss calculation
            optimizer.zero_grad()
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

        # Validation
        val_bar = tqdm(loaders["val"])
        student.eval()
        with torch.no_grad():
            total_student_loss = 0
            total_student_pred_loss = 0
            losses = []
            for i, batch in enumerate(val_bar):
                if torch.cuda.is_available():
                    for key in batch:
                        batch[key] = batch[key].cuda()
                student_output = student(batch["image"])
                student_loss = student_criterion(student_output, batch["mask"])
                total_student_loss += student_loss.item()

                student_pred_loss = torch.tensor(0.0).cuda()
                for teacher in teachers:
                    loss = teacher(batch["image"], student_output)
                    losses.append(loss.item())
                    student_pred_loss += loss.mean()
                total_student_pred_loss += student_pred_loss.item() / len(teachers)
            writer.add_histogram(
                "Val_Student_Pred_Loss",
                losses,
                (epoch + 1) * len(loaders["train"]) - 1,
            )
            total_student_loss /= len(loaders["val"])
            total_student_pred_loss /= len(loaders["val"])

            scalars = {
                "Val_Student_BCE": total_student_loss,
                "Val_Student_Pred_Loss": total_student_pred_loss,
            }
            log_dict(writer, scalars, (epoch + 1) * len(loaders["train"]) - 1)
            total_loss = (total_student_loss + total_student_pred_loss) / 2

        epoch_bar.set_description(f"Validation Loss: {total_loss}")

        torch.save(
            student, save_path.format(model="student_ensemble_expanded", seed=seed)
        )


if __name__ == "__main__":
    print("Starting committee knowledge expansion")
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
        ensemble_expanded(seed)
    else:
        # Launch subprocesses for each seed
        for seed in seeds:
            print(f"Launching training for seed {seed}")
            os.system(launch_command.format(script=__file__, seed=seed))


# %%
