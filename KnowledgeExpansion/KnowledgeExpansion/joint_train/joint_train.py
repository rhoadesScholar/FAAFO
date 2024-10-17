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
    pred_weight,
    log_dict,
)


# %%
# Define the baseline student training function
def joint_train(seed: int):

    print(f"Joint training student and teacher with seed {seed}")

    # Set the random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Save path for the models
    save_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "models", "checkpoints"
    )
    save_path = save_path + os.path.sep + "{model}_" + f"{seed}.pth"

    # Load the student model
    from KnowledgeExpansion.models import student

    teacher = torch.load(
        save_path.format(model="teacher_pretrained"), weights_only=False
    )

    if torch.cuda.is_available():
        student = student.cuda()
        teacher = teacher.cuda()

    # Load the dataset
    loaders = get_dataloaders(
        batch_size, num_workers, spatial_transform, raw_transform=raw_transform
    )

    # Define the optimizer and scheduler
    optimizer = get_optimizer([student, teacher], lr)
    scheduler = get_scheduler(optimizer, steps)

    # Define the loss function
    student_criterion = torch.nn.BCEWithLogitsLoss()
    teacher_criterion = lambda x, y: (x - y).abs().float().mean()

    # Make the tensorboard writer
    writer = SummaryWriter(f"logs/joint_{seed}")

    # Train the student model
    epoch_bar = tqdm(range(n_epochs))
    for epoch in epoch_bar:
        train_bar = tqdm(loaders["train"])
        student.train()
        teacher.train()
        for i, batch in enumerate(train_bar):
            if torch.cuda.is_available():
                for key in batch:
                    batch[key] = batch[key].cuda()

            optimizer.zero_grad()

            # Student forward pass and loss calculation
            student_output = student(batch["image"])
            student_loss = student_criterion(student_output, batch["mask"])

            # Teacher Forward pass and loss calculation for the augmented mask
            if pred_weight > 0:
                teacher_output = teacher(batch["image"], student_output)
                teacher_loss_pred = teacher_criterion(teacher_output, student_loss)
            else:
                teacher_loss_pred = torch.tensor(0.0)

            # Forward pass and loss calculation for the original mask
            if pred_weight < 1:
                teacher_output = teacher(batch["image"], batch["mask"])
                teacher_loss_gt = teacher_criterion(teacher_output, 0.0)
            else:
                teacher_loss_gt = torch.tensor(0.0)

            loss = (
                student_loss
                + pred_weight * teacher_loss_pred
                + (1 - pred_weight) * teacher_loss_gt
            )
            loss.backward()
            optimizer.step()
            train_bar.set_description(
                f"Student loss: {student_loss.item()}, Teacher loss: {teacher_loss_pred.item()}"
            )
            scalars = {
                "Student": student_loss.item(),
                "Teacher_Pred": teacher_loss_pred.item(),
                "Teacher_GT": teacher_loss_gt.item(),
            }
            log_dict(writer, scalars, epoch * len(loaders["train"]) + i)
        scheduler.step()

        val_bar = tqdm(loaders["val"])
        student.eval()
        teacher.eval()
        with torch.no_grad():
            total_student_loss = 0
            total_teacher_loss_pred = 0
            total_teacher_loss_gt = 0
            for i, batch in enumerate(val_bar):
                if torch.cuda.is_available():
                    for key in batch:
                        batch[key] = batch[key].cuda()
                student_output = student(batch["image"])
                student_loss = student_criterion(student_output, batch["mask"])
                total_student_loss += student_loss.item()

                teacher_output = teacher(batch["image"], student_output)
                teacher_loss_pred = teacher_criterion(teacher_output, student_loss)
                total_teacher_loss_pred += teacher_loss_pred.item()

                teacher_output = teacher(batch["image"], batch["mask"])
                teacher_loss_gt = teacher_criterion(teacher_output, 0.0)
                total_teacher_loss_gt += teacher_loss_gt.item()

            total_student_loss /= len(loaders["val"])
            total_teacher_loss_pred /= len(loaders["val"])
            total_teacher_loss_gt /= len(loaders["val"])

            scalars = {
                "Val_Student": total_student_loss,
                "Val_Teacher_Pred": total_teacher_loss_pred,
                "Val_Teacher_GT": total_teacher_loss_gt,
            }
            total_loss = (
                total_student_loss + total_teacher_loss_pred + total_teacher_loss_gt
            )
            log_dict(writer, scalars, (epoch + 1) * len(loaders["train"]) - 1)

        epoch_bar.set_description(f"Validation Loss: {total_loss}")

        # # Save the student model if it is the best one so far
        # if total_loss < best_val:
        #     if os.path.exists(save_path):
        #         os.remove(save_path)
        torch.save(student, save_path.format(model="student_joint"))
        torch.save(teacher, save_path.format(model="teacher_joint"))
        # best_val = total_loss


if __name__ == "__main__":
    print("Starting joint training")
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
        joint_train(seed)
    else:
        # Launch subprocesses for each seed
        for seed in seeds:
            print(f"Launching training for seed {seed}")
            # joint_train(seed)
            os.system(launch_command.format(script=__file__, seed=seed))


# %%
