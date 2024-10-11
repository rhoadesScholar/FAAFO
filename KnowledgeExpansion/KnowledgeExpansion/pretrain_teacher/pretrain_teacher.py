# %%
import numpy as np
import random
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

import os
import sys

from KnowledgeExpansion.training_utils import (
    get_dataloaders,
    get_optimizer,
    get_scheduler,
    spatial_transform,
    gt_transform,
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
# Define the pretraining function
def pretrain_teacher(seed: int):

    print(f"Pretraining teacher with seed {seed}")

    # Set the random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Load the teacher model
    from KnowledgeExpansion.models import teacher

    if torch.cuda.is_available():
        teacher = teacher.cuda()

    # Load the dataset
    loaders = get_dataloaders(
        batch_size, num_workers, spatial_transform, gt_transform, raw_transform
    )

    # Define the optimizer and scheduler
    optimizer = get_optimizer([teacher], lr)
    scheduler = get_scheduler(optimizer, steps)

    # Define the loss function
    criterion = lambda x, y: (x - y).abs().float().mean()

    # Make the tensorboard writer
    writer = SummaryWriter(f"logs/teacher_{seed}")

    # # Initiate best validation loss
    # best_val = np.inf

    # Save path for the teacher model
    save_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "models",
        "checkpoints",
        f"teacher_pretrained_{seed}.pth",
    )

    # Train the teacher model
    epoch_bar = tqdm(range(n_epochs))
    for epoch in epoch_bar:
        train_bar = tqdm(loaders["train"])
        teacher.train()
        for i, batch in enumerate(train_bar):
            if torch.cuda.is_available():
                for key in batch:
                    batch[key] = batch[key].cuda()

            optimizer.zero_grad()

            # Forward pass and loss calculation for the augmented mask
            if pred_weight > 0:
                output = teacher(batch["image"], batch["fake_mask"])
                teacher_loss_pred = criterion(output, batch["score"])
            else:
                teacher_loss_pred = torch.tensor(0.0)

            # Forward pass and loss calculation for the original mask
            if pred_weight < 1:
                output = teacher(batch["image"], batch["mask"])
                teacher_loss_gt = criterion(output, 0.0)
            else:
                teacher_loss_gt = torch.tensor(0.0)

            loss = pred_weight * teacher_loss_pred + (1 - pred_weight) * teacher_loss_gt
            loss.backward()
            optimizer.step()
            train_bar.set_description(f"Loss: {loss.item()}")
            scalars = {
                "Teacher_Pred": teacher_loss_pred.item(),
                "Teacher_GT": teacher_loss_gt.item(),
            }
            log_dict(writer, scalars, epoch * len(loaders["train"]) + i)
        scheduler.step()

        val_bar = tqdm(loaders["val"])
        teacher.eval()
        with torch.no_grad():
            total_loss = 0
            for i, batch in enumerate(val_bar):
                if torch.cuda.is_available():
                    for key in batch:
                        batch[key] = batch[key].cuda()
                output = teacher(batch["image"], batch["mask"])
                loss = criterion(output, 0.0)
                total_loss += loss.item()
            total_loss /= len(loaders["val"])
            writer.add_scalar(
                "Validation Loss", total_loss, (epoch + 1) * len(loaders["train"]) - 1
            )

        epoch_bar.set_description(f"Validation Loss: {total_loss}")

        # # Save the teacher model if it is the best one so far
        # if total_loss < best_val:
        #     if os.path.exists(save_path):
        #         os.remove(save_path)
        torch.save(teacher, save_path)
        # best_val = total_loss


if __name__ == "__main__":
    print("Starting pretraining")
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
        pretrain_teacher(seed)
    else:
        # Launch subprocesses for each seed
        for seed in seeds:
            print(f"Launching training for seed {seed}")
            # pretrain_teacher(seed)
            os.system(launch_command.format(script=__file__, seed=seed))


# %%
