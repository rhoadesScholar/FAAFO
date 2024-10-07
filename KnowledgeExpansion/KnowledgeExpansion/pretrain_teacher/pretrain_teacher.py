# %%
import torch
import numpy as np
import random
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from KnowledgeExpansion.training_utils import (
    get_data_loaders,
    get_optimizer,
    get_scheduler,
    spatial_augment,
    augment_gt,
    launch_command,
    seeds,
    lr,
    batch_size,
    n_epochs,
    steps,
)


# %%
# Launch subprocesses for each seed
for seed in seeds:
    os.system(launch_command.format(script=__file__, seed=seed))


# %%
# Define the pretraining function
def pretrain_teacher(seed: int):
    # Set the random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Load the teacher model
    from KnowledgeExpansion.models import teacher

    if torch.cuda.is_available():
        teacher = teacher.cuda()

    # Load the dataset
    train_loader, val_loader = get_data_loaders(batch_size, spatial_augment, augment_gt)

    # Define the optimizer and scheduler
    optimizer = get_optimizer([teacher], lr)
    scheduler = get_scheduler(optimizer, steps)

    # Define the loss function
    criterion = torch.nn.MSELoss()

    # Train the teacher model
    epoch_bar = tqdm(range(n_epochs))
    for epoch in epoch_bar:
        train_bar = tqdm(train_loader)
        teacher.train()
        for i, (image, labels, score) in enumerate(train_bar):
            optimizer.zero_grad()
            output = teacher(torch.stack([image, labels], dim=1))
            loss = criterion(output, score)
            loss.backward()
            optimizer.step()
            train_bar.set_description(f"Loss: {loss.item()}")
        scheduler.step()

        val_bar = tqdm(val_loader)
        teacher.eval()
        with torch.no_grad():
            total_loss = 0
            for i, (image, labels, score) in enumerate(val_bar):
                output = teacher(torch.stack([image, labels], dim=1))
                loss = criterion(output, score)
                total_loss += loss.item()
            val_bar.set_description(f"Validation Loss: {total_loss / len(val_loader)}")

        epoch_bar.set_description(f"Validation Loss: {total_loss / len(val_loader)}")

    # Save the teacher model
    torch.save(teacher, f"teacher_{seed}.pt")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
    else:
        seed = seeds[0]

    pretrain_teacher(seed)
