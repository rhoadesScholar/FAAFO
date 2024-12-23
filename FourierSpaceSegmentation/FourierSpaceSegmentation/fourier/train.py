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

from FourierSpaceSegmentation.training_utils import (
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
)


# %%
# Define the training function
def train(seed: int):

    print(f"Training Fourier Space Segmentation model with seed {seed}")

    # Set the random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Load the model
    from FourierSpaceSegmentation.models import FourierSpaceModel

    model = FourierSpaceModel()

    if torch.cuda.is_available():
        model = model.cuda()

    # Load the dataset
    loaders = get_dataloaders(
        batch_size, num_workers, spatial_transform, raw_transform=raw_transform
    )

    # Define the optimizer and scheduler
    optimizer = get_optimizer([model], lr)
    scheduler = get_scheduler(optimizer, steps)

    # Define the loss function
    criterion = torch.nn.MSELoss()

    # Make the tensorboard writer
    writer = SummaryWriter(f"logs/FSS_{seed}")

    # # Initiate best validation loss
    # best_val = np.inf

    # Save path for the model model
    save_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "models",
        "checkpoints",
        f"FSS_{seed}.pth",
    )

    # Train the model model
    epoch_bar = tqdm(range(n_epochs))
    for epoch in epoch_bar:
        train_bar = tqdm(loaders["train"])
        model.train()
        for i, batch in enumerate(train_bar):
            if torch.cuda.is_available():
                for key in batch:
                    batch[key] = batch[key].cuda()

            optimizer.zero_grad()

            # Forward pass and loss calculation
            # output = model(torch.fft.fft2(batch["image"]))
            output = model(torch.fft.fftshift(torch.fft.fft2(batch["image"])))
            # target = torch.fft.fft2(batch["mask"])
            target = torch.fft.fftshift(torch.fft.fft2(batch["mask"]))
            target = torch.concatenate([torch.real(target), torch.imag(target)], dim=1)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()
            train_bar.set_description(f"Loss: {loss.item()}")
            writer.add_scalar("Loss", loss.item(), epoch * len(loaders["train"]) + i)
        scheduler.step()

        val_bar = tqdm(loaders["val"])
        model.eval()
        with torch.no_grad():
            total_loss = 0
            for i, batch in enumerate(val_bar):
                if torch.cuda.is_available():
                    for key in batch:
                        batch[key] = batch[key].cuda()
                # output = model(torch.fft.fft2(batch["image"]))
                output = model(torch.fft.fftshift(torch.fft.fft2(batch["image"])))
                # target = torch.fft.fft2(batch["mask"])
                target = torch.fft.fftshift(torch.fft.fft2(batch["mask"]))
                target = torch.concatenate(
                    [torch.real(target), torch.imag(target)], dim=1
                )
                loss = criterion(output, target)
                total_loss += loss.item()
            total_loss /= len(loaders["val"])
            writer.add_scalar(
                "Validation Loss", total_loss, (epoch + 1) * len(loaders["train"]) - 1
            )
            writer.add_images(
                "Input",
                batch["image"].cpu().numpy(),
                (epoch + 1) * len(loaders["train"]) - 1,
            )
            writer.add_images(
                "Output - real",
                model.render(output).detach().cpu().numpy().real,
                (epoch + 1) * len(loaders["train"]) - 1,
            )
            writer.add_images(
                "Output - absolute",
                model.render(output).detach().cpu().abs().numpy(),
                (epoch + 1) * len(loaders["train"]) - 1,
            )
            writer.add_images(
                "GT",
                batch["mask"].cpu().numpy(),
                w(epoch + 1) * len(loaders["train"]) - 1,
            )
        epoch_bar.set_description(f"Validation Loss: {total_loss}")

        # # Save the model model if it is the best one so far
        # if total_loss < best_val:
        #     if os.path.exists(save_path):
        #         os.remove(save_path)
        torch.save(model, save_path)
        # best_val = total_loss


if __name__ == "__main__":
    print("Starting FSS training")
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
        train(seed)
    else:
        # Launch subprocesses for each seed
        for seed in seeds:
            print(f"Launching FSS training for seed {seed}")
            os.system(launch_command.format(script=__file__, seed=seed))


# %%
