import os
import random
import subprocess
from glob import glob

# Set datasplit percentages and seed
train_percentage = 0.8
val_percentage = 0.15
test_percentage = 0.05

seed = 42

mitolab_prefix = os.path.join(os.path.dirname(__file__), "CEM-MitoLab")

assert (
    train_percentage + val_percentage + test_percentage == 1.0
), "The sum of the train, validation, and test percentages must be equal to 1."


def download_empiar_dataset(dataset_id, output_dir):
    """
    Downloads the dataset from EMPIAR using wget and saves it to the specified directory.

    Args:
        dataset_id (str): The dataset ID from EMPIAR.
        output_dir (str): The directory where the dataset will be saved.
    """
    # Check if wget is installed
    if subprocess.call(["which", "wget"]) != 0:
        raise EnvironmentError(
            "wget is not installed. Please install wget to use this script."
        )

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Construct the wget command for downloading the dataset
    url = f"ftp://ftp.ebi.ac.uk/empiar/world_availability/{dataset_id}"
    command = f"wget -r -nH --cut-dirs=3 --no-parent --reject='index.html*' -P {output_dir} {url}"

    # Run the wget command
    try:
        print(f"Downloading dataset {dataset_id} to {output_dir}...")
        subprocess.run(command, shell=True, check=True)
        print("Download completed.")
    except subprocess.CalledProcessError as e:
        print(f"Error during download: {e}")


if __name__ == "__main__":

    # Download CEM1.5M dataset
    download_empiar_dataset(
        dataset_id="11035",
        output_dir=os.path.join(os.path.dirname(__file__), "CEM1500k_unlabelled"),
    )

    # Unzip the downloaded dataset
    subprocess.run(
        f"unzip {os.path.join(os.path.dirname(__file__), 'CEM1500k_unlabelled', 'data', 'cem1.5m.zip')} -d {os.path.join(os.path.dirname(__file__), 'CEM1500k_unlabelled')}",
        shell=True,
    )

    # Download CEM-MitoLab dataset
    download_empiar_dataset(
        dataset_id="11037",
        output_dir=mitolab_prefix,
    )

    # Unzip the downloaded dataset
    subprocess.run(
        f"unzip {os.path.join(mitolab_prefix, 'data', 'cem_mitolab.zip')} -d {mitolab_prefix}",
        shell=True,
    )

    # Split the CEM-MitoLab dataset into training (80%), validation (15%), and test (5%) sets
    # Create directories for each dataset type
    data_paths = {}
    for dataset_type in ["train", "val", "test"]:
        data_paths[dataset_type] = {}
        for data_type in ["images", "masks"]:
            data_paths[dataset_type][data_type] = []
            os.makedirs(
                os.path.join(mitolab_prefix, dataset_type, data_type),
                exist_ok=True,
            )

    # Divide the dataset into training, validation, and test sets

    # First get the entire list of images and masks
    images = glob(os.path.join(mitolab_prefix, "cem_mitolab", "*", "images", "*.tiff"))
    masks = glob(os.path.join(mitolab_prefix, "cem_mitolab", "*", "masks", "*.tiff"))
    assert all(
        [
            m.split(os.path.sep)[-1] == i.split(os.path.sep)[-1]
            for m, i in zip(masks, images)
        ]
    ), "Images and masks do not match entirely. Make sure the images and masks are named the same way, and there are no extra or missing files in the data folders. If necessary, redownload or re-unzip the dataset."

    # Get random indices for each dataset type
    n_images = len(images)
    n_train = int(train_percentage * n_images)
    n_val = int(val_percentage * n_images)
    n_test = n_images - (n_train + n_val)

    # Set the seed for reproducibility
    random.seed(seed)

    indices = list(range(n_images))
    random.shuffle(indices)

    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]
    assert len(train_indices) == n_train
    assert len(val_indices) == n_val
    assert len(test_indices) == n_test

    # Make simlinks to the images and masks in the respective directories
    for dataset_type, indices in zip(
        ["train", "val", "test"], [train_indices, val_indices, test_indices]
    ):
        try:
            image_prefix = os.path.join(mitolab_prefix, dataset_type, "images")
            mask_prefix = os.path.join(mitolab_prefix, dataset_type, "masks")
            os.makedirs(image_prefix)
            os.makedirs(mask_prefix)
        except FileExistsError:
            # Remove existing simlinks
            print(f"Removing existing simlinks in {image_prefix} and {mask_prefix}")
            for f in glob(os.path.join(image_prefix, "*")):
                os.remove(f)
            for f in glob(os.path.join(mask_prefix, "*")):
                os.remove(f)
        for i in indices:
            os.symlink(
                images[i],
                os.path.join(image_prefix, images[i].split(os.path.sep)[-1]),
            )
            os.symlink(
                masks[i],
                os.path.join(mask_prefix, masks[i].split(os.path.sep)[-1]),
            )
