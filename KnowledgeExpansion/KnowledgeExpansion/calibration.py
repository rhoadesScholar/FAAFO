# %%
import os
import sys
import json
import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from KnowledgeExpansion.training_utils import (
    get_dataloaders,
    batch_size,
    num_workers,
    launch_command,
)
import torch


def accuracy(y_true, y_pred):
    """
    Compute the accuracy of the predicted segmentation mask.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred : array-like of shape (n_samples,)
        Predicted probabilities (after softmax or sigmoid).

    Returns
    -------
    float
        Accuracy value.
    """
    y_true = np.array(y_true).astype(bool)
    y_pred = np.array(y_pred) > 0.5
    accuracy = (y_true == y_pred).mean()
    return accuracy


def iou(y_true, y_pred):
    """
    Compute the Intersection over Union (IoU) of the predicted segmentation mask.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred : array-like of shape (n_samples,)
        Predicted probabilities (after softmax or sigmoid).

    Returns
    -------
    float
        IoU value.
    """
    y_true = np.array(y_true).astype(bool)
    y_pred = np.array(y_pred) > 0.5
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    iou = intersection / union
    return iou


def calibration_error(y_true, y_pred, num_bins=15):
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE) and return the calibration curve

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred : array-like of shape (n_samples,)
        Predicted probabilities (after softmax or sigmoid).
    num_bins : int, optional (default=15)
        Number of bins.

    Returns
    -------
    float
        ECE value.
    float
        MCE value.
    list
        List of accuracies.
    list
        List of confidences.
    """
    bins = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]

    ece = 0.0
    mce = 0.0
    accs = []
    confs = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred >= bin_lower) & (y_pred < bin_upper)
        proportion_in_bin = in_bin.mean()
        if proportion_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_pred[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * proportion_in_bin
            mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
            accs.append(accuracy_in_bin)
            confs.append(avg_confidence_in_bin)

    return ece, float(mce), accs, confs


def plot_calibration_curves(accs_dict, confs_dict, ece_dict, mce_dict):
    """
    Plot the calibration curves of the student models.

    Parameters
    ----------
    accs_dict : dict
        Dictionary containing the accuracies of the student models.
    confs_dict : dict
        Dictionary containing the confidences of the student models.
    ece_dict : dict
        Dictionary containing the Expected Calibration Error (ECE) of the student models.
    mce_dict : dict
        Dictionary containing the Maximum Calibration Error (MCE) of the student models.

    Returns
    -------
    matplotlib.pyplot.Figure
        The calibration curves.
    """
    student_type_colors = {
        "baseline": "blue",
        # "joint": "green",
        "split": "purple",
        # "expansion": "red",
        "ensemble_expanded": "orange",
        "ensemble_alternating": "cyan",
    }
    num_types = len(student_type_colors)
    # num_students = len(accs_dict) // num_types + 1
    num_students = 5
    fig, axes = plt.subplots(
        nrows=num_types,
        ncols=num_students,
        figsize=(5 * num_students, 5 * num_types),
        sharex=True,
        sharey=True,
    )
    avg_accs = {k: [] for k in student_type_colors.keys()}
    avg_confs = {k: [] for k in student_type_colors.keys()}
    avg_ece = {k: [] for k in student_type_colors.keys()}
    avg_mce = {k: [] for k in student_type_colors.keys()}
    student_inds = {k: 0 for k in student_type_colors.keys()}
    # Plot the calibration curves
    for student_name, accs in accs_dict.items():
        confs = confs_dict[student_name]
        found = False
        for i, (student_type, color) in enumerate(student_type_colors.items()):
            if student_type in student_name:
                avg_accs[student_type].append(accs)
                avg_confs[student_type].append(confs)
                avg_ece[student_type].append(ece_dict[student_name])
                avg_mce[student_type].append(mce_dict[student_name])
                found = True
                break
        if not found:
            raise ValueError(f"Student type not found in {student_name}")
        j = student_inds[student_type]
        # Add ECE and MCE values to the legend
        result_string = f"{student_name}:\nECE = {ece_dict[student_name]:.4f},\nMCE = {mce_dict[student_name]:.4f}"
        axes[i, j].bar(confs, accs, width=0.1, label=result_string, color=color)
        axes[i, j].plot([0, 1], [0, 1], linestyle="--", color="black")
        axes[i, j].set_xlabel("Confidence")
        axes[i, j].set_ylabel("Accuracy")
        axes[i, j].set_title(f"{student_name}")
        axes[i, j].legend(loc="upper left")
        student_inds[student_type] += 1

    # Now plot the averages
    for j, (student_type, color) in enumerate(student_type_colors.items()):
        accs_std = np.std(avg_accs[student_type], axis=0)
        avg_accs[student_type] = np.mean(avg_accs[student_type], axis=0)
        confs_std = np.std(avg_confs[student_type], axis=0)
        avg_confs[student_type] = np.mean(avg_confs[student_type], axis=0)
        std_ece = np.std(avg_ece[student_type])
        avg_ece[student_type] = np.mean(avg_ece[student_type])
        std_mce = np.std(avg_mce[student_type])
        avg_mce[student_type] = np.mean(avg_mce[student_type])
        # Add ECE and MCE values to the legend
        result_string = f"{student_type} Avg:\nECE = {avg_ece[student_type]:.4f} ± {std_ece},\nMCE = {avg_mce[student_type]:.4f} ± {std_mce}"
        axes[j, -1].bar(
            avg_confs[student_type],
            avg_accs[student_type],
            yerr=accs_std,
            width=0.1,
            label=result_string,
            color=color,
            capsize=5,
        )
        axes[j, -1].plot([0, 1], [0, 1], linestyle="--", color="black")
        axes[j, -1].set_xlabel("Confidence")
        axes[j, -1].set_ylabel("Accuracy")
        axes[j, -1].set_title(f"{student_type} Average")
        axes[j, -1].legend(loc="upper left")
    plt.show()

    return plt.gcf()


def main():
    """
    Calculate the Expected Calibration Error (ECE) and the Maximum Calibration Error (MCE) of the student models for each condition.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Load the dataset
    loaders = get_dataloaders(batch_size, num_workers, datasets=["test"])

    # Get the student model checkpoint paths
    student_paths = glob("models/checkpoints/student_*.pth")

    # Initialize the ECE, MCE, accs, and confs dictionaries
    ece_dict = {}
    mce_dict = {}
    accs_dict = {}
    confs_dict = {}
    ious_dict = {}
    accuracy_dict = {}

    for student_path in student_paths:
        print(f"Calculating calibration error for {student_path}")
        # Load the student model
        student = torch.load(student_path, weights_only=False)
        student.eval()
        student.to(device)
        student_name = (
            os.path.basename(student_path).split(".")[0].removeprefix("student_")
        )

        # Initialize the true and predicted labels
        y_true = []
        y_pred = []

        # Iterate over the test dataset
        with torch.no_grad():
            for batch in loaders["test"]:
                # Extract the image and mask from the batch
                image = batch["image"].to(device)
                mask = batch["mask"].to(device)

                # Forward pass
                output = student(image)

                # Calculate the predicted probabilities
                prob = torch.sigmoid(output)

                # Append the true and predicted labels
                y_true.extend(mask.cpu().numpy().flatten())
                y_pred.extend(prob.cpu().numpy().flatten())

        # Calculate the ECE and MCE
        ious_dict[student_name] = iou(y_true, y_pred)
        accuracy_dict[student_name] = accuracy(y_true, y_pred)
        ece, mce, accs, confs = calibration_error(np.array(y_true), np.array(y_pred))
        ece_dict[student_name] = ece
        mce_dict[student_name] = mce
        accs_dict[student_name] = accs
        confs_dict[student_name] = confs
        print(f"Student: {student_name}")
        print(
            f"\tIoU = {ious_dict[student_name]:.4f}\n\tAccuracy = {accuracy_dict[student_name]:.4f}"
        )
        print(f"\tECE = {ece:.4f}\n\tMCE = {mce:.4f}")

    # Plot the calibration curves
    print("Plotting calibration curves")
    fig = plot_calibration_curves(accs_dict, confs_dict, ece_dict, mce_dict)

    return fig, ece_dict, mce_dict, accs_dict, confs_dict, ious_dict, accuracy_dict


# %%
if __name__ == "__main__":
    if len(sys.argv) > 1:
        fig, ece_dict, mce_dict, accs_dict, confs_dict, ious_dict, accuracy_dict = (
            main()
        )
        fig.savefig("calibration_curves.png")
        res_names = ["ece", "mce", "ious", "accuracy"]
        for i, results in enumerate([ece_dict, mce_dict, ious_dict, accuracy_dict]):
            with open(f"{res_names[i]}.txt", "w") as f:
                json.dump(results, f)
    else:
        success = os.system(launch_command.format(script=__file__, seed="all_calib"))
        if success == 0:
            print("All calibration curves have been plotted")
        else:
            os.system(f"python {__file__} all_calib")
