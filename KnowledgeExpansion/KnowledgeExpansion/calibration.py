# %%
import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from KnowledgeExpansion.training_utils import get_dataloaders, batch_size, num_workers
import torch


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

    return ece, mce, accs, confs


def plot_calibration_curves(accs_dict, confs_dict, result_strings):
    """
    Plot the calibration curves of the student models.

    Parameters
    ----------
    accs_dict : dict
        Dictionary containing the accuracies of the student models.
    confs_dict : dict
        Dictionary containing the confidences of the student models.
    result_strings : dict
        Dictionary containing the formatted name and ECE / MCE values of the student models for the legend.

    Returns
    -------
    matplotlib.pyplot.Figure
        The calibration curves.
    """
    student_type_colors = {"baseline": "blue", "joint": "green", "expanded": "red"}
    plt.figure(figsize=(10, 10))
    for student_name, accs in accs_dict.items():
        for student_type, color in student_type_colors.items():
            if student_type in student_name:
                break
        confs = confs_dict[student_name]
        plt.plot(
            confs,
            accs,
            label=result_strings[student_name],
            color=color,
            marker="o",
        )
    plt.plot([0, 1], [0, 1], linestyle="--", color="black")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Student Calibration Curves")
    plt.legend()
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
        ece, mce, accs, confs = calibration_error(np.array(y_true), np.array(y_pred))
        ece_dict[student_name] = ece
        mce_dict[student_name] = mce
        accs_dict[student_name] = accs
        confs_dict[student_name] = confs
        print(f"ECE = {ece:.4f}, MCE = {mce:.4f}")

    # Print the ECE and MCE values
    result_strings = {}
    for student_name, ece in ece_dict.items():
        result_strings[student_name] = (
            f"{student_name}:\n\tECE = {ece:.4f},\n\tMCE = {mce_dict[student_name]:.4f}"
        )
        print(result_strings[student_name])

    # Plot the calibration curves
    print("Plotting calibration curves")
    fig = plot_calibration_curves(accs_dict, confs_dict, result_strings)

    return fig, ece_dict, mce_dict, accs_dict, confs_dict


# %%
if __name__ == "__main__":
    fig, ece_dict, mce_dict, accs_dict, confs_dict = main()
    fig.savefig("calibration_curves.png")
