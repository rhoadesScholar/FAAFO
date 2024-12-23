# %%
import os
import subprocess

# Specify the base directory where all experiment logs are stored
base_log_dir = os.path.dirname(os.path.realpath(__file__))

# Generate a dictionary for TensorBoard where each experiment's path is mapped to a name
log_dirs = {}
for experiment_name in os.listdir(base_log_dir):
    experiment_path = os.path.join(base_log_dir, experiment_name, "logs")
    if os.path.isdir(experiment_path):
        for seed_dir in os.listdir(experiment_path):
            full_path = os.path.join(experiment_path, seed_dir)
            if os.path.isdir(full_path):
                log_dirs[f"{experiment_name}/seed_{seed_dir}"] = full_path

# Prepare the log directory string for TensorBoard
log_dir_str = ",".join([f"{name}:{path}" for name, path in log_dirs.items()])

# Start TensorBoard with the specified log directories
subprocess.run(["tensorboard", f"--logdir_spec={log_dir_str}"])
