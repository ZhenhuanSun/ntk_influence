import os
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent # run experiments in project root directory
env = os.environ.copy()
env["PYTHONPATH"] = str(project_root)

exp_dir = "experiments"

commands = [
    [sys.executable, f"{exp_dir}/exp_mnist_fcnn_lin.py", "--overwrite_results", "--delete_cache"],
    [sys.executable, f"{exp_dir}/exp_cifar10_cnn_lin.py", "--overwrite_results", "--delete_cache"],
    [sys.executable, f"{exp_dir}/exp_lambda.py"],
    [sys.executable, f"{exp_dir}/exp_mnist_fcnn.py", "--overwrite_results", "--delete_cache"],
    [sys.executable, f"{exp_dir}/exp_mnist_fcnn_inf.py"],
    [sys.executable, f"{exp_dir}/exp_cifar10_cnn_inf.py"],
]

for cmd in commands:
    print(f"Running {' '.join(cmd)}...", flush=True)
    subprocess.run(cmd, check=True, env=env)

print("All experiments complete.", flush=True)