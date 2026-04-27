import subprocess
import sys

commands = [
    [sys.executable, "exp_mnist_fcnn_lin.py", "--overwrite_results", "--delete_cache"],
    [sys.executable, "exp_cifar10_cnn_lin.py", "--overwrite_results", "--delete_cache"],
    [sys.executable, "exp_lambda.py"],
    [sys.executable, "exp_mnist_fcnn.py", "--overwrite_results", "--delete_cache"],
    [sys.executable, "exp_mnist_fcnn_inf.py"],
    [sys.executable, "exp_cifar10_cnn_inf.py"],
]

for cmd in commands:
    print(f"Running {' '.join(cmd)}...", flush=True)
    subprocess.run(cmd, check=True)

print("All experiments complete.", flush=True)