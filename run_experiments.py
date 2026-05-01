import os, subprocess, sys, argparse
from pathlib import Path

# run experiments in project root directory
project_root = Path(__file__).resolve().parent
env = os.environ.copy()
env["PYTHONPATH"] = str(project_root)

exp_dir = "experiments"

EXPERIMENTS = {
    "mnist_linear": [sys.executable, f"{exp_dir}/exp_mnist_linear.py", "--overwrite_results", "--delete_cache"],
    "mnist_fcnn_lin": [sys.executable, f"{exp_dir}/exp_mnist_fcnn_lin.py", "--overwrite_results", "--delete_cache"],
    "cifar10_cnn_lin": [sys.executable, f"{exp_dir}/exp_cifar10_cnn_lin.py", "--overwrite_results", "--delete_cache"],
    "lambda": [sys.executable, f"{exp_dir}/exp_lambda.py"],
    "mnist_fcnn": [sys.executable, f"{exp_dir}/exp_mnist_fcnn.py", "--overwrite_results", "--delete_cache"],
    "mnist_fcnn_inf": [sys.executable, f"{exp_dir}/exp_mnist_fcnn_inf.py"],
    "cifar10_cnn_inf": [sys.executable, f"{exp_dir}/exp_cifar10_cnn_inf.py"],
}

FIGURES = {
    "fig1": ["mnist_fcnn_lin", "cifar10_cnn_lin"],
    "fig2": ["lambda"],
    "fig3": ["mnist_fcnn"],
    "fig4": ["mnist_fcnn_inf", "cifar10_cnn_inf"],
}

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--figure",
        choices=FIGURES.keys(),
        help="Run experiments needed to generate a specific figure.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=EXPERIMENTS.keys(),
        help="Run selected experiments by name.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all experiments.",
    )

    args = parser.parse_args()

    if args.all:
        # Select all experiments except the one for linear model
        # To include the experiment on linear model, remove [1:]
        selected_exps = list(EXPERIMENTS.keys())[1:]
    elif args.figure:
        selected_exps = FIGURES[args.figure]
    elif args.experiments:
        selected_exps = args.experiments
    else:
        parser.error("Specify --all, --figure, or --experiments.")

    for exp in selected_exps:
        cmd = EXPERIMENTS[exp]
        print(f"Running {exp}: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True, env=env)

    print("All selected experiments complete.", flush=True)

if __name__ == "__main__":
    main()