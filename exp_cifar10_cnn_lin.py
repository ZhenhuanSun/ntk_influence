import os, sys, subprocess, argparse, json
from exp_utils import format_cmd

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--overwrite_results",
        action="store_true",
        help="If set, delete existing result files before running.",
    )
    ap.add_argument(
        "--delete_cache",
        action="store_true",
        help="If set, delete existing cached model parameters and NTK matrix.",
    )
    args = ap.parse_args()

    output_dir = "outputs/experiment_results"
    os.makedirs(output_dir, exist_ok=True)

    # If not overwrite results, new results will be appended to existing results
    if args.overwrite_results:
        results_path = os.path.join(output_dir, "results_cifar10_cnn_lin.json")
        try:
            os.remove(results_path)
        except FileNotFoundError:
            pass

    # Be sure to delete the cached model parameters and NTK matrix whenever any of the experiment settings below change.
    if args.delete_cache:
        params_path = os.path.join(output_dir, "params_cifar10_cnn_lin.pkl")
        ntk_path = os.path.join(output_dir, "ntk_cifar10_cnn_lin.pkl")

        for path in [params_path, ntk_path]:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    env = os.environ.copy()
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"

    dataset_cfg = {
        "train_per_class": "0:1000,1:1000",
        "test_per_class": "0:1000,1:1000",
    }

    forget_percentages = [10, 30, 50, 70, 90]

    for i, pct in enumerate(forget_percentages, start=1):
        cmd = [
            sys.executable, "run_one_pct.py",
            "--forget_pct", str(pct),
            "--forget_classes", "0",
            "--runs", "5",

            "--model_name", "cnn",
            "--model_cfg", '{"channels": [128, 128, 128], "kernel_size": [3, 3]}',
            "--linearize",

            "--dataset_name", "cifar10",
            "--dataset_cfg", json.dumps(dataset_cfg),

            "--optimizer_name", "momentum",
            "--optimizer_cfg", '{"num_epochs": 500, "learning_rate": 1e-1}',

            "--loss", "rls",
            "--regularization_const", "1e-1",

            "--batch_size_ntk", "250",
            "--device_count_ntk", "2",

            "--output_dir", output_dir,
        ]

        print(
            f"[{i}/{len(forget_percentages)}]: running:\n{format_cmd(cmd)}",
            flush=True,
        )

        subprocess.run(cmd, check=True, env=env)

        print(f"[{i}/{len(forget_percentages)}]: done", flush=True)

    print("All runs complete.")


if __name__ == "__main__":
    main()