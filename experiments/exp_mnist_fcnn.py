import os, sys, subprocess, argparse, json, glob
from utils.exp_utils import format_cmd

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
    # If overwrite results, all files with filenames that start with "results_mnist_fcnn_" and contain "lam" will be removed.
    if args.overwrite_results:
        # delete all results file containing "lam"
        pattern = os.path.join(output_dir, "results_mnist_fcnn_*lam*.json")
        for path in glob.glob(pattern):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    # Be sure to delete the cached model parameters and NTK matrix whenever any of the experiment settings below change.
    if args.delete_cache:
        # delete all params files with filenames that start with "params_mnist_fcnn_" and contain "lam".
        pattern = os.path.join(output_dir, "params_mnist_fcnn_*lam*.pkl")
        for path in glob.glob(pattern):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

        ntk_path = os.path.join(output_dir, "ntk_mnist_fcnn.pkl")
        try:
            os.remove(ntk_path)
        except FileNotFoundError:
            pass

    env = os.environ.copy()
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # dataset_cfg = {
    #     "train_per_class": "0:200, 1:200, 2:200, 3:200, 4:200, 5:200, 6:200, 7:200, 8:200, 9:200",
    #     "test_per_class": "0:200, 1:200, 2:200, 3:200, 4:200, 5:200, 6:200, 7:200, 8:200, 9:200",
    # }

    dataset_cfg = {
        "train_per_class": "3: 5000, 8: 5000",
        "test_per_class": "3: None, 8: None",
    }

    lams = [1e-2, 1e-4]
    forget_percentages = [10, 30, 50, 70, 90]

    for lam in lams:
        print(f"\n=== Running experiments for lambda={lam} ===\n", flush=True)

        for i, pct in enumerate(forget_percentages, start=1):
            cmd = [
                sys.executable, "utils/run_one_pct.py",
                "--forget_pct", str(pct),
                "--runs", "5",

                "--model_name", "fcnn",
                "--model_cfg", '{"hidden_widths": [2048, 2048]}',

                "--dataset_name", "mnist",
                "--dataset_cfg", json.dumps(dataset_cfg),

                "--optimizer_name", "sgd",
                "--optimizer_cfg", '{"num_epochs": 4000, "learning_rate": 1e-1}',

                "--loss", "rls",
                "--regularization_const", str(lam),

                "--batch_size_ntk", "100",
                "--device_count_ntk", "2",

                "--output_dir", output_dir,
            ]

            print(
                f"[lam={lam}] [{i}/{len(forget_percentages)}]: running:\n{format_cmd(cmd)}",
                flush=True,
            )

            subprocess.run(cmd, check=True, env=env)

            print(
                f"[lam={lam}] [{i}/{len(forget_percentages)}]: done",
                flush=True,
            )

        print(f"=== All runs complete for lambda={lam} ===\n", flush=True)


if __name__ == "__main__":
    main()