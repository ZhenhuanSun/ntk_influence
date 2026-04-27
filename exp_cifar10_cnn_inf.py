import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys, subprocess, json
from exp_utils import format_cmd

def main():

    output_dir = "outputs/experiment_results"
    os.makedirs(output_dir, exist_ok=True)

    env = os.environ.copy()
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"

    dataset_cfg = {
        "train_per_class": "0:500,1:500",
        "test_per_class": "0:500,1:500",
    }

    forget_pct = 50

    cmd = [
        sys.executable, "run_one_pct_inf.py",
        "--forget_pct", str(forget_pct),

        "--model_name", "cnn",
        "--model_cfg", '{"channels": [128, 128, 128], "kernel_size": [3, 3]}',

        "--dataset_name", "cifar10",
        "--dataset_cfg", json.dumps(dataset_cfg),

        "--num_epochs", "5000",
        "--learning_rate", "1e-2",

        "--loss", "rls",
        "--regularization_const", "1e-1",

        "--batch_size_ntk", "10",
        "--device_count_ntk", "2",
        "--store_on_device",

        "--output_dir", output_dir,
    ]

    print(
        f"Running:\n{format_cmd(cmd)}",
        flush=True,
    )

    subprocess.run(cmd, check=True, env=env)

    print(f"Done", flush=True)


if __name__ == "__main__":
    main()