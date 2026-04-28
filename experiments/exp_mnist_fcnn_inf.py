import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys, subprocess, json
from utils.exp_utils import format_cmd

def main():

    output_dir = "outputs/experiment_results"
    os.makedirs(output_dir, exist_ok=True)

    env = os.environ.copy()
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"

    dataset_cfg = {
        "train_per_class": "0:1000, 1:1000, 2:1000, 3:1000, 4:1000, 5:1000, 6:1000, 7:1000, 8:1000, 9:1000",
        "test_per_class": "0:200, 1:200, 2:200, 3:200, 4:200, 5:200, 6:200, 7:200, 8:200, 9:200",
    }

    forget_pct = 50

    cmd = [
        sys.executable, "utils/run_one_pct_inf.py",
        "--forget_pct", str(forget_pct),

        "--model_name", "fcnn",
        "--model_cfg", '{"hidden_widths": [1024, 1024, 1024]}',

        "--dataset_name", "mnist",
        "--dataset_cfg", json.dumps(dataset_cfg),

        "--num_epochs", "5000",
        "--learning_rate", "1e-1",

        "--loss", "rce",
        "--regularization_const", "1e-1",

        "--batch_size_ntk", "500",
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