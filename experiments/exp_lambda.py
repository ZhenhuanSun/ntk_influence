import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import time, json
from jax import random
import jax.numpy as jnp

from utils.dataset_utils import build_datasets, binary, one_hot
from utils.model import build_fcnn_model
from utils.exp_utils import run_one_lambda

def main():

    output_dir = "outputs/experiment_results"
    os.makedirs(output_dir, exist_ok=True)

    key = random.PRNGKey(7)

    # --------------------------------------------------------------------------------------------------------------
    # Build dataset
    # --------------------------------------------------------------------------------------------------------------

    key, k_ds = random.split(key)

    dataset_name = "mnist"
    dataset_cfg = {
        "train_per_class": {3: 5000, 8: 5000},
        "test_per_class": {3: None, 8: None},
    }
    # dataset_cfg = {
    #     "train_per_class": {0:200, 1:200, 2:200, 3:200, 4:200, 5:200, 6:200, 7:200, 8:200, 9:200},
    #     "test_per_class": {0:200, 1:200, 2:200, 3:200, 4:200, 5:200, 6:200, 7:200, 8:200, 9:200},
    # }
    ds = build_datasets(dataset_name, dataset_cfg, k_ds)

    train_images = ds["train_images"]
    train_labels = ds["train_labels"]
    test_images = ds["test_images"]
    test_labels = ds["test_labels"]
    classes = ds["classes"]

    N = int(train_images.shape[0])
    Nt = int(test_images.shape[0])
    d_in = int(jnp.prod(jnp.asarray(train_images.shape[1:])))

    # --------------------------------------------------------------------------------------------------------------
    # Encoding
    # --------------------------------------------------------------------------------------------------------------

    train_images = jnp.array(train_images).reshape(N, -1)
    test_images = jnp.array(test_images).reshape(Nt, -1)

    if len(classes) == 2:
        d_out = 1
        train_labels = binary(train_labels, classes)
        test_labels = binary(test_labels, classes)
    else:
        d_out = int(len(classes))
        train_labels = one_hot(train_labels, classes)
        test_labels = one_hot(test_labels, classes)

    # --------------------------------------------------------------------------------------------------------------
    # Model
    # --------------------------------------------------------------------------------------------------------------

    key, k_model = random.split(key)

    model_cfg = {"hidden_widths": [2048, 2048], "linearize": False}
    model_lin_cfg = {"hidden_widths": [2048, 2048], "linearize": True}

    m = build_fcnn_model(model_cfg, k_model, d_in, d_out)
    m_lin = build_fcnn_model(model_lin_cfg, k_model, d_in, d_out)

    params_0 = m["params_0"] # same params_0 for f and f_lin since same random key is used
    apply_fn = m["apply_fn"]
    apply_fn_lin = m_lin["apply_fn"]

    # --------------------------------------------------------------------------------------------------------------
    # Experiment
    # --------------------------------------------------------------------------------------------------------------

    lams = [1e-1, 1e-2, 1e-3, 1e-4]
    # lams = [1e-1]

    num_epochs = 4000
    eval_interval = 1
    eta = 1e-1

    results = []
    for i, lam in enumerate(lams, 1):
        t0 = time.time()
        print(f"[{i}/{len(lams)}] Running lambda={lam:.0e} ...", flush=True)
        results.append(
            run_one_lambda(
                lam,
                num_epochs=num_epochs,
                eval_interval=eval_interval,
                eta=eta,
                loss_kind="rls",
                params_0=params_0,
                train_images=train_images, train_labels=train_labels,
                test_images=test_images, test_labels=test_labels,
                apply_fn=apply_fn, apply_fn_lin=apply_fn_lin,
            )
        )
        print(f"    done in {time.time() - t0:.1f}s", flush=True)

    with open(os.path.join(output_dir, "results_lambda.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()