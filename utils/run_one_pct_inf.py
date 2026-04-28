import os
import argparse
import json
import numpy as np

import jax
import jax.numpy as jnp
from jax import random
from utils.exp_utils import gpus_warmup

# -----------------------------
# Multi-GPU warmup workaround
# -----------------------------
# We found that importing neural_tangents before running any multi-GPU workload can cause multi-GPU execution to fail.
# (Probably because importing neural_tangents triggers initialization of GPU communication libraries (e.g., NCCL) in a
# way that interferes with subsequent multi-GPU execution)
# We work around this by running a small multi-GPU computation first (to initialize the GPU communication), and only then
# importing neural_tangents.
gpus_warmup()

import neural_tangents as nt
from utils.dataset_utils import build_datasets, build_retain_and_forget_by_percentage, binary, one_hot
from utils.model import build_linear_model, build_fcnn_model, build_cnn_model
from utils.train_utils import make_kgd_update_fn
from utils.influence import (prepare_solve_delta_alpha, influence_on_delta_alpha, influence_on_outputs_delta_alpha,
                             influence_on_loss_delta_alpha)
from utils.loss import difference_in_loss_fnl

MODEL_CONFIG_EXAMPLES = (
    "Example Usage:\n"
    "  linear:\n"
    "    required: num_rrf\n"
    "    optional: W_std, b_std, parameterization\n"
    "    example: '{\"num_rrf\": 20000}'\n"
    "    full example: '{\"num_rrf\": 20000, \"W_std\": 2.0, \"b_std\": 0.1, \"parameterization\": \"ntk\"}'\n"
    "\n"
    "  fcnn:\n"
    "    required: hidden_widths\n"
    "    optional: W_std, b_std, parameterization\n"
    "    example: '{\"hidden_widths\": [512, 512]}'\n"
    "    full example: '{\"hidden_widths\": [512, 512], \"W_std\": 2.0, \"b_std\": 0.1, \"parameterization\": \"ntk\"}'\n"
    "\n"
    "  cnn:\n"
    "    required: channels, kernel_size\n"
    "    optional: pool_window, pool_strides, conv_padding, pool_padding, W_std, b_std, parameterization\n"
    "    example: '{\"channels\": [128, 128, 128], \"kernel_size\": [3, 3]}'\n"
    "    full example: '{\"channels\": [128, 128, 128], \"kernel_size\": [3, 3], \"pool_window\": [2, 2], \"pool_strides\": [2, 2], \"conv_padding\": \"SAME\", \"pool_padding\": \"SAME\", \"W_std\": 2.0, \"b_std\": 0.1, \"parameterization\": \"ntk\"}'\n"
)

DATASET_CONFIG_EXAMPLES = (
    "Example Usage:\n"
    "  required: train_per_class, test_per_class\n"
    "  example: '{\"train_per_class\": \"3:5000,8:5000\", \"test_per_class\": \"3:1000,8:1000\"}'\n"
)

OPTIMIZER_CONFIG_EXAMPLES = (
    "Example Usage:\n"
    "  sgd:\n"
    "    required: num_epochs, learning_rate\n"
    "    optional: batch_size, lr_schedule\n"
    "    example: '{\"num_epochs\": 1000, \"learning_rate\": 1.0, \"batch_size\": 100, "
    "\"lr_schedule\": {\"name\": \"exponential_decay\", \"decay_steps\": 500, \"decay_rate\": 0.9}}'\n"
    "\n"
    "  momentum:\n"
    "    required: num_epochs, learning_rate\n"
    "    optional: mass, batch_size, lr_schedule\n"
    "    example: '{\"num_epochs\": 1000, \"learning_rate\": 1.0, \"mass\": 0.9, \"batch_size\": 100, "
    "\"lr_schedule\": {\"name\": \"exponential_decay\", \"decay_steps\": 500, \"decay_rate\": 0.9}}'\n"
)

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--forget_pct", type=int, required=True)
    ap.add_argument("--forget_classes", type=int, nargs="+", default=None)

    ap.add_argument("--model_name", type=str, required=True, choices=["linear", "fcnn", "cnn"])
    ap.add_argument("--model_cfg", type=str, required=True,
                    help="JSON string describing the model architecture.\n" + MODEL_CONFIG_EXAMPLES,)
    ap.add_argument("--linearize", action="store_true",
                    help="If set, build and return linearized model.")

    ap.add_argument("--dataset_name", type=str, required=True, choices=["mnist", "cifar10"])
    ap.add_argument("--dataset_cfg", type=str, required=True,
                    help="JSON string describing dataset details.\n" + DATASET_CONFIG_EXAMPLES)

    ap.add_argument("--num_epochs", type=int, default=1000)
    ap.add_argument("--learning_rate", type=float, default=0.1)

    ap.add_argument("--loss", type=str, default="rls", choices=["rls", "rce"])
    ap.add_argument("--regularization_const", type=float, default=1e-3)

    ap.add_argument("--batch_size_ntk", type=int, default=1000)
    ap.add_argument("--device_count_ntk", type=int, default=2)
    ap.add_argument("--store_on_device", action="store_true")

    ap.add_argument("--output_dir", type=str, required=True,
                    help="Directory where the output JSON file will be saved.")

    args = ap.parse_args()

    # ------------------------------------------------------------------------------------------------------------------
    # Model configuration
    # ------------------------------------------------------------------------------------------------------------------

    model_name = args.model_name

    try:
        model_cfg = json.loads(args.model_cfg)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"--model_cfg must be valid JSON. Got: {args.model_cfg!r}\n"
            f"{MODEL_CONFIG_EXAMPLES}"
        ) from e

    if not isinstance(model_cfg, dict):
        raise ValueError(
            f"--model_cfg must decode to a JSON object (dict). Got type={type(model_cfg).__name__}.\n"
            f"{MODEL_CONFIG_EXAMPLES}"
        )

    model_cfg["linearize"] = bool(args.linearize)

    if model_name == "linear":

        # expect {"num_features": int}
        if "num_rrf" not in model_cfg:
            raise ValueError(
                f"linear model requires --model_cfg to contain key 'num_rrf'. Got keys={list(model_cfg.keys())}.\n"
                f"{MODEL_CONFIG_EXAMPLES}"
            )

        model_cfg["num_rrf"] = int(model_cfg["num_rrf"])

    elif model_name == "fcnn":

        # expect {"hidden_widths": [..]}
        if "hidden_widths" not in model_cfg:
            raise ValueError(
                f"fcnn model requires --model_spec to contain key 'hidden_widths'. Got keys={list(model_cfg.keys())}.\n"
                f"{MODEL_CONFIG_EXAMPLES}"
            )

        hw = model_cfg["hidden_widths"]
        if not isinstance(hw, (list, tuple)):
            raise ValueError(
                f"'hidden_widths' must be a list, got type={type(hw).__name__}.\n"
                f"{MODEL_CONFIG_EXAMPLES}"
            )

        model_cfg["hidden_widths"] = [int(x) for x in hw]

    elif model_name == "cnn":

        # expect {"channels": [..], "kernel_size": [h, w]}
        missing = [k for k in ("channels", "kernel_size") if k not in model_cfg]
        if missing:
            raise ValueError(
                f"cnn model requires --model_spec to contain keys {missing}. "
                f"Got keys={list(model_cfg.keys())}.\n"
                f"{MODEL_CONFIG_EXAMPLES}"
            )

        channels = model_cfg["channels"]
        kernel_size = model_cfg["kernel_size"]

        if not isinstance(channels, (list, tuple)):
            raise ValueError(
                f"'channels' must be a list, got type={type(channels).__name__}.\n"
                f"{MODEL_CONFIG_EXAMPLES}"
            )
        if not isinstance(kernel_size, (list, tuple)) or len(kernel_size) != 2:
            raise ValueError(
                f"'kernel_size' must be a list/tuple of length 2, got {kernel_size}.\n"
                f"{MODEL_CONFIG_EXAMPLES}"
            )

        model_cfg["channels"] = [int(c) for c in channels]
        model_cfg["kernel_size"] = tuple(int(k) for k in kernel_size)

    else:
        raise ValueError(
            f"Unsupported model_name={model_name!r}. "
            f"Supported models are 'linear', 'fcnn', and 'cnn'."
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Dataset configuration
    # ------------------------------------------------------------------------------------------------------------------

    def parse_map(s):
        """
        Parse a comma-separated string, e.g., "3:5000,8:5000" into a {int: int/None} dict.
        """
        out = {}
        for part in s.split(","):
            k, v = part.split(":")
            out[int(k.strip())] = None if v.strip() == "None" else int(v.strip())
        return out

    # def parse_map(s):
    #     """Parse a comma-separated string, e.g., "3:5000,8:5000" into a {int: int} dict."""
    #     out = {}
    #     for part in s.split(","):
    #         k, v = part.split(":")
    #         out[int(k.strip())] = int(v.strip())
    #     return out

    dataset_name = args.dataset_name

    try:
        dataset_cfg = json.loads(args.dataset_cfg)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"--dataset_cfg must be valid JSON. Got: {args.dataset_cfg!r}\n"
            f"{MODEL_CONFIG_EXAMPLES}"
        ) from e

    if not isinstance(dataset_cfg, dict):
        raise ValueError(
            f"--dataset_cfg must decode to a JSON object (dict). Got type={type(dataset_cfg).__name__}.\n"
            f"{MODEL_CONFIG_EXAMPLES}"
        )


    missing = [k for k in ["train_per_class", "test_per_class"] if k not in dataset_cfg]
    if missing:
        raise ValueError(
            f"mnist dataset requires keys {missing}. "
            f"Got keys={list(dataset_cfg.keys())}.\n{DATASET_CONFIG_EXAMPLES}"
        )

    if not isinstance(dataset_cfg["train_per_class"], str):
        raise ValueError(
            f"'train_per_class' must be a string like '3:5000,8:5000'. "
            f"Got type={type(dataset_cfg['train_per_class']).__name__}.\n"
            f"{DATASET_CONFIG_EXAMPLES}"
        )

    if not isinstance(dataset_cfg["test_per_class"], str):
        raise ValueError(
            f"'test_per_class' must be a string like '3:1000,8:1000'. "
            f"Got type={type(dataset_cfg['test_per_class']).__name__}.\n"
            f"{DATASET_CONFIG_EXAMPLES}"
        )

    dataset_cfg["train_per_class"] = parse_map(dataset_cfg["train_per_class"])
    dataset_cfg["test_per_class"] = parse_map(dataset_cfg["test_per_class"])

    # ------------------------------------------------------------------------------------------------------------------
    # Training configuration
    # ------------------------------------------------------------------------------------------------------------------

    eta = args.learning_rate
    num_epochs = args.num_epochs
    loss = args.loss
    lam = args.regularization_const

    # ------------------------------------------------------------------------------------------------------------------
    # Setup for building NTK matrix
    # ------------------------------------------------------------------------------------------------------------------

    batch_size_ntk = args.batch_size_ntk
    device_count_ntk = args.device_count_ntk
    store_on_device = args.store_on_device

    def run_one_pct(forget_classes, forget_pct):

        # Same universal random seed is used to make sure same dataset is built
        key = random.PRNGKey(7)

        # --------------------------------------------------------------------------------------------------------------
        # Build original dataset
        # --------------------------------------------------------------------------------------------------------------

        key, k_ds = random.split(key)

        ds = build_datasets(dataset_name, dataset_cfg, k_ds)
        train_images = ds["train_images"]
        train_labels = ds["train_labels"]
        test_images = ds["test_images"]
        test_labels = ds["test_labels"]
        classes = ds["classes"]

        N = int(train_images.shape[0])
        Nt = int(test_images.shape[0])

        # --------------------------------------------------------------------------------------------------------------
        # Build retain and forget dataset
        # --------------------------------------------------------------------------------------------------------------

        key, k_split = random.split(key)

        (retain_indices, retain_images, retain_labels), (forget_indices, forget_images, forget_labels) = \
            build_retain_and_forget_by_percentage(k_split, train_images, train_labels, forget_classes, forget_pct)

        Nf = int(len(forget_indices))
        Nr = int(N - Nf)

        # --------------------------------------------------------------------------------------------------------------
        # Encoding
        # --------------------------------------------------------------------------------------------------------------

        train_images = jnp.array(train_images)
        retain_images = jnp.array(retain_images)
        forget_images = jnp.array(forget_images)
        test_images = jnp.array(test_images)

        if len(classes) == 2:
            d_out = 1
            train_labels = binary(train_labels, classes)
            retain_labels = binary(retain_labels, classes)
            forget_labels = binary(forget_labels, classes)
            test_labels = binary(test_labels, classes)
        else:
            d_out = int(len(classes))
            train_labels = one_hot(train_labels, classes)
            retain_labels = one_hot(retain_labels, classes)
            forget_labels = one_hot(forget_labels, classes)
            test_labels = one_hot(test_labels, classes)

        # --------------------------------------------------------------------------------------------------------------
        # Model
        # --------------------------------------------------------------------------------------------------------------

        key, k_model = random.split(key)

        if model_name == "linear":
            d_in = int(jnp.prod(jnp.asarray(train_images.shape[1:])))
            m = build_linear_model(model_cfg, k_model, d_in, d_out)

            params_fixed = m["params_fixed"]
            extract_rrf = m["extract_rrf"]
            kernel_fn = m["kernel_fn"]

            train_images = extract_rrf(params_fixed, train_images.reshape(N, -1))
            retain_images = extract_rrf(params_fixed, retain_images.reshape(Nr, -1))
            forget_images = extract_rrf(params_fixed, forget_images.reshape(Nf, -1))
            test_images = extract_rrf(params_fixed, test_images.reshape(Nt, -1))


        elif model_name == "fcnn":
            d_in = int(jnp.prod(jnp.asarray(train_images.shape[1:])))
            m = build_fcnn_model(model_cfg, k_model, d_in, d_out)

            train_images = train_images.reshape(N, -1)
            retain_images = retain_images.reshape(Nr, -1)
            forget_images = forget_images.reshape(Nf, -1)
            test_images = test_images.reshape(Nt, -1)

            kernel_fn = m["kernel_fn"]

        else:
            d_in = train_images.shape[1:]
            m = build_cnn_model(model_cfg, k_model, d_in, d_out)

            kernel_fn = m["kernel_fn"]

        # --------------------------------------------------------------------------------------------------------------
        # Build NTK matrix
        # --------------------------------------------------------------------------------------------------------------

        kernel_fn_batched = nt.batch(
            kernel_fn,
            batch_size=int(batch_size_ntk),
            device_count=int(device_count_ntk),
            store_on_device=bool(store_on_device),
        )

        K_X_X = kernel_fn_batched(train_images, train_images, 'ntk')
        K_X_Xr = K_X_X[:, retain_indices]

        # --------------------------------------------------------------------------------------------------------------
        # Train model on original dataset
        # --------------------------------------------------------------------------------------------------------------

        if loss == "rls":
            loss_fnl = lambda f, t: 0.5 * jnp.mean((f - t) ** 2)
        elif loss == "rce":
            loss_fnl = lambda f, t: -jnp.mean(jnp.sum(t * jax.nn.log_softmax(f), axis=-1))
        else:
            raise ValueError(f"Unsupported loss={loss!r}. Use 'rls' or 'rce'.")

        f0_X = jnp.zeros((N, d_out)) # assume initial model outputs are all zeros

        f_X = f0_X.copy()
        kgd_update = make_kgd_update_fn(loss_fnl, lam, f0=f0_X)

        for epoch in range(num_epochs):
            f_X, _, _ = kgd_update(f_X, train_labels, K_X_X, eta)

        alpha_star = -jax.grad(loss_fnl)(f_X, train_labels) / lam

        # --------------------------------------------------------------------------------------------------------------
        # Train model on retain dataset
        # --------------------------------------------------------------------------------------------------------------

        fr_X = f0_X.copy()
        kgd_update_r = make_kgd_update_fn(loss_fnl, lam, f0=f0_X, active_indices=retain_indices)

        for epoch in range(num_epochs):
            fr_X, _, _ = kgd_update_r(fr_X, train_labels, K_X_Xr, eta)

        alpha_r_star = -jax.grad(loss_fnl)(fr_X[retain_indices], retain_labels) / lam

        # --------------------------------------------------------------------------------------------------------------
        # Influence on delta alpha
        # --------------------------------------------------------------------------------------------------------------

        H_rr, rhs, delta_alpha_f = prepare_solve_delta_alpha(loss_fnl, f_X, train_labels, forget_indices, lam, K_X_X, shard_K_X_X=False)
        delta_alpha = influence_on_delta_alpha(H_rr, rhs, delta_alpha_f, train_images, forget_indices)

        # --------------------------------------------------------------------------------------------------------------
        # Influence on outputs on test data points
        # --------------------------------------------------------------------------------------------------------------

        K_Xt_X = kernel_fn_batched(test_images, train_images, 'ntk')
        K_Xt_Xr = K_Xt_X[:, retain_indices]

        f0_Xt = jnp.zeros((Nt, d_out))  # since we assume initial model outputs are all zeros
        f_Xt = K_Xt_X @ alpha_star + f0_Xt
        fr_Xt = K_Xt_Xr @ alpha_r_star + f0_Xt

        true_output_diffs_at_Xt = fr_Xt - f_Xt
        est_output_diffs_at_Xt = influence_on_outputs_delta_alpha(delta_alpha, K_Xt_X, shard=True)

        # --------------------------------------------------------------------------------------------------------------
        # Influence on loss on test data points
        # --------------------------------------------------------------------------------------------------------------

        true_loss_diffs_at_Xt = difference_in_loss_fnl(loss_fnl, f_Xt, fr_Xt, test_labels)
        est_loss_diffs_at_Xt = influence_on_loss_delta_alpha(loss_fnl, f_X, train_labels, delta_alpha, lam, K_X_X,
                                                             f_Xt, test_labels, K_Xt_X, shard=True)

        return {
            "true_output_diffs_at_Xt": np.asarray(true_output_diffs_at_Xt).tolist(),
            "est_output_diffs_at_Xt": np.asarray(est_output_diffs_at_Xt).tolist(),
            "true_loss_diffs_at_Xt": np.asarray(true_loss_diffs_at_Xt).tolist(),
            "est_loss_diffs_at_Xt": np.asarray(est_loss_diffs_at_Xt).tolist(),
        }

    out = run_one_pct(args.forget_classes, args.forget_pct)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"results_{args.dataset_name}_{args.model_name}_inf.json")
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()