import os
import argparse
import json
import pickle
import numpy as np

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
from jax import random
from jax.example_libraries import optimizers
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
from utils.loss import make_rls_loss_fn, make_rce_loss_fn
from utils.train_utils import make_update_fn, make_batched_apply
from utils.exp_utils import run_training_loop, time_first_and_steady, perturb_params
from utils.metrics import make_accuracy_fn, relative_l2_distance
from utils.influence import prepare_solve_theta, prepare_solve_delta_alpha, influence_on_theta, influence_on_delta_alpha, \
    map_back_to_theta_space

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
    ap.add_argument("--runs", type=int, default=1)

    ap.add_argument("--model_name", type=str, required=True, choices=["linear", "fcnn", "cnn"])
    ap.add_argument("--model_cfg", type=str, required=True,
                    help="JSON string describing the model architecture.\n" + MODEL_CONFIG_EXAMPLES,)
    ap.add_argument("--linearize", action="store_true",
                    help="If set, build and return linearized model.")

    ap.add_argument("--dataset_name", type=str, required=True, choices=["mnist", "cifar10"])
    ap.add_argument("--dataset_cfg", type=str, required=True,
                    help="JSON string describing dataset details.\n" + DATASET_CONFIG_EXAMPLES)

    ap.add_argument("--optimizer_name", type=str, default="sgd", choices=["sgd", "momentum"])
    ap.add_argument("--optimizer_cfg", type=str, required=True,
                    help="JSON string describing optimizer configuration.\n" + OPTIMIZER_CONFIG_EXAMPLES,)

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

    optimizer_name = args.optimizer_name

    try:
        optimizer_cfg = json.loads(args.optimizer_cfg)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"--optimizer_cfg must be valid JSON. Got: {args.optimizer_cfg!r}\n"
            f"{OPTIMIZER_CONFIG_EXAMPLES}"
        ) from e

    if not isinstance(optimizer_cfg, dict):
        raise ValueError(
            f"--optimizer_cfg must decode to a JSON object (dict). "
            f"Got type={type(optimizer_cfg).__name__}.\n"
            f"{OPTIMIZER_CONFIG_EXAMPLES}"
        )

    missing = [k for k in ["num_epochs", "learning_rate"] if k not in optimizer_cfg]
    if missing:
        raise ValueError(
            f"optimizer_cfg requires keys {missing}. "
            f"Got keys={list(optimizer_cfg.keys())}.\n{OPTIMIZER_CONFIG_EXAMPLES}"
        )

    num_epochs = int(optimizer_cfg["num_epochs"])
    eta = float(optimizer_cfg["learning_rate"])
    batch_size = None if optimizer_cfg.get("batch_size") is None else int(optimizer_cfg["batch_size"])

    lr_schedule_cfg = optimizer_cfg.get("lr_schedule", None)
    if lr_schedule_cfg is not None:
        schedule_name = lr_schedule_cfg.get("name", None)
        if schedule_name == "exponential_decay":
            missing = [k for k in ["decay_steps", "decay_rate"] if k not in lr_schedule_cfg]
            if missing:
                raise ValueError(
                    f"lr_schedule='exponential_decay' requires keys {missing}. "
                    f"Got keys={list(lr_schedule_cfg.keys())}."
                )

            schedule = optimizers.exponential_decay(
                step_size=eta,
                decay_steps=int(lr_schedule_cfg["decay_steps"]),
                decay_rate=float(lr_schedule_cfg["decay_rate"]),
            )

        else:
            raise ValueError(
                f"Unsupported lr_schedule name={schedule_name!r}. "
                "Currently supported: 'exponential_decay'."
            )
    else:
        schedule = eta

    loss = args.loss
    lam = args.regularization_const

    # ------------------------------------------------------------------------------------------------------------------
    # Setup for building NTK matrix
    # ------------------------------------------------------------------------------------------------------------------

    batch_size_ntk = args.batch_size_ntk
    device_count_ntk = args.device_count_ntk
    store_on_device = args.store_on_device

    def run_one_pct(forget_classes, forget_pct, runs, *, params_path, ntk_path):

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
            params_0 = m["params_0"]
            apply_fn = m["apply_fn"]

            train_images = extract_rrf(params_fixed, train_images.reshape(N, -1))
            retain_images = extract_rrf(params_fixed, retain_images.reshape(Nr, -1))
            forget_images = extract_rrf(params_fixed, forget_images.reshape(Nf, -1))
            test_images = extract_rrf(params_fixed, test_images.reshape(Nt, -1))

            params = params_0
            params_r = params_0
            params_p = tree_map(jnp.zeros_like, params_0)

        elif model_name == "fcnn":
            d_in = int(jnp.prod(jnp.asarray(train_images.shape[1:])))
            m = build_fcnn_model(model_cfg, k_model, d_in, d_out)

            train_images = train_images.reshape(N, -1)
            retain_images = retain_images.reshape(Nr, -1)
            forget_images = forget_images.reshape(Nf, -1)
            test_images = test_images.reshape(Nt, -1)

            params_0 = m["params_0"]
            apply_fn = m["apply_fn"]

            params = params_0
            params_r = params_0
            params_p = params_0

        else:
            d_in = train_images.shape[1:]
            m = build_cnn_model(model_cfg, k_model, d_in, d_out)

            params_0 = m["params_0"]
            apply_fn = m["apply_fn"]

            params = params_0
            params_r = params_0
            params_p = params_0

        # --------------------------------------------------------------------------------------------------------------
        # Train model on original dataset
        # --------------------------------------------------------------------------------------------------------------

        if loss == "rls":
            loss_fn = make_rls_loss_fn(apply_fn, float(lam))
            loss_fnl = lambda f, t: 0.5 * jnp.mean((f - t) ** 2)
        elif loss == "rce":
            loss_fn = make_rce_loss_fn(apply_fn, float(lam))
            loss_fnl = lambda f, t: -jnp.mean(jnp.sum(t * jax.nn.log_softmax(f), axis=-1))
        else:
            raise ValueError(f"Unsupported loss={loss!r}. Use 'rls' or 'rce'.")

        if optimizer_name == "sgd":
            opt_init, opt_update, get_params = optimizers.sgd(schedule)
        elif optimizer_name == "momentum":
            mass = float(optimizer_cfg.get("mass", 0.9))
            opt_init, opt_update, get_params = optimizers.momentum(schedule, mass)
        else:
            raise ValueError(
                f"Unsupported optimizer_name={optimizer_name!r}. "
                "Currently supported: 'sgd', 'momentum'."
            )

        update = make_update_fn(loss_fn, opt_update, get_params)

        # Only train the full model if no cached parameters are found; otherwise load them from disk
        if os.path.exists(params_path):
            # When unpickled, JAX arrays are typically restored to the default device, rather than their original device.
            with open(params_path, "rb") as f:
                params = pickle.load(f) # on gpus[0]
        else:
            opt_state = opt_init(params)
            batch_size_train = N if batch_size is None else batch_size
            opt_state, key = run_training_loop(
                key=key,
                num_epochs=num_epochs,
                xs=train_images,
                ys=train_labels,
                batch_size=batch_size_train,
                update=update,
                opt_state=opt_state,
                params_p=params_p,
            )
            params = get_params(opt_state)

            with open(params_path, "wb") as f:
                pickle.dump(params, f)

        # --------------------------------------------------------------------------------------------------------------
        # Build NTK matrix
        # --------------------------------------------------------------------------------------------------------------

        # Only rebuild the NTK matrix if no cached matrix is found; otherwise load it from disk
        if os.path.exists(ntk_path):
            with open(ntk_path, "rb") as f:
                K_X_X = jnp.array(pickle.load(f), device=jax.devices("cpu")[0]) # jax array on cpu
        else:
            ntk_fn = nt.empirical_ntk_fn(
                apply_fn,
                trace_axes=(),
                vmap_axes=0,
                implementation=nt.NtkImplementation.JACOBIAN_CONTRACTION
            )
            # Number of rows of kernel matrix must divide batch size.
            # When device parallelism is used, the batch size was expanded by a factor of device_count.
            ntk_fn_batched = nt.batch(
                ntk_fn,
                device_count=int(device_count_ntk),
                batch_size=int(batch_size_ntk),
                store_on_device=bool(store_on_device),
            )
            # Store K_X_X in CPU to allow larger kernel matrices to be computed
            K_X_X = ntk_fn_batched(train_images, None, params_p)

            # Store K_X_X as numpy array so that when unpickled K_X_X stays on cpu
            with open(ntk_path, "wb") as f:
                pickle.dump(np.asarray(K_X_X), f)

        # --------------------------------------------------------------------------------------------------------------
        # Train model on retain dataset
        # --------------------------------------------------------------------------------------------------------------

        opt_state_r = opt_init(params_r)
        batch_size_retrain = Nr if batch_size is None else batch_size
        opt_state_r, key = run_training_loop(
            key=key,
            num_epochs=num_epochs,
            xs=retain_images,
            ys=retain_labels,
            batch_size=batch_size_retrain,
            update=update,
            opt_state=opt_state_r,
            params_p=params_p,
        )
        params_r = get_params(opt_state_r)

        # Measure retrained model's accuracies on test set and forget set
        # We do the following instead of accuracy = make_accuracy_fn(apply_fn) to reduce peak memory for cases where model
        # and dataset are large
        batched_apply_fn = make_batched_apply(apply_fn, batch_size=100)
        accuracy = make_accuracy_fn(batched_apply_fn)
        test_accuracy_r = float(accuracy(params_r, test_images, test_labels))
        forget_accuracy_r = float(accuracy(params_r, forget_images, forget_labels))

        results_retrain = {
            "test accuracy": test_accuracy_r,
            "forget accuracy": forget_accuracy_r,
        }

        # --------------------------------------------------------------------------------------------------------------
        # Influence on theta
        # --------------------------------------------------------------------------------------------------------------

        # Measure the total time required to prepare and solve the linear system for theta
        def _influence_on_theta():
            H_fn, rhs = prepare_solve_theta(loss_fn, params, params_p, train_images, train_labels, forget_indices)
            params_u, _ = influence_on_theta(H_fn, rhs, params)

            # Wait until JAX arrays are fully computed on devices
            tree_map(lambda x: x.block_until_ready(), params_u)

            return params_u

        theta_time_stats = time_first_and_steady(_influence_on_theta, num_runs=runs)

        # Measure the l2 distance between params_r and theta_params_u and theta_params_u's accuracies on test set and forget set
        theta_params_u = _influence_on_theta()
        theta_rel_l2 = float(relative_l2_distance(params_r, theta_params_u))
        theta_test_acc = float(accuracy(theta_params_u, test_images, test_labels))
        theta_forget_acc = float(accuracy(theta_params_u, forget_images, forget_labels))

        results_theta = {
            "total time": theta_time_stats,
            "relative l2": theta_rel_l2,
            "forget accuracy": theta_forget_acc,
            "test accuracy": theta_test_acc,
        }

        # --------------------------------------------------------------------------------------------------------------
        # Influence on delta alpha
        # --------------------------------------------------------------------------------------------------------------

        # Record time statistics for prepare_solve_delta_alpha and influence_on_delta_alpha under different argument settings
        def _collect_da_stats(mode, **prepare_kwargs):

            def _influence_on_delta_alpha():

                # Use batched_apply_fn() instead of apply_fn() to reduce peak memory
                H_rr, rhs, delta_alpha_f = prepare_solve_delta_alpha(
                    loss_fnl, batched_apply_fn(params, train_images),
                    train_labels, forget_indices, lam,
                    K_X_X, **prepare_kwargs)

                delta_alpha = influence_on_delta_alpha(H_rr, rhs, delta_alpha_f, train_images, forget_indices)
                params_u = map_back_to_theta_space(apply_fn, params, params_0, train_images, delta_alpha)

                # Wait until JAX arrays are fully computed on devices
                tree_map(lambda x: x.block_until_ready(), params_u)

                return params_u

            # Measure the total time required to prepare and solve the linear system for delta alpha for `prepare_kwargs` argument setting
            da_time_stats = time_first_and_steady(_influence_on_delta_alpha, num_runs=runs)

            # Measure the l2 distance between params_r and da_params_u and da_params_u's accuracies on test set and forget set
            da_params_u = _influence_on_delta_alpha()
            da_rel_l2 = float(relative_l2_distance(params_r, da_params_u))
            da_test_acc = float(accuracy(da_params_u, test_images, test_labels))
            da_forget_acc = float(accuracy(da_params_u, forget_images, forget_labels))

            return {
                "mode": mode,
                "total time": da_time_stats,
                "relative l2": da_rel_l2,
                "forget accuracy": da_forget_acc,
                "test accuracy": da_test_acc,
            }

        results_delta_alpha = [
            _collect_da_stats("operator_wo_sharding", shard_K_X_X=False),
            _collect_da_stats("operator_w_sharding", shard_K_X_X=True),
        ]

        # --------------------------------------------------------------------------------------------------------------
        # Create a baseline for comparison
        # --------------------------------------------------------------------------------------------------------------

        # Measure the l2 distance between params_r and params_noisy and params_noisy's accuracies on test set and forget set
        key, random_key = random.split(key)
        params_noisy = perturb_params(params, params_r, random_key)
        noisy_rel_l2 = float(relative_l2_distance(params_r, params_noisy))
        test_accuracy_noisy = float(accuracy(params_noisy, test_images, test_labels))
        forget_accuracy_noisy = float(accuracy(params_noisy, forget_images, forget_labels))

        results_noise = {
            "relative_l2": noisy_rel_l2,
            "test accuracy": test_accuracy_noisy,
            "forget accuracy": forget_accuracy_noisy,
        }

        return {
            "forget_pct": int(forget_pct),
            "N": int(N),
            "Nr": int(Nr),
            "Nf": int(Nf),
            "retrain": results_retrain,
            "theta": results_theta,
            "delta_alpha": results_delta_alpha,
            "noise": results_noise,
        }

    os.makedirs(args.output_dir, exist_ok=True)

    if args.model_name == "linear":
        ntk_cache_path = os.path.join(args.output_dir, f"ntk_{args.dataset_name}_{args.model_name}.pkl")
        params_cache_path = os.path.join(args.output_dir, f"params_{args.dataset_name}_{args.model_name}.pkl")
        output_path = os.path.join(args.output_dir, f"results_{args.dataset_name}_{args.model_name}.json")
    elif args.linearize:
        ntk_cache_path = os.path.join(args.output_dir, f"ntk_{args.dataset_name}_{args.model_name}_lin.pkl")
        params_cache_path = os.path.join(args.output_dir, f"params_{args.dataset_name}_{args.model_name}_lin.pkl")
        output_path = os.path.join(args.output_dir, f"results_{args.dataset_name}_{args.model_name}_lin.json")
    else:
        ntk_cache_path = os.path.join(args.output_dir, f"ntk_{args.dataset_name}_{args.model_name}.pkl")

        lam_str = f"{args.regularization_const}".replace(".", "_")
        params_cache_path = os.path.join(args.output_dir, f"params_{args.dataset_name}_{args.model_name}_lam_{lam_str}.pkl")
        output_path = os.path.join(args.output_dir, f"results_{args.dataset_name}_{args.model_name}_lam_{lam_str}.json")

    out = run_one_pct(args.forget_classes, args.forget_pct, args.runs, params_path=params_cache_path, ntk_path=ntk_cache_path)

    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            results = json.load(f)
    else:
        results = []

    # Remove old result with the same forget_pct, add the new one, then sort the results by forget_pct
    results = [r for r in results if r.get("forget_pct") != out["forget_pct"]]
    results.append(out)
    results.sort(key=lambda r: r["forget_pct"])

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()