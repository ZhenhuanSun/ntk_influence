import time
import numpy as np
from functools import partial

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map, tree_leaves, tree_structure, tree_unflatten
from jax.example_libraries import optimizers
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map

from utils.loss import make_rce_loss_fn, make_rls_loss_fn
from utils.train_utils import make_update_fn
from utils.metrics import make_accuracy_fn, squared_l2_norm, l2_distance, relative_l2_distance

def gpus_warmup():
    """
    Running a small multi-GPU computation to initialize the GPU communication.
    """

    gpus = jax.devices("gpu")
    num_gpus = len(gpus)
    if num_gpus == 0:
        raise RuntimeError("No GPU devices found.")

    mesh = Mesh(np.array(gpus), axis_names=("x",))

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P("x"),),
        out_specs=P(),
    )
    def warmup(x):
        return jax.lax.psum(x, "x")

    _ = warmup(jnp.ones((num_gpus,), dtype=jnp.float32)).block_until_ready()

def format_cmd(cmd):
    """
    Format a command into a multi-line, readable string.

    Parameters
    ----------
    cmd:
        list of str, command and its arguments.
    """

    lines = [f"{cmd[0]} {cmd[1]}"]
    j = 2
    while j < len(cmd):
        if j == len(cmd) - 1 or cmd[j + 1].startswith("--"):
            lines.append(f"  {cmd[j]}")
            j += 1
        else:
            lines.append(f"  {cmd[j]:<22} {cmd[j + 1]}")
            j += 2

    return "\n".join(lines)

def time_first_and_steady(fn, *, num_runs=10):
    """
    Run `fn` once to record the cold-start run time of `fn`, then run it `runs` more times and record the mean and standard
    deviation of the warm-start run time of `fn` (in seconds).

    Parameters:
    -----------
    fn:
        callable, function to time.
    runs:
        int, number of runs after warm up.

    Returns:
    --------
        dict, dictionary containing the number of runs and the recorded time statistics.
    """

    # Measure cold start run time
    t0 = time.perf_counter()
    _ = fn()
    first_ms = (time.perf_counter() - t0) #* 1e3

    if num_runs == 0:
        return {
            "first_s": float(first_ms),
            "num_runs": int(num_runs),
            "steady_mean_s": None,
            "steady_std_s": None,
        }

    times = np.empty((num_runs,), dtype=np.float64)

    # Measure warm start run time
    for i in range(num_runs):
        t0 = time.perf_counter()
        _ = fn()
        times[i] = (time.perf_counter() - t0) #* 1e3

    return {
        "first_s": float(first_ms),
        "num_runs": int(num_runs),
        "steady_mean_s": float(times.mean()),
        "steady_std_s": float(times.std(ddof=0)),
    }

def perturb_params(params, params_ref, key):
    """
    Create a random parameter pytree that has the same l2 distance to `params` as `params_ref` has to `params`.

    Parameters
    ----------
    params:
        pytree, base parameters.
    params_ref:
        pytree, reference parameters.
    key:
        JAX PRNGKey.

    Returns
    -------
    params_noisy:
        pytree with the same structure as `params`, such that ||params_noisy - params||_2 = ||params_r - params||_2.
    """

    target_norm = l2_distance(params_ref, params)

    leaves = tree_leaves(params)
    keys = jax.random.split(key, len(leaves))
    key_tree = tree_unflatten(tree_structure(params), keys)

    # Random perturbation pytree with same structure
    noise = tree_map(
        lambda k, x: jax.random.normal(k, shape=x.shape, dtype=x.dtype),
        key_tree,
        params,
    )

    noise_norm = jnp.sqrt(squared_l2_norm(noise))

    # Scale random perturbation to have the same norm as target_norm
    scaled_noise = tree_map(lambda x: x * (target_norm / noise_norm), noise)
    params_noisy = tree_map(lambda p, n: p + n, params, scaled_noise)

    return params_noisy

def run_training_loop(*, key, num_epochs, xs, ys, batch_size, update, opt_state, params_p):
    """
    Train for num_epochs on (train_images, train_labels) using the specified batch size and optimizer

    Parameters
    ----------
    key:
        JAX PRNGKey, random key for minibatch gradient descent.
    num_epochs:
        int, number of training epochs.
    xs:
        jax.Array of shape (|D|, d_in), input feature matrix.
    ys:
        jax.Array of shape (|D|, d_out), target matrix.
    batch_size:
        int, batch size for minibatch gradient descent.
    update:
        callable, update(step, opt_state, params_p, xs, ys) -> (opt_state, loss, grads).
    opt_state:
        optimizer state returned by opt_init or opt_update.
    params_p:
        linear models:
            pytree, all zeros.
        linearized models:
            pytree, parameters at which the model is linearized or all zeros.
        nonlinear models:
            pytree, parameters at initialization.

    Returns
    -------
    opt_state:
        optimizer state returned by opt_init or opt_update.
    key:
        JAX PRNGKey, random key after splits.
    """

    N = xs.shape[0]
    steps_per_epoch = N // batch_size
    global_step = 0

    for epoch in range(num_epochs):

        key, train_key = jax.random.split(key)

        # Shuffle indices each epoch
        perm = jax.random.permutation(train_key, N)

        # drop remainder to avoid recompilation
        perm = perm[:steps_per_epoch * batch_size]
        perm = perm.reshape((steps_per_epoch, batch_size))

        for step in range(steps_per_epoch):

            global_step += 1

            idx = perm[step]
            x_b = xs[idx]
            y_b = ys[idx]

            opt_state, _, _ = update(global_step, opt_state, params_p, x_b, y_b)

    return opt_state, key

def extract_data(results_list):
    """
    Extract plotting data from experiment results.

    Parameters
    ----------
    results_list:
        list of dict
        Each entry has the form
        {
            "forget_pct": int,
            "N": int,
            "Nr": int,
            "Nf": int,
            "retrain": {...},
            "theta": {...},
            "delta_alpha": [...],
            "noise": {...},
        }

    Returns
    -------
    tuple
        (
            x,
            methods_with_runtime,
            all_methods,
            cold,
            warm_mean,
            warm_std,
            rel,
            acc_forget,
            acc_test,
            acc_forget_retrain,
            acc_test_retrain,
        )
    """

    # Sort by forget percentage
    data = sorted(results_list, key=lambda d: d["forget_pct"])
    x = np.array([d["forget_pct"] for d in data]) # x-axis of the plots

    # Methods with runtime information
    methods_with_runtime = ["theta"] + [m["mode"] for m in data[0]["delta_alpha"]]

    # All methods
    all_methods = methods_with_runtime + ["noise"]

    cold = {m: [] for m in methods_with_runtime}
    warm_mean = {m: [] for m in methods_with_runtime}
    warm_std = {m: [] for m in methods_with_runtime}

    rel = {m: [] for m in all_methods}
    acc_forget = {m: [] for m in all_methods}
    acc_test = {m: [] for m in all_methods}

    acc_forget_retrain = []
    acc_test_retrain = []

    for d in data:
        # retrain baseline
        acc_forget_retrain.append(d["retrain"]["forget accuracy"])
        acc_test_retrain.append(d["retrain"]["test accuracy"])

        # theta method
        t = d["theta"]
        cold["theta"].append(t["total time"]["first_s"])
        warm_mean["theta"].append(t["total time"]["steady_mean_s"])
        warm_std["theta"].append(t["total time"]["steady_std_s"])
        rel["theta"].append(t["relative l2"])
        acc_forget["theta"].append(t["forget accuracy"])
        acc_test["theta"].append(t["test accuracy"])

        # delta_alpha methods
        for m in d["delta_alpha"]:
            mode = m["mode"]
            cold[mode].append(m["total time"]["first_s"])
            warm_mean[mode].append(m["total time"]["steady_mean_s"])
            warm_std[mode].append(m["total time"]["steady_std_s"])
            rel[mode].append(m["relative l2"])
            acc_forget[mode].append(m["forget accuracy"])
            acc_test[mode].append(m["test accuracy"])

        # noise
        n = d["noise"]
        rel["noise"].append(n["relative_l2"])
        acc_forget["noise"].append(n["forget accuracy"])
        acc_test["noise"].append(n["test accuracy"])

    return (
        x,
        methods_with_runtime,
        all_methods,
        cold,
        warm_mean,
        warm_std,
        rel,
        acc_forget,
        acc_test,
        acc_forget_retrain,
        acc_test_retrain,
    )

def run_one_lambda(
    lam: float,
    *,
    num_epochs: int,
    eval_interval: int,
    eta: float,
    loss_kind = "rls",
    params_0,
    train_images, train_labels,
    test_images, test_labels,
    apply_fn,
    apply_fn_lin,
):
    """
    Train model (apply_fn) and linearized model (apply_fn_lin) side-by-side for one lambda, and record metrics vs iteration.

    Parameters
    ----------
    lam:
        float, regularization strength.
    num_epochs:
        int, number of optimizer steps.
    eval_interval:
        int, record metrics every eval_interval steps.
    eta:
        float, learning rate.
    loss_kind:
        {"rls","rce"}, choose which loss to use.
    params_0:
        pytree, initialization parameters (also linearization point).
    train_images, train_labels, test_images, test_labels :
        jax.Array, train adn test datasets.
    apply_fn, apply_fn_lin :
        Forward functions for model and linearized model.

    Returns
    -------
        dict with keys: lam, iters, rels, rmses, accs_nn, accs_lin, gnorms_nn, gnorms_lin
    """

    # Choose loss factory
    if loss_kind == "rls":
        loss_factory = make_rls_loss_fn
    elif loss_kind == "rce":
        loss_factory = make_rce_loss_fn
    else:
        raise ValueError(f"Unknown loss_kind={loss_kind!r}. Use 'rls' or 'rce'.")

    # Build losses from apply_fn and apply_fn_lin
    loss_fn = loss_factory(apply_fn, lam)
    loss_fn_lin = loss_factory(apply_fn_lin, lam)

    # Build optimizers for both models
    opt_init_nn, opt_update_nn, get_params_nn = optimizers.sgd(eta)
    opt_init_lin, opt_update_lin, get_params_lin = optimizers.sgd(eta)

    # Build update_fn from loss
    update_nn = make_update_fn(loss_fn, opt_update_nn, get_params_nn)
    update_lin = make_update_fn(loss_fn_lin, opt_update_lin, get_params_lin)

    # Build accuracy_fn from apply_fn
    accuracy = make_accuracy_fn(apply_fn)
    accuracy_lin = make_accuracy_fn(apply_fn_lin)

    # Initialization
    opt_state_nn = opt_init_nn(params_0)
    opt_state_lin = opt_init_lin(params_0)

    iters, rels, rmses, accs_nn, accs_lin, gnorms_nn, gnorms_lin = [], [], [], [], [], [], []
    global_step = 0

    for _ in range(int(num_epochs)):
        global_step += 1

        opt_state_nn, loss_nn, grads_nn = update_nn(
            global_step, opt_state_nn, params_0, train_images, train_labels
        )
        opt_state_lin, loss_lin, grads_lin = update_lin(
            global_step, opt_state_lin, params_0, train_images, train_labels
        )

        if global_step % eval_interval == 0:
            params_nn = get_params_nn(opt_state_nn)
            params_lin = get_params_lin(opt_state_lin)

            # (1) relative L2 distance between NN params and linearized params (relative to linearized params)
            rel = relative_l2_distance(params_lin, params_nn)

            # L2 distance
            # rel = l2_distance(params_lin, params_nn)

            # RMSE
            # flat_lin, _ = ravel_pytree(params_lin)
            # flat_nn, _  = ravel_pytree(params_nn)
            # rel = jnp.sqrt(jnp.mean((flat_nn - flat_lin) ** 2))

            # (2) RMSE between model and linearized models' outputs on test set
            f_test = apply_fn(params_nn, test_images)
            f_test_lin = apply_fn_lin(params_lin, test_images)
            rmse = jnp.sqrt(jnp.mean((f_test - f_test_lin) ** 2))

            # (3) accuracies on test set
            a_nn = accuracy(params_nn, test_images, test_labels)
            a_lin = accuracy_lin(params_lin, test_images, test_labels)

            # (4) gradient norm of NN params and linearized params
            gnorm_nn = optimizers.l2_norm(grads_nn)
            gnorm_lin = optimizers.l2_norm(grads_lin)

            iters.append(int(global_step))
            rels.append(float(rel))
            rmses.append(float(rmse))
            accs_nn.append(float(a_nn))
            accs_lin.append(float(a_lin))
            gnorms_nn.append(float(gnorm_nn))
            gnorms_lin.append(float(gnorm_lin))

    return {
        "lam": float(lam),
        "loss_kind": str(loss_kind),
        "iters": iters,
        "rels": rels,
        "rmses": rmses,
        "accs_nn": accs_nn,
        "accs_lin": accs_lin,
        "gnorms_nn": gnorms_nn,
        "gnorms_lin": gnorms_lin,
    }