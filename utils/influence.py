import jax
import jax.numpy as jnp
from jax import grad, jvp, vjp, vmap, jit, lax
from jax.tree_util import tree_map
from jax.scipy.sparse.linalg import cg
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map

import math
import numpy as np
from loss import make_upweight_loss_fn
from utils.train_utils import make_kvp, make_batched_grad
from functools import partial


def hvp(f, primals, tangents):
    """
    Compute the Hessian vector product via forward-over-reverse automatic differentiation without materializing the
    Hessian matrix. Adapted from: https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html

    Parameters:
    -----------
    f:
        callable, f(inputs) -> outputs.
    primals:
        pytree, parameters at which Hessian is evaluated.
    tangents:
        pytree, vector to be multiplied by Hessian.

    Returns:
    --------
        pytree (with the same structure as primals), Hessian–vector product.
    """

    return jvp(grad(f), primals, tangents)[1]

def make_sharded_kvp(mesh, K_ndim):
    """
    Build a fully-sharded NTK matrix-vector product function for a given mesh.

    Parameters
    ----------
    mesh:
        jax.sharding.Mesh, device mesh used for sharding.
    K_ndim:
        int, number of dimensions of K

    Returns
    -------
    callable
        sharded_kvp(v, K) -> Kv
    """

    if K_ndim == 2:

        @partial(
            shard_map,
            mesh=mesh,
            in_specs=(P("x", None), P("x", None)),  # v and K are sharded by rows
            out_specs=P("x", None),                           # output is sharded by rows
            check_rep=False,
        )
        def sharded_kvp(v_block, K_block):
            """
            Parameters
            ----------
            v_block:
                jax.Array of shape (local_j, d_out), row-sharded vector block.
            K_block:
                jax.Array of shape (local_i, j), NTK matrix block sharded on axis 0.

            Returns
            -------
                jax.Array of shape (local_i, d_out), row-sharded output block.
            """

            # Reconstruct full v on each device from row shards.
            v_full = jax.lax.all_gather(v_block, "x", axis=0, tiled=True)
            return K_block @ v_full

        return sharded_kvp

    elif K_ndim == 4:

        @partial(
            shard_map,
            mesh=mesh,
            in_specs=(P("x", None), P("x", None, None, None)),  # v and K are sharded by rows
            out_specs=P("x", None),                                       # output is sharded by rows
            check_rep=False,
        )
        def sharded_kvp(v_block, K_block):
            """
            Parameters
            ----------
            v_block:
                jax.Array of shape (local_j, b), row-sharded vector block.
            K_block:
                jax.Array of shape (local_i, j, a, b), NTK matrix block sharded on axis 0.

            Returns
            -------
                jax.Array of shape (local_i, a), row-sharded output block.
            """

            # Reconstruct full v on each device from row shards.
            v_full = jax.lax.all_gather(v_block, "x", axis=0, tiled=True)
            return jnp.einsum("ijab,jb->ia", K_block, v_full)

        return sharded_kvp

    else:
        raise ValueError(f"K must be 2D or 4D, got K_ndim={K_ndim}")

def make_batched_hvp(loss_fn, params, params_p, xrs, yrs, batch_size, N):
    """
    Build a batched Hessian vector product function for the upweighted empirical risk L_D(theta, -1/|D|). Note that
    (|D_r| / |D|) L_{D_r}(theta) = (1 / |D|) sum_{(x, y) in D_r} (loss + regularization). This implementation produces
    the same HVP as
                upweight_loss_fn = make_upweight_loss_fn(loss_fn, params_p, xs, ys, forget_indices)
                hvp_fn = lambda v: hvp(upweight_loss_fn, (params,), (v,))  # Linear operator: v -> Hv
    but with lower peak memory usage, so it can be used for larger model and dataset.

    Parameters:
    -----------
    loss_fn:
        callable, loss_fn(params, params_p, xs, ys) -> loss.
    params:
        pytree, model parameters.
    params_p:
        linear models:
            pytree, all zeros.
        linearized models:
            pytree, parameters at which the model is linearized or all zeros.
        nonlinear models:
            pytree, parameters at initialization.
    xrs:
        jax.Array of shape (|D_r|, d_in), retain feature matrix, where d_in may be a scalar or a tuple.
    yrs:
        jax.Array of shape (|D_r|, d_out), retain target matrix.
    batch_size:
        int, default=100, minibatch size for batched HVP.
    N:
        int, number of data points in full dataset.

    Returns:
    --------
    hvp_fn:
        callable, hvp_fn(v) -> Hv, where H is the hessian of upweight_loss_fn with respect to params.
    """

    Nr = xrs.shape[0]
    num_steps = math.ceil(Nr / batch_size)
    pad = num_steps * batch_size - Nr

    # Pad to make sure fixed shapes across iterations
    xrs = jnp.pad(xrs, ((0, pad),) + ((0, 0),) * (xrs.ndim - 1))
    yrs = jnp.pad(yrs, ((0, pad),) + ((0, 0),) * (yrs.ndim - 1))

    xrs = xrs.reshape(num_steps, batch_size, *xrs.shape[1:])
    yrs = yrs.reshape(num_steps, batch_size, *yrs.shape[1:])

    # mask to recover true batch sizes
    mask = jnp.pad(jnp.ones(Nr, dtype=xrs.dtype), (0, pad))
    mask = mask.reshape(num_steps, batch_size)

    def hvp_fn(v):

        init = tree_map(jnp.zeros_like, v)

        def body(carry, inputs):
            x_b, y_b, m_b = inputs
            Nb = jnp.sum(m_b)
            fn = lambda p: loss_fn(p, params_p, x_b, y_b) # mean over batch
            hvp_b = tree_map(lambda x: Nb * x, hvp(fn, (params,), (v,))) # sum over batch
            carry = tree_map(lambda x, y: x + y, carry, hvp_b)
            return carry, None

        hvp_sum, _ = lax.scan(body, init, (xrs, yrs, mask))

        return tree_map(lambda x: x / N, hvp_sum)

    return hvp_fn

def prepare_solve_theta(loss_fn, params, params_p, xs, ys, forget_indices, batch_size=100):
    """
    Prepare all components needed to solve linear system (4) in the paper.

    Parameters:
    -----------
    loss_fn:
        callable, loss_fn(params, params_p, xs, ys) -> loss.
    params:
        pytree, model parameters.
    params_p:
        linear models:
            pytree, all zeros.
        linearized models:
            pytree, parameters at which the model is linearized or all zeros.
        nonlinear models:
            pytree, parameters at initialization.
    xs:
        jax.Array of shape (|D|, d_in), feature matrix.
    ys:
        jax.Array of shape (|D|, d_out), target matrix.
    forget_indices:
        jax.Array of shape (|D_f|,), indices of forget samples.
    batch_size:
        int, default=100, minibatch size for batched HVP and batched grad.

    Returns:
    --------
    hvp_fn:
        callable, H_fn(v) -> Hv, where H is the hessian of upweight_loss_fn with respect to params.
    rhs:
        pytree (with the same structure as params), right-hand side vector of the linear system in theta-space.
    """

    N = xs.shape[0]
    Nf = len(forget_indices)

    retain_indices = jnp.setdiff1d(jnp.arange(N), forget_indices)
    xrs, yrs = xs[retain_indices], ys[retain_indices]
    xfs, yfs = xs[forget_indices], ys[forget_indices]

    # The following code is the batched version of
    # upweight_loss_fn = make_upweight_loss_fn(loss_fn, params_p, xs, ys, forget_indices)
    # H_fn = lambda v: hvp(upweight_loss_fn, (params,), (v,))  # Linear operator: v -> Hv
    # It enables the computation of HVPs for larger models and retain datasets
    H_fn = make_batched_hvp(loss_fn, params, params_p, xrs, yrs, batch_size=batch_size, N=N)

    # The following code is the batched version of
    # rhs = tree_map(lambda s: (Nf / N) * s, grad(loss_fn)(params, params_p, xfs, yfs))
    # It enables the computation of gradients for larger models and forget datasets
    batched_grad = make_batched_grad(loss_fn, batch_size=batch_size)
    rhs = tree_map(lambda s: (Nf / N) * s, batched_grad(params, params_p, xfs, yfs))

    # Wait until JAX arrays are fully computed on devices
    # tree_map(lambda x: x.block_until_ready(), rhs)

    return H_fn, rhs

def influence_on_theta(H_fn, rhs, params, maxiter=500, tol=1e-8):
    """
    Solve linear system (4) in the paper.

    Parameters:
    -----------
    H_fn:
        callable, H_fn(v) -> Hv, where H is the hessian of upweight_loss_fn with respect to params.
    rhs:
        pytree (with the same pytree structure as params), the right hand side of the linear system in theta-space.
    maxiter:
        int, default=500, maximum number of iterations for conjugate gradient descent solver.
    tol:
        float, default=1e-8, tolerance for conjugate gradient descent solver.

    Returns:
    --------
    params_r:
        pytree (with the same pytree structure as params), retrained model parameters estimated by influence function on theta.
    delta_theta:
        pytree (with the same pytree structure as params), difference between the retrained model parameter and the original
        model parameters.
    """

    x0 = tree_map(jnp.zeros_like, params)
    delta_theta, _ = cg(H_fn, rhs, x0=x0, maxiter=maxiter, tol=tol)
    params_r = tree_map(lambda p, dp: p + dp, params, delta_theta)

    # Wait until JAX arrays are fully computed on devices
    # tree_map(lambda x: x.block_until_ready(), (params_r, delta_theta))

    return params_r, delta_theta

def _build_vs_block(K_rr, start, real_end, batch_size, d_out):
    """
    Build a block of K(Xr, Xr) \in R^{d_out|D_r| \times d_out|D_r|} from either 2D K_rr or 4D K_rr.

    Parameters
    ----------
    K_rr:
        jax.Array of shape (|D_r|, |D_r|) or (|D_r|, |D_r}, d_out, d_out), K(X_r, X_r). If K_rr is 2D, it is equivalent to K_rr \otimes I_{d_out}.
    start:
        int, starting column index (inclusive).
    real_end:
        int, ending column index (exclusive).
    batch_size:
        int, output batch size. If real_end - start < batch_size, the result is zero-padded along the batch dimension.
    d_out:
        int, output dimension.

    Returns
    -------
    vs:
        jax.Array of shape (batch_size, Nr, d_out), a block of K(Xr, Xr) containing columns from start to real_end.
    """

    b = real_end - start
    Nr = K_rr.shape[0]

    if K_rr.ndim == 2:

        # column indices of K(Xr, Xr)
        flat_indices = jnp.arange(start, real_end)
        # The column of K(Xr, Xr) corresponding to flat_indices[k] is constructed by selecting column_indices[k]-th column
        # of K_rr and placing it in output_indices[k]-th output dimension, with zeros in all other dimensions.
        column_indices = flat_indices // d_out
        output_indices = flat_indices % d_out

        selected_columns = K_rr[:, column_indices].T   # shape (b, Nr)

        vs = jnp.zeros((b, Nr, d_out), dtype=K_rr.dtype)
        vs = vs.at[jnp.arange(b), :, output_indices].set(selected_columns)

    elif K_rr.ndim == 4:
        vs = K_rr.transpose(1, 3, 0, 2).reshape(Nr * d_out, Nr, d_out)[start:real_end]

    else:
        raise ValueError(f"K_rr must be 2D or 4D, got shape {K_rr.shape}")

    # Fix the shape of vs to avoid recompiling the jitted function.
    if b != batch_size:
        padded = jnp.zeros((batch_size, Nr, d_out), dtype=vs.dtype)
        vs = padded.at[:b].set(vs)

    return vs

def prepare_solve_delta_alpha(loss_fnl, fs, ys, forget_indices, lam, K_X_X, *, materialize_H_rr=False, shard_K_X_X=False, batch_size=None):
    """
    Prepare all components needed to solve linear system (13) in the paper.

    Parameters:
    -----------
    loss_fnl:
        callable, loss_fnl(fs, ys) -> loss.
    fs:
        jax.Array of shape (|D|, d_out), model outputs on the full dataset.
    ys:
        jax.Array of shape (|D|, d_out), target matrix.
    forget_indices:
        jax.Array of shape (|D_f|,), indices of forget samples.
    lam:
        float, regularization constant.
    K_X_X:
        jax.Array of shape (|D|, |D|, d_out, d_out) or shape (|D|, |D|), the NTK matrix. If it is 2D, it is equivalent to
        K_X_X \otimes I_{d_out}.
    materialize_H_rr:
        bool, default=False, whether to materialize H_rr matrix.
    shard_K_X_X:
        bool, default=False, whether to shard K_X_X matrix.
    batch_size:
        int, default=None, the number of columns of K_Xr_Xr processed concurrently when materializing H_rr matrix. A smaller
        batch_size reduces peak memory usage and shortens the initial compilation time of batched_K_rr_H_rr_vp_fn, at the
        cost of requiring more batches.

    Returns:
    --------
    if materialize_H_rr:
        H_rr:
            jax.Array of shape (d_out * |D_r|, |D_r|, d_out), left hand side matrix of Equation (13).
    else:
        H_rr_fn:
            callable, H_rr_fn(v_r) = H_rr v_r, where H_rr is the left hand side matrix of Equation (13).
    if shard_K_X_X:
        rhs:
            jax.Array of shape (|D_r| + pad, d_out), right hand side vector of Equation (13), where |D_r| + pad is the
            smallest multiple of the number of GPUs such that the input and output of H_rr_fn(v_r) can be evenly sharded
            across GPUs.
        rhs:
            jax.Array of shape (|D_r|, d_out), right hand side vector of Equation (13).
    delta_alpha_f:
        jax.Array of shape (|D_f|, d_out), entries of delta alpha that are known.
    """

    if materialize_H_rr and batch_size is None:
        raise ValueError(
            "batch_size must be provided when materialize_H_rr=True."
        )

    if not materialize_H_rr and batch_size is not None:
        raise ValueError(
            "batch_size is only used when materialize_H_rr=True."
        )

    K_ndim = K_X_X.ndim
    N, d_out = ys.shape
    Nf = len(forget_indices)
    Nr = N - Nf
    H_rr_dim = d_out * Nr
    retain_indices = jnp.setdiff1d(jnp.arange(N), forget_indices)

    gpus = jax.devices("gpu")

    if not shard_K_X_X:

        # Put linear system related variables in second GPU because the default GPU is usually occupied with other miscellaneous
        # stuff like dataset and model parameters.
        fs = jax.device_put(fs, gpus[1])
        ys = jax.device_put(ys, gpus[1])
        frs = fs[retain_indices]
        ffs = fs[forget_indices]
        yrs = ys[retain_indices]
        yfs = ys[forget_indices]

        loss_fnl_r = lambda f: loss_fnl(f, yrs)
        hvp_fn = lambda v: hvp(loss_fnl_r, (frs,), (v,))
        kvp = make_kvp(K_ndim)

        alpha_star = -grad(loss_fnl)(fs, ys) / lam
        delta_alpha_f = -alpha_star[forget_indices]

        # Put K_Xr_Xr and K_Xr_Xf in second GPU
        K_Xr_Xr = jax.device_put(K_X_X[jnp.ix_(retain_indices, retain_indices)], device=gpus[1])
        K_Xr_Xf = jax.device_put(K_X_X[jnp.ix_(retain_indices, forget_indices)], device=gpus[1])

        if materialize_H_rr:

            def batched_K_rr_H_rr_vp_fn(vs, K_rr):
                def one(v):
                    return kvp(hvp_fn(v), K_rr)
                return vmap(one, in_axes=0)(vs)

            def H_rf_vp_fn(v, K_rf, K_rr):
                K_rf_v = kvp(v, K_rf)
                return (Nr / N) * (kvp(hvp_fn(K_rf_v), K_rr) + lam * K_rf_v)

            rhs = (
                (Nf / N) * (
                    kvp(grad(loss_fnl)(ffs, yfs), K_Xr_Xf)
                    + lam * (
                        kvp(alpha_star[forget_indices], K_Xr_Xf)
                        + kvp(alpha_star[retain_indices], K_Xr_Xr)
                    )
                )
                - H_rf_vp_fn(delta_alpha_f, K_Xr_Xf, K_Xr_Xr)
            )

            @jit
            def build_H_rr_block(vs, K_rr):
                return (Nr / N) * (batched_K_rr_H_rr_vp_fn(vs, K_rr) + lam * vs)

            H_rr = np.empty((H_rr_dim, Nr, d_out), dtype=np.float32)

            # Materialize H_rr batch by batch and stream each computed block back to CPU to reduce GPU memory pressure.
            for start in range(0, H_rr_dim, batch_size):
                real_end = min(start + batch_size, H_rr_dim)

                vs = _build_vs_block(K_Xr_Xr, start, real_end, batch_size, d_out)

                H_rr_block = build_H_rr_block(vs, K_Xr_Xr)
                H_rr[start:real_end] = jax.device_get(H_rr_block[:(real_end - start)])

            # H_rr is on cpu by the time the loop finishes, so there is no need to block it
            # tree_map(lambda x: x.block_until_ready(), (rhs, delta_alpha_f))

            return H_rr, rhs, delta_alpha_f

        else:

            def H_rr_vp_fn(v, K_rr):
                K_rr_v = kvp(v, K_rr)[:Nr]
                return (Nr / N) * (kvp(hvp_fn(K_rr_v), K_rr)[:Nr] + lam * K_rr_v)

            def H_rf_vp_fn(v, K_rf, K_rr):
                K_rf_v = kvp(v, K_rf)
                return (Nr / N) * (kvp(hvp_fn(K_rf_v), K_rr) + lam * K_rf_v)

            rhs = (
                (Nf / N) * (
                    kvp(grad(loss_fnl)(ffs, yfs), K_Xr_Xf)
                    + lam * (
                        kvp(alpha_star[forget_indices], K_Xr_Xf)
                        + kvp(alpha_star[retain_indices], K_Xr_Xr)
                    )
                )
                - H_rf_vp_fn(delta_alpha_f, K_Xr_Xf, K_Xr_Xr)
            )

            # tree_map(lambda x: x.block_until_ready(), (rhs, delta_alpha_f))

            return lambda v: H_rr_vp_fn(v, K_Xr_Xr), rhs, delta_alpha_f

    else:

        mesh = Mesh(np.array(gpus), axis_names=("x",))
        v_shard = NamedSharding(mesh, P("x", None))

        # Padding
        num_gpus = mesh.shape['x']
        Nr_pad = math.ceil(Nr / num_gpus) * num_gpus # Smallest multiple of num_gpus that is >= Nr
        Nf_pad = math.ceil(Nf / num_gpus) * num_gpus # Smallest multiple of num_gpus that is >= Nf

        def pad_rows(x, target_num_rows):
            """Pad rows of x into target_num_rows."""
            return jnp.pad(x, ((0, target_num_rows - x.shape[0]), (0, 0)))

        frs = fs[retain_indices]
        ffs = fs[forget_indices]
        yrs = ys[retain_indices]
        yfs = ys[forget_indices]
        frs_pad = jax.device_put(pad_rows(frs, Nr_pad), v_shard)
        yrs_pad = jax.device_put(pad_rows(yrs, Nr_pad), v_shard)

        loss_fnl_r_pad = lambda f: loss_fnl(f, yrs_pad)
        hvp_fn = lambda v_pad: hvp(loss_fnl_r_pad, (frs_pad,), (v_pad,))
        sharded_kvp = make_sharded_kvp(mesh, K_ndim)

        alpha_star = -grad(loss_fnl)(fs, ys) / lam
        delta_alpha_f = -alpha_star[forget_indices]
        delta_alpha_f_pad = jax.device_put(pad_rows(delta_alpha_f, Nf_pad), v_shard)

        # Pad K_Xr_Xr and K_Xr_Xf so that they shard evenly across multiple GPUs and are compatible with the sharded matvec
        if K_ndim == 2:
            K_shard = NamedSharding(mesh, P("x", None))
            K_Xr_Xr = jax.device_put(jnp.pad(K_X_X[jnp.ix_(retain_indices, retain_indices)],
                                             ((0, Nr_pad - Nr), (0, Nr_pad - Nr))), K_shard)
            K_Xr_Xf = jax.device_put(jnp.pad(K_X_X[jnp.ix_(retain_indices, forget_indices)],
                                             ((0, Nr_pad - Nr), (0, Nf_pad - Nf))), K_shard)
        elif K_ndim == 4:
            K_shard = NamedSharding(mesh, P("x", None, None, None))
            K_Xr_Xr = jax.device_put(jnp.pad(K_X_X[jnp.ix_(retain_indices, retain_indices)],
                                             ((0, Nr_pad - Nr), (0, Nr_pad - Nr), (0, 0), (0, 0))), K_shard)
            K_Xr_Xf = jax.device_put(jnp.pad(K_X_X[jnp.ix_(retain_indices, forget_indices)],
                                             ((0, Nr_pad - Nr), (0, Nf_pad - Nf), (0, 0), (0, 0))), K_shard)
        else:
            raise ValueError(f"K must be 2D or 4D, got K_ndim={K_ndim}")

        def H_rr_vp_fn(v_pad):
            K_rr_v = sharded_kvp(v_pad, K_Xr_Xr)
            hv = hvp_fn(K_rr_v)
            out = sharded_kvp(hv, K_Xr_Xr)
            return (Nr / N) * (out + lam * K_rr_v)

        def H_rf_vp_fn(v_f_pad):
            K_rf_v = sharded_kvp(v_f_pad, K_Xr_Xf)
            hv = hvp_fn(K_rf_v)
            out = sharded_kvp(hv, K_Xr_Xr)
            return (Nr / N) * (out + lam * K_rf_v)

        grad_f_pad = jax.device_put(pad_rows(grad(loss_fnl)(ffs, yfs), Nf_pad), v_shard)
        alpha_f_pad = jax.device_put(pad_rows(alpha_star[forget_indices], Nf_pad), v_shard)
        alpha_r_pad = jax.device_put(pad_rows(alpha_star[retain_indices], Nr_pad), v_shard)

        rhs = (
            (Nf / N) * (
                sharded_kvp(grad_f_pad, K_Xr_Xf)
                + lam * (
                    sharded_kvp(alpha_f_pad, K_Xr_Xf)
                    + sharded_kvp(alpha_r_pad, K_Xr_Xr)
                )
            )
            - H_rf_vp_fn(delta_alpha_f_pad)
        )

        # tree_map(lambda x: x.block_until_ready(), (rhs, delta_alpha_f_pad))

        return H_rr_vp_fn, rhs, delta_alpha_f

def influence_on_delta_alpha(H_rr, rhs, delta_alpha_f, xs, forget_indices, eps=1e-2, maxiter=500, tol=1e-8):
    """
    Solve linear system (13) in the paper.

    Parameters:
    -----------
    if H_rr is jax.Array:
        H_rr:
            jax.Array of shape (d_out * |D_r|, |D_r|, d_out), left hand side matrix of Equation (13).
        rhs:
            jax.Array of shape (|D_r|, d_out), right hand side vector of Equation (13).
    else:
        H_rr:
            callable, H_rr_fn(v_r) = H_rr v_r, where H_rr is the left hand side matrix of Equation (13).
        rhs:
            jax.Array of shape (|D_r| + pad, d_out), right hand side vector of Equation (13), where |D_r| + pad is the
            smallest multiple of the number of GPUs such that the input and output of H_rr_fn(v_r) can be evenly sharded
            across GPUs.
    delta_alpha_f:
        jax.Array of shape (|D_f|, d_out), entries of delta alpha that are known.
    xs:
        jax.Array of shape (|D|, d_in), feature matrix.
    forget_indices:
        jax.Array of shape (|D_f|,), indices of forget samples.
    eps:
        float, default=1e-2, diagonal damping term added to H_rr (When H_rr is materialized) to improve numerical stability
        in solving the linear system.
    maxiter:
        int, default=500, maximum number of iterations for conjugate gradient descent solver.
    tol:
        float, default=1e-8, tolerance for conjugate gradient descent solver.

    Returns:
    --------
    delta_alpha:
        jax.Array of shape (|D|, d_out), difference between the dual parameters of the retrained and original models.
    """

    N = xs.shape[0]
    d_out = rhs.shape[1]
    retain_indices = jnp.setdiff1d(jnp.arange(N), forget_indices)
    Nr =len(retain_indices)

    delta_alpha = jnp.zeros((N, d_out)).at[forget_indices].set(delta_alpha_f)

    if isinstance(H_rr, np.ndarray):

        # Put H_rr to the same GPU as rhs
        H_rr = jax.device_put(H_rr, rhs.device)

        # Solve delta_alpha_r using standard linear system solver, faster but less accurate sometimes
        delta_alpha_r = jnp.linalg.solve(H_rr.reshape(Nr * d_out, Nr * d_out), rhs.reshape(Nr * d_out)) # (Nr * d_out,)

        # Solve delta_alpha_r using standard linear system solver
        # Add a small damping term to the diagonal of H_rr to improve numerical stability.
        # delta_alpha_r = jnp.linalg.solve(
        #     H_rr.reshape(Nr * d_out, Nr * d_out) + eps * jnp.eye(Nr * d_out, dtype=H_rr.dtype),
        #     rhs.reshape(Nr * d_out)
        # )  # shape: (Nr * d_out,)

        delta_alpha = delta_alpha.at[retain_indices].set(delta_alpha_r.reshape(Nr, d_out))

    else:

        x0 = jnp.zeros_like(rhs)
        # The cg() function requires H_rr operator to have the same input-output structure (same shard)
        delta_alpha_r, _ = cg(H_rr, rhs, x0=x0, maxiter=maxiter, tol=tol)   # (Nr + pad, d_out) sharded
        delta_alpha_r = jax.device_put(delta_alpha_r, delta_alpha.device)   # (Nr + pad, d_out) gather
        delta_alpha = delta_alpha.at[retain_indices].set(delta_alpha_r[:Nr])

    # Wait until JAX arrays are fully computed on devices
    # tree_map(lambda x: x.block_until_ready(), delta_alpha)

    return delta_alpha

def _batched_vjp(apply_fn, primals, xs, tangents, batch_size):
    """
    batched version of
                        _, f_vjp = vjp(lambda p: apply_fn(p, xs), primals,)
                        result, = f_vjp(tangents)
    It helps reduce the peak memory of vector-Jacobian product when model and dataset are large.

    Parameters:
    -----------
    apply_fn:
        callable, apply_fn(params, xs, ys) -> logits.
    primals:
        linear models:
            pytree, whatever, the Jacobian of linear model does not depend on parameters.
        linearized models:
            pytree, parameters at which the model is linearized.
        nonlinear models:
            pytree, parameters at initialization.
    xs:
        jax.Array of shape (|D|, d_in), where d_in may be a scalar or a tuple, feature matrix.
    tangents:
        jax.Array of shape (|D|, d_out), vector to be multiplied by Jacobian matrix.
    batch_size:
        int, default=100, minibatch size for batched VJP.

    Returns:
    --------
    result:
        pytree with the same structure as primals, result of VJP.
    """

    # N = xs.shape[0]
    # result = tree_map(jnp.zeros_like, primals)
    #
    # for start in range(0, N, batch_size):
    #     end = min(start + batch_size, N)
    #     x_batch = xs[start:end]
    #     tangents_batch = tangents[start:end]
    #
    #     _, f_vjp = jax.vjp(lambda p: apply_fn(p, x_batch), primals)
    #     result_partial, = f_vjp(tangents_batch)
    #     result = tree_map(lambda x, y: x + y, result, result_partial) # accumulate the partial results

    N = xs.shape[0]
    num_steps = math.ceil(N / batch_size)
    pad = num_steps * batch_size - N

    # Pad to make sure fixed shapes across iterations
    xs = jnp.pad(xs, ((0, pad),) + ((0, 0),) * (xs.ndim - 1))
    tangents = jnp.pad(tangents, ((0, pad),) + ((0, 0),) * (tangents.ndim - 1))

    xs = xs.reshape(num_steps, batch_size, *xs.shape[1:])
    tangents = tangents.reshape(num_steps, batch_size, *tangents.shape[1:])

    init = tree_map(jnp.zeros_like, primals)

    def body(carry, inputs):
        x_batch, t_batch = inputs
        _, f_vjp = vjp(lambda p: apply_fn(p, x_batch), primals)
        partial, = f_vjp(t_batch)
        carry = tree_map(lambda a, b: a + b, carry, partial)
        return carry, None

    result, _ = lax.scan(body, init, (xs, tangents))

    return result

def map_back_to_theta_space(apply_fn, params, params_p, xs, delta_alpha, batch_size=100):
    """
    Map delta alpha back to the theta space via the reparameterization function \phi() in the paper.

    Parameters:
    -----------
    apply_fn:
        callable, apply_fn(params, xs, ys) -> logits.
    params:
        pytree, model parameters.
    params_p:
        linear models:
            pytree, whatever, the Jacobian of linear model does not depend on parameters.
        linearized models:
            pytree, parameters at which the model is linearized.
        nonlinear models:
            pytree, parameters at initialization.
    xs:
        jax.Array of shape (|D|, d_in), feature matrix.
    delta_alpha:
        jax.Array of shape (d_out * |D|, 1), output of influence_on_delta_alpha function

    Returns:
    --------
    params_r:
        pytree (with the same pytree structure as params), retrained model parameters estimated by influence function on delta alpha.
    """

    # Map back to the original theta space via the reparameterization function
    # This version is only viable for small model and small dataset
    # _, f_vjp = vjp(
    #     lambda p: apply_fn(p, xs),
    #     params_p,
    # )
    # delta_theta, = f_vjp(delta_alpha)

    # Map back to the original theta space via the reparameterization function
    delta_theta = _batched_vjp(apply_fn, params_p, xs, delta_alpha, batch_size=batch_size)
    params_r = tree_map(lambda p, dp: p + dp, params, delta_theta)

    # Wait until JAX arrays are fully computed on devices
    # tree_map(lambda x: x.block_until_ready(), params_r)

    return params_r

def influence_on_outputs_theta(apply_fn, params, delta_params, xts):
    """
    Implement a batched version of Equation (2) in the paper.

    Parameters:
    -----------
    apply_fn:
        callable, apply_fn(params, xs, ys) -> logits.
    params:
        pytree, model parameters.
    delta_params:
        pytree, difference between the retrained model parameter and the original model parameters.
    xts:
        jax.Array of shape (|D_t|, d_in), test feature matrix.

    Returns:
    --------
         jax.Array of shape (|D_t|, d_out), outputs difference at all data points in a test set estimated by theta-space
         influence function.
    """

    _, tangents = jvp(lambda p: apply_fn(p, xts), (params,), (delta_params,))

    return tangents

def influence_on_outputs_delta_alpha(delta_alpha, K_Xt_X, shard=False):
    """
    Implement a batched version of Equation (8) in the paper.

    Parameters:
    -----------
    delta_alpha:
        jax.Array of shape (|D|, d_out), difference between the function-space parameters of the retrained model and
        original model.
    K_Xt_X:
        jax.Array of shape (|D_t|, |D|, d_out, d_out) or shape (|D_t|, |D|), the NTK matrix containing kernel value
        between xts and xs. If it is 2D, it is equivalent to K_Xt_X \otimes I_{d_out}.
    shard:
        bool, whether to shard K_Xt_X.

    Returns:
    --------
        jax.Array of shape (|D_t|, d_out), outputs difference at all data points in a test set estimated by delta
        alpha-space influence function.
    """

    Nt, N = K_Xt_X.shape[:2]
    K_ndim = K_Xt_X.ndim
    gpus = jax.devices("gpu")

    if shard:

        mesh = Mesh(np.array(gpus), axis_names=("x",))  # make use of all available GPUs
        v_shard = NamedSharding(mesh, P("x", None))
        sharded_kvp = make_sharded_kvp(mesh, K_ndim)

        num_gpus = mesh.shape['x']
        # Pad K_Xt_X so it shards evenly across multiple GPUs
        Nt_pad = math.ceil(Nt / num_gpus) * num_gpus
        N_pad = math.ceil(N / num_gpus) * num_gpus
        # Shard K_Xt_X across multiple GPUs
        if K_ndim == 2:
            K_shard = NamedSharding(mesh, P("x", None))
            K_Xt_X_shard = jax.device_put(jnp.pad(K_Xt_X, ((0, Nt_pad - Nt), (0, N_pad - N))), K_shard)
        elif K_ndim == 4:
            K_shard = NamedSharding(mesh, P("x", None, None, None))
            K_Xt_X_shard = jax.device_put(jnp.pad(K_Xt_X, ((0, Nt_pad - Nt), (0, N_pad - N), (0, 0), (0, 0))), K_shard)
        else:
            raise ValueError(f"K_Xt_X must be 2D or 4D, got K_ndim={K_ndim}")

        delta_alpha_pad = jax.device_put(jnp.pad(delta_alpha, ((0, N_pad - N), (0, 0))), v_shard)

        return sharded_kvp(delta_alpha_pad, K_Xt_X_shard)[:Nt]

    else:

        kvp = make_kvp(K_ndim)
        delta_alpha = jax.device_put(delta_alpha, gpus[1])
        K_Xt_X_gpu = jax.device_put(K_Xt_X, device=delta_alpha.device)

        return kvp(delta_alpha, K_Xt_X_gpu)

def influence_on_loss_theta(loss_fn, params, params_p, delta_params, xts, yts):
    """
    Implement a batched version of Equation (3) in the paper.

    Parameters:
    -----------
    loss_fn:
        callable, loss_fn(params, params_p, xs, ys) -> loss.
    params:
        pytree, model parameters.
    params_p:
        linear models:
            pytree, all zeros.
        linearized models:
            pytree, parameters at which the model is linearized or all zeros.
        nonlinear models:
            pytree, parameters at initialization.
    delta_params:
        pytree, difference between the retrained model parameter and the original model parameters.
    xts:
        jax.Array of shape (|D_t|, d_in), test feature matrix.
    yts:
        jax.Array of shape (|D_t|, d_out), test target matrix.

    Returns:
    --------
         jax.Array of shape (|D_t|,), loss difference at all data points in a test set estimated by theta-space influence function.
    """

    def influence_on_loss_at_zt(xt, yt):
        """
        Implement Equation (3) in the paper.

        Parameters:
        -----------
        xt:
            jax.Array of shape (d_in,), feature of a test data point.
        yt:
            jax.Array of shape (d_out,), label of a test data point.

        Returns:
        --------
            jax.Array of shape (), loss difference at the data point estimated by theta-space influence function.
        """

        _, tangents = jvp(lambda p: loss_fn(p, params_p, xt[None, ...], yt[None, ...]), (params,), (delta_params,))

        return tangents

    return vmap(influence_on_loss_at_zt, in_axes=(0, 0))(xts, yts)

def influence_on_loss_delta_alpha(loss_fnl, fs, ys, delta_alpha, lam,  K_X_X,
                                  fts=None, yts=None, K_Xt_X=None, shard=False):
    """
    Implement a batched version of Equation (9) in the paper, i.e., Equation (13).

    Parameters:
    -----------
    loss_fnl:
        callable, loss_fnl(f(xs), ys) -> loss.
    fs:
        jax.Array of shape (|D|, d_out), model outputs on the full dataset.
    ys:
        jax.Array of shape (|D|, d_out), target matrix.
    delta_alpha:
        pytree, difference between the function-space parameters of the retrained model and original model, with shape
        (|D|, d_out).
    lam:
        float, regularization constant.
    K_X_X:
        jax.Array of shape (|D|, |D|, d_out, d_out) or shape (|D|, |D|), the NTK matrix containing kernel value between
        xs and xs. If it is 2D, it is equivalent to K_X_X \otimes I_{d_out}.
    fts:
        jax.Array of shape (|D_t|, d_out), model outputs on the test dataset.
    yts:
        jax.Array of shape (|D_t|, d_out), test target matrix.
    K_Xt_X:
        jax.Array of shape (|D_t|, |D|, d_out, d_out) or shape (|D_t|, |D|), the NTK matrix containing kernel value
        between xts and xs. If it is 2D, it is equivalent to K_Xt_X \otimes I_{d_out}.
    shard:
        bool, whether to shard K_X_X and K_Xt_X.

    Returns:
    --------
        jax.Array of shape (|D_t|,), loss difference at all data points in a test set estimated by delta alpha-space influence function.
    """

    if fts is None:
        fts, yts = fs, ys
        K_Xt_X = K_X_X
    N = K_X_X.shape[0]
    Nt = K_Xt_X.shape[0]
    K_ndim = K_Xt_X.ndim
    gpus = jax.devices("gpu")

    alpha_star = -grad(loss_fnl)(fs, ys) / lam
    g = Nt * grad(loss_fnl)(fts, yts)

    if shard:

        mesh = Mesh(np.array(gpus), axis_names=("x",))  # make use of all available GPUs
        v_shard = NamedSharding(mesh, P("x", None))
        sharded_kvp = make_sharded_kvp(mesh, K_ndim)

        num_gpus = mesh.shape['x']
        # Pad K_X_X and K_Xt_X so they shard evenly across multiple GPUs
        N_pad = math.ceil(N / num_gpus) * num_gpus
        Nt_pad = math.ceil(Nt / num_gpus) * num_gpus
        # Shard K_X_X and K_Xt_X across multiple GPUs
        if K_ndim == 2:
            K_shard = NamedSharding(mesh, P("x", None))
            K_X_X_shard = jax.device_put(jnp.pad(K_X_X, ((0, N_pad - N), (0, N_pad - N))), K_shard)
            K_Xt_X_shard = jax.device_put(jnp.pad(K_Xt_X, ((0, Nt_pad - Nt), (0, N_pad - N))), K_shard)
        elif K_ndim == 4:
            K_shard = NamedSharding(mesh, P("x", None, None, None))
            K_X_X_shard = jax.device_put(jnp.pad(K_X_X, ((0, N_pad - N), (0, N_pad - N), (0, 0), (0, 0))), K_shard)
            K_Xt_X_shard = jax.device_put(jnp.pad(K_Xt_X, ((0, Nt_pad - Nt), (0, N_pad - N), (0, 0), (0, 0))), K_shard)
        else:
            raise ValueError(f"K_Xt_X must be 2D or 4D, got K_ndim={K_ndim}")

        delta_alpha_pad = jax.device_put(jnp.pad(delta_alpha, ((0, N_pad - N), (0, 0))), v_shard)
        g_pad = jax.device_put(jnp.pad(g, ((0, N_pad - N), (0, 0))), v_shard)
        alpha_star_pad = jax.device_put(jnp.pad(alpha_star, ((0, N_pad - N), (0, 0))), v_shard)

        a = (g_pad * sharded_kvp(delta_alpha_pad, K_Xt_X_shard)).sum(axis=1)
        b = lam * alpha_star_pad.reshape(-1,) @ sharded_kvp(delta_alpha_pad, K_X_X_shard).reshape(-1,)

    else:

        kvp = make_kvp(K_ndim)
        delta_alpha = jax.device_put(delta_alpha, gpus[1])
        K_Xt_X_gpu = jax.device_put(K_Xt_X, device=delta_alpha.device)
        K_X_X_gpu = jax.device_put(K_X_X, device=delta_alpha.device)
        a = (g * kvp(delta_alpha, K_Xt_X_gpu)).sum(axis=1)
        b = lam * alpha_star.reshape(-1,) @ kvp(delta_alpha, K_X_X_gpu).reshape(-1,)

    return a + b

# Deprecated
########################################################################################################################

# def make_kvp(K_ndim):
#     """
#     Build an NTK matrix-vector product function depend on the shape of K.
#
#     Parameters
#     ----------
#     K_ndim:
#         int, number of dimensions of K
#
#     Returns
#     -------
#     callable
#         kvp(v, K) -> Kv
#     """
#
#     if K_ndim == 2:
#
#         def kvp(v, K_mat):
#             """
#             Compute the NTK matrix-vector product for a 2D NTK matrix.
#
#             Parameters
#             ----------
#             v:
#                 jax.Array of shape (j, d_out).
#             K_mat:
#                 jax.Array of shape (i, j), equivalent to K_math \otimes I_{d_out}.
#
#             Returns
#             -------
#                 jax.Array of shape (i, d_out).
#             """
#             return K_mat @ v
#
#         return kvp
#
#     elif K_ndim == 4:
#
#         def kvp(v, K_mat):
#             """
#             Compute the NTK matrix-vector product for a 4D NTK matrix.
#
#             Parameters
#             ----------
#             v:
#                 jax.Array of shape (j, b).
#             K_mat:
#                 jax.Array of shape (i, j, a, b).
#
#             Returns
#             -------
#                 jax.Array of shape (i, a).
#             """
#             return jnp.einsum("ijab,jb->ia", K_mat, v)
#
#         return kvp
#
#     else:
#         raise ValueError(f"K must be 2D or 4D, got K_ndim={K_ndim}")

# def kvp(v, K):
#     """
#     Compute the NTK matrix vector product with NTK matrix given.
#
#     Parameters:
#     -----------
#     v:
#         jax.Array of shape (j, b), vector to be multiplied by the NTK matrix.
#     K:
#         jax.Array of shape (i, j, a, b), NTK matrix.
#
#     Returns:
#     --------
#         jnp.ndarrady of shape (i, a), NTK matrix-vector product.
#     """
#
#     # block matrix vector product
#     return jnp.einsum("ijab,jb->ia", K, v)
#
# def make_sharded_kvp(mesh):
#     """
#     Build a shard_map NTK matrix-vector product function for a given mesh.
#
#     Parameters
#     ----------
#     mesh:
#         jax.sharding.Mesh, device mesh used for sharding.
#
#     Returns
#     -------
#     callable
#         sharded_kvp(v, K_block) -> K_block v
#     """
#
#     @partial(
#         shard_map,
#         mesh=mesh,
#         in_specs=(P(), P("x", None, None, None)),   # v replicated, K sharded by rows
#         out_specs=P("x", None),                     # output sharded by rows
#         check_rep=False,
#     )
#     def sharded_kvp(v, K_block):
#         """
#         Compute the NTK matrix-vector product between a block of NTK matrix and v on a single GPU.
#
#         Parameters
#         ----------
#         v:
#             jax.Array of shape (j, b), vector to be multiplied by the NTK matrix.
#         K_block:
#             jax.Array of shape (i, j, a, b), a block of NTK matrix.
#
#         Returns
#         -------
#             jax.Array of shape (i, a), NTK matrix-vector product.
#         """
#
#         return jnp.einsum("ijab,jb->ia", K_block, v)
#
#     return sharded_kvp

# def prepare_solve_delta_alpha(
#         loss_fnl,
#         apply_fn,
#         params,
#         params_p,
#         xs,
#         ys,
#         forget_indices,
#         lam,
#         K_X_X,
#         materialize_H_rr=False,
#         shard_K_X_X=False,
#         batch_size=None
# ):
#     """
#     Prepare all components needed to solve Equation (13) in the paper and to map the solution back to theta-space.
#
#     Parameters:
#     -----------
#     loss_fnl:
#         callable, loss_fnl(fs, ys) -> loss.
#     apply_fn:
#         callable, apply_fn(params, xs, ys) -> logits.
#     params:
#         pytree, model parameters.
#     params_p:
#          pytree (with the same structure as params), parameters at which the model is linearized. For linear models, it
#          should be a pytree full of zeros.
#     xs:
#         jax.Array of shape (|D|, d_in), feature matrix.
#     ys:
#         jax.Array of shape (|D|, d_out), target matrix.
#     forget_indices:
#         jax.Array of shape (|D_f|,), indices of forget samples.
#     lam:
#         float, regularization constant.
#     K_X_X:
#         jax.Array of shape (|D|, |D|, d_out, d_out), the NTK matrix.
#     materialize_H_rr
#         bool, default=False, whether to materialize H_rr matrix.
#     shard_K_X_X
#         bool, default=False, whether to shard K_X_X matrix.
#     batch_size
#         int, default=None, the number of columns of K_Xr_Xr processed concurrently when materializing H_rr matrix. A smaller
#         batch_size reduces peak memory usage and shortens the initial compilation time of batched_K_rr_H_rr_vp_fn, at the
#         cost of requiring more batches.
#
#     Returns:
#     --------
#     if materialize_H_rr:
#         H_rr:
#             jax.Array of shape (d_out * |D_r|, |D_r|, d_out), left hand side matrix of Equation (13).
#     else:
#         H_rr_fn:
#             callable, H_rr_fn(v_r) = H_rr v_r, where H_rr is the left hand side matrix of Equation (13).
#     rhs:
#         jax.Array of shape (|D_r|, d_out), right hand side vector of Equation (13).
#     delta_alpha_f:
#         jax.Array of shape (|D_f|, d_out), entries of delta alpha that are known.
#     """
#
#     N, d_out = ys.shape
#     Nf = len(forget_indices)
#     Nr = N - Nf
#     retain_indices = jnp.setdiff1d(jnp.arange(N), forget_indices)
#     xfs, yfs = xs[forget_indices], ys[forget_indices]
#     xrs, yrs = xs[retain_indices], ys[retain_indices]
#
#     loss_fnl_r = lambda f: loss_fnl(f, yrs)
#     frs = apply_fn(params, xrs)
#     hvp_fn = lambda v: hvp(loss_fnl_r, (frs,), (v,))
#
#     alpha_star = -grad(loss_fnl)(apply_fn(params, xs), ys) / lam
#     delta_alpha_f = -alpha_star[forget_indices]
#
#     _, f_vjp = vjp(
#         lambda p: apply_fn(p, xs),
#         params_p,
#     )
#
#     gpus = jax.devices("gpu")
#     mesh = Mesh(np.array(gpus), axis_names=("x",)) # make use of all available GPUs
#
#     if materialize_H_rr:
#
#         if batch_size is None:
#             raise ValueError("batch_size must be set when materialize_H_rr=True")
#
#         # We put K_Xr_Xr and K_Xr_Xf in second GPU because the default GPU is usually occupied with other miscellaneous
#         # stuff like dataset and model parameters.
#         K_Xr_Xr = jax.device_put(K_X_X[jnp.ix_(retain_indices, retain_indices)], device=gpus[1])
#         K_Xr_Xf = jax.device_put(K_X_X[jnp.ix_(retain_indices, forget_indices)], device=gpus[1])
#
#         def batched_K_rr_H_rr_vp_fn(vs, K_rr):
#             def one(v):
#                 return kvp(hvp_fn(v), K_rr)
#             return vmap(one, in_axes=0)(vs)
#
#         def H_rf_vp_fn(v, K_rf, K_rr):
#             K_rf_v = kvp(v, K_rf)
#             return (Nr / N) * (kvp(hvp_fn(K_rf_v), K_rr) + lam * K_rf_v)
#
#         rhs = ((Nf / N) * (kvp(grad(loss_fnl)(apply_fn(params, xfs), yfs), K_Xr_Xf)
#                 + lam * (kvp(alpha_star[forget_indices], K_Xr_Xf) + kvp(alpha_star[retain_indices], K_Xr_Xr)))
#                 - H_rf_vp_fn(delta_alpha_f, K_Xr_Xf, K_Xr_Xr))
#
#         H_rr_dim = d_out * Nr
#
#         if H_rr_dim <= 0:
#
#             vs = K_Xr_Xr.transpose(1, 3, 0, 2).reshape(H_rr_dim, Nr, d_out)
#             # Move H_rr to CPU after computation to reduce GPU memory pressure
#             H_rr = jax.device_get((Nr / N) * (batched_K_rr_H_rr_vp_fn(vs, K_Xr_Xr) + lam * vs))
#
#         else:
#
#             @jit
#             def build_H_rr_block(vs, K_rr):
#                 return (Nr / N) * (batched_K_rr_H_rr_vp_fn(vs, K_rr) + lam * vs)
#
#             reshaped_K_Xr_Xr = K_Xr_Xr.transpose(1, 3, 0, 2).reshape(H_rr_dim, Nr, d_out)
#             H_rr = np.empty((H_rr_dim, Nr, d_out), dtype=np.float32)
#
#             # Materialize H_rr batch by batch and stream each computed block back to CPU to reduce GPU memory pressure.
#             for start in range(0, H_rr_dim, batch_size):
#                 real_end = min(start + batch_size, H_rr_dim)
#                 b = real_end - start
#
#                 vs = reshaped_K_Xr_Xr[start:real_end]  # (batch_size, Nr, d_out)
#                 # Fix the shape of vs to avoid recompiling the jitted function.
#                 if b != batch_size:
#                     vs = jnp.zeros((batch_size, Nr, d_out), dtype=vs.dtype).at[:b].set(vs)
#
#                 H_rr_block = build_H_rr_block(vs, K_Xr_Xr)  # (batch_size, Nr, d_out)
#                 # H_rr[start:real_end] = np.asarray(jax.device_get(H_rr_block[:b]), dtype=np.float32)
#                 H_rr[start:real_end] = jax.device_get(H_rr_block[:b])  # Ignore padded part
#
#         # Wait until JAX arrays are fully computed on devices
#         tree_map(lambda x: x.block_until_ready(), (rhs, delta_alpha_f))
#
#         return H_rr, rhs, f_vjp, delta_alpha_f
#
#     else:
#
#         if shard_K_X_X is None:
#             raise ValueError("shard_K_X_X must be set when materialize_H_rr=False")
#
#         if not shard_K_X_X:
#
#             # We put K_Xr_Xr and K_Xr_Xf in second GPU because the default GPU is usually occupied with other miscellaneous
#             # stuff like dataset and model parameters.
#             K_Xr_Xr = jax.device_put(K_X_X[jnp.ix_(retain_indices, retain_indices)], device=gpus[1])
#             K_Xr_Xf = jax.device_put(K_X_X[jnp.ix_(retain_indices, forget_indices)], device=gpus[1])
#
#             def H_rr_vp_fn(v, K_rr):
#                 K_rr_v = kvp(v, K_rr)[:Nr]
#                 return (Nr / N) * (kvp(hvp_fn(K_rr_v), K_rr)[:Nr] + lam * K_rr_v)
#
#             def H_rf_vp_fn(v, K_rf, K_rr):
#                 K_rf_v = kvp(v, K_rf)
#                 return (Nr / N) * (kvp(hvp_fn(K_rf_v), K_rr) + lam * K_rf_v)
#
#             rhs = ((Nf / N) * (kvp(grad(loss_fnl)(apply_fn(params, xfs), yfs), K_Xr_Xf)
#                     + lam * (kvp(alpha_star[forget_indices], K_Xr_Xf) + kvp(alpha_star[retain_indices], K_Xr_Xr)))
#                     - H_rf_vp_fn(delta_alpha_f, K_Xr_Xf, K_Xr_Xr))
#
#             # Wait until JAX arrays are fully computed on devices
#             tree_map(lambda x: x.block_until_ready(), (rhs, delta_alpha_f))
#
#         else:
#
#             # Shard K_Xr_Xr and K_Xr_Xf and store them in multiple GPUs
#             K_shard = NamedSharding(mesh, P("x", None, None, None))
#
#             # Pad H_rr_dim so it shards evenly across multiple GPUs
#             num_gpus = mesh.shape['x']
#             # Smallest multiple of num_gpus that is >= Nr
#             Nr_pad = math.ceil(Nr / num_gpus) * num_gpus
#             # Shard K_Xr_Xr and K_Xr_Xf across multiple GPUs
#             K_Xr_Xr = jax.device_put(jnp.pad(K_X_X[jnp.ix_(retain_indices, retain_indices)],
#                                              ((0, Nr_pad - Nr), (0, 0), (0, 0), (0, 0))), K_shard)
#             K_Xr_Xf = jax.device_put(jnp.pad(K_X_X[jnp.ix_(retain_indices, forget_indices)],
#                                              ((0, Nr_pad - Nr), (0, 0), (0, 0), (0, 0))), K_shard)
#
#             # Build a NTK matrix vector product function that distributes the computation across multiple GPUs.
#             # sharded_kvp(v, K) -> Kv, where v is replicated across GPUs, K and outputs are sharded across GPUs.
#             sharded_kvp = make_sharded_kvp(mesh)
#
#             def H_rr_vp_fn(v, K_rr):
#                 K_rr_v = sharded_kvp(v, K_rr)[:Nr]
#                 return (Nr / N) * (sharded_kvp(hvp_fn(K_rr_v), K_rr)[:Nr] + lam * K_rr_v)
#
#             def H_rf_vp_fn(v, K_rf, K_rr):
#                 K_rf_v = sharded_kvp(v, K_rf)[:Nr]
#                 return (Nr / N) * (sharded_kvp(hvp_fn(K_rf_v), K_rr)[:Nr] + lam * K_rf_v)
#
#             rhs = ((Nf / N) * (sharded_kvp(grad(loss_fnl)(apply_fn(params, xfs), yfs), K_Xr_Xf)[:Nr]
#                     + lam * (sharded_kvp(alpha_star[forget_indices], K_Xr_Xf)[:Nr] + sharded_kvp(alpha_star[retain_indices], K_Xr_Xr)[:Nr]))
#                     - H_rf_vp_fn(delta_alpha_f, K_Xr_Xf, K_Xr_Xr))
#
#             # Wait until JAX arrays are fully computed on devices
#             tree_map(lambda x: x.block_until_ready(), (rhs, delta_alpha_f))
#
#         return lambda v: H_rr_vp_fn(v, K_Xr_Xr), rhs, f_vjp, delta_alpha_f
#
# def influence_on_delta_alpha(H_rr, rhs, f_vjp, delta_alpha_f, params, xs, forget_indices, maxiter=500, tol=1e-8):
#     """
#     Solve Equation (13) in the paper and map the solution back to theta-space.
#
#     Parameters:
#     -----------
#     if H_rr is jax.Array:
#         H_rr:
#             jax.Array of shape (d_out * |D_r|, |D_r|, d_out), left hand side matrix of Equation (13).
#     else:
#         H_rr:
#             callable, H_rr_fn(v_r) = H_rr v_r, where H_rr is the left hand side matrix of Equation (13).
#     rhs:
#         jax.Array, right hand side vector of Equation (11), with shape (|D_r|, d_out).
#     f_vjp:
#         callable, f_vjp(v) = J^Tv, where J is the Jacobian of apply_fn with respect to parameters, evaluated at params.
#     delta_alpha_f:
#         jax.Array of shape (|D_f|, d_out), entries of delta alpha that are known.
#     params:
#         pytree, model parameters.
#     xs:
#         jax.Array of shape (|D|, d_in), feature matrix.
#     forget_indices:
#         jax.Array of shape (|D_f|,), indices of forget samples.
#     maxiter:
#         int, default=500, maximum number of iterations for conjugate gradient descent solver.
#     tol:
#         float, default=1e-8, tolerance for conjugate gradient descent solver.
#
#     Returns:
#     --------
#     params_r:
#         pytree (with the same structure as params), retrained model parameters estimated by influence function on delta
#         alpha.
#     delta_alpha:
#         jax.Array of shape (d_out * |D|, 1), difference between the function-space parameters of the retrained model
#         and original model.
#     """
#
#     N = xs.shape[0]
#     Nr, d_out = rhs.shape
#     retain_indices = jnp.setdiff1d(jnp.arange(N), forget_indices)
#
#     if isinstance(H_rr, np.ndarray):
#
#         # Put H_rr to the same GPU as rhs
#         H_rr = jax.device_put(H_rr, rhs.device)
#
#         # Solve delta_alpha_r using standard linear system solver, faster but less accurate sometimes
#         # delta_alpha_r = jnp.linalg.solve(H_rr.reshape(Nr * d_out, Nr * d_out), rhs.reshape(Nr * d_out))
#
#         # When H_rr has large condition number, solving H_rr x = b with CG is more accurate than jnp.linalg.solve
#         def mv(v):
#             # H_rr: (Nr*d_out, Nr, d_out)
#             Hv = jnp.einsum("iab,ab->i", H_rr, v)  # (Nr*d_out,)
#             return Hv.reshape((Nr, d_out))
#
#         x0 = jnp.zeros((Nr, d_out))
#         delta_alpha_r, _ = cg(mv, rhs, x0=x0, maxiter=maxiter, tol=tol)
#
#     else:
#
#         x0 = jnp.zeros((Nr, d_out))
#         delta_alpha_r, _ = cg(H_rr, rhs, x0=x0, maxiter=maxiter, tol=tol)
#
#     delta_alpha = jnp.zeros((N, d_out)).at[forget_indices].set(delta_alpha_f)
#     delta_alpha = delta_alpha.at[retain_indices].set(delta_alpha_r.reshape(Nr, d_out))
#
#     # Map back to the original theta space via the reparameterization function
#     delta_theta, = f_vjp(delta_alpha)
#     params_r = tree_map(lambda p, dp: p + dp, params, delta_theta)
#
#     # Wait until JAX arrays are fully computed on devices
#     tree_map(lambda x: x.block_until_ready(), (params_r, delta_alpha))
#
#     return params_r, delta_alpha

# def influence_on_outputs_delta_alpha(delta_alpha, K_Xt_X, shard=False):
#     """
#     Implement the batched version of equation (9) in the paper.
#
#     Parameters:
#     -----------
#     delta_alpha:
#         jax.Array of shape (|D|, d_out), difference between the function-space parameters of the retrained model and
#         original model.
#     K_Xt_X:
#         jax.Array of shape (|D_t|, |D|, d_out, d_out), kernel matrix containing kernel value between xts and xs.
#     shard:
#         bool, whether to shard the NTK matrix.
#
#     Returns:
#     --------
#         jax.Array of shape (|D_t|, d_out), outputs difference at all data points in a test set estimated by delta
#         alpha-space influence function.
#     """
#
#     Nt = K_Xt_X.shape[0]
#
#     if shard:
#
#         gpus = jax.devices("gpu")
#         mesh = Mesh(np.array(gpus), axis_names=("x",))  # make use of all available GPUs
#         sharded_kvp = make_sharded_kvp(mesh)
#
#         num_gpus = mesh.shape['x']
#         # Pad K_Xt_X so it shards evenly across multiple GPUs
#         Nt_pad = math.ceil(Nt / num_gpus) * num_gpus
#         # Shard K_Xt_X across multiple GPUs
#         K_Xt_X_shard = jax.device_put(jnp.pad(K_Xt_X, ((0, Nt_pad - Nt), (0, 0), (0, 0), (0, 0))),
#                                       NamedSharding(mesh, P("x", None, None, None)))
#         # Replicate delta_alpha across multiple GPUs
#         delta_alpha_repl = jax.device_put(delta_alpha, NamedSharding(mesh, P()))
#
#         return sharded_kvp(delta_alpha_repl, K_Xt_X_shard)[:Nt]
#
#     else:
#
#         K_Xt_X_gpu = jax.device_put(K_Xt_X, device=delta_alpha.device)
#
#         return kvp(delta_alpha, K_Xt_X_gpu)

# def influence_on_loss_delta_alpha(loss_fnl, apply_fn, params, delta_alpha, lam, xs, ys, K_X_X,
#                                   xts=None, yts=None, K_Xt_X=None, shard=False):
#     """
#     Implement Equation (11) in the paper.
#
#     Parameters:
#     -----------
#     loss_fnl:
#         callable, loss_fnl(f(xs), ys) -> loss.
#     apply_fn:
#         callable, apply_fn(params, xs, ys) -> logits.
#     params:
#         pytree, model parameters.
#     delta_alpha:
#         pytree, difference between the function-space parameters of the retrained model and original model, with shape
#         (|D|, d_out).
#     lam:
#         float, regularization constant.
#     xs:
#         jax.Array of shape (|D|, d_in), feature matrix.
#     ys:
#         jax.Array of shape (|D|, d_out), target matrix.
#     K_X_X:
#         jax.Array of shape (|D|, |D|, d_out, d_out), kernel matrix containing kernel value between xs and xs.
#     xts:
#         jax.Array of shape (|D_t|, d_in), default=None, test feature matrix.
#     yts:
#         jax.Array of shape (|D_t|, d_out), default=None, test target matrix.
#     K_Xt_X:
#         jax.Array of shape (|D_t|, |D|, d_out, d_out), default=None, kernel matrix containing kernel value between xts and xs.
#     shard:
#         bool, whether to shard the NTK matrix.
#
#     Returns:
#     --------
#         jax.Array of shape (|D_t|,), loss difference at all data points in a test set estimated by delta alpha-space influence function.
#     """
#
#     if xts is None:
#         xts, yts = xs, ys
#         K_Xt_X = K_X_X
#     N = K_X_X.shape[0]
#     Nt = K_Xt_X.shape[0]
#     alpha_star = -grad(loss_fnl)(apply_fn(params, xs), ys) / lam
#     g = Nt * grad(loss_fnl)(apply_fn(params, xts), yts)
#
#     if shard:
#
#         gpus = jax.devices("gpu")
#         mesh = Mesh(np.array(gpus), axis_names=("x",))  # make use of all available GPUs
#         sharded_kvp = make_sharded_kvp(mesh)
#
#         num_gpus = mesh.shape['x']
#         # Pad K_X_X and K_Xt_X so they shard evenly across multiple GPUs
#         N_pad = math.ceil(N / num_gpus) * num_gpus
#         Nt_pad = math.ceil(Nt / num_gpus) * num_gpus
#         # Shard K_X_X and K_Xt_X across multiple GPUs
#         K_X_X_shard = jax.device_put(jnp.pad(K_X_X, ((0, N_pad - N), (0, 0), (0, 0), (0, 0))),
#                                       NamedSharding(mesh, P("x", None, None, None)))
#         K_Xt_X_shard = jax.device_put(jnp.pad(K_Xt_X, ((0, Nt_pad - Nt), (0, 0), (0, 0), (0, 0))),
#                                       NamedSharding(mesh, P("x", None, None, None)))
#         # Replicate delta_alpha across multiple GPUs
#         delta_alpha_repl = jax.device_put(delta_alpha, NamedSharding(mesh, P()))
#
#         a = (g * sharded_kvp(delta_alpha_repl, K_Xt_X_shard)).sum(axis=1)
#         b = lam * alpha_star.reshape(-1, ) @ kvp(delta_alpha_repl, K_X_X_shard).reshape(-1, )
#
#     else:
#
#         K_Xt_X_gpu = jax.device_put(K_Xt_X, device=delta_alpha.device)
#         K_X_X_gpu = jax.device_put(K_X_X, device=delta_alpha.device)
#         a = (g * kvp(delta_alpha, K_Xt_X_gpu)).sum(axis=1)
#         b = lam * alpha_star.reshape(-1, ) @ kvp(delta_alpha, K_X_X_gpu).reshape(-1, )
#
#     return a + b