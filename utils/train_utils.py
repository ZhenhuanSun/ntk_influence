import math
import jax.numpy as jnp
from jax.tree_util import tree_map
from jax import jit, grad, value_and_grad, lax

def make_update_fn(loss_fn, opt_update, get_params):
    """
    Build a JIT-compiled update function for a given empirical risk.

    Parameters
    ----------
    loss_fn:
        callable, loss_fn(params, params_p, xs, ys) -> loss.
    opt_update:
        callable, opt_update(step, grads, opt_state) -> opt_state, optimizer from jax.example_libraries.optimizers
        (e.g., SGD, Momentum, Adam, etc.)
    get_params:
        callable, get_params(opt_state) -> params, function that extracts parameters from an optimizer state.

    Returns
    -------
    update:
        callable, update(step, opt_state, params_p, xs, ys) -> (opt_state, loss, grads).
    """

    @jit
    def update(step, opt_state, params_p, xs, ys):
        """
        Update the model parameters by batch gradient descent.

        Parameters
        ----------
        step:
            int, optimization step index.
        opt_state:
            optimizer state returned by opt_init or opt_update.
        params_p:
            linear models:
                pytree, all zeros.
            linearized models:
                pytree, parameters at which the model is linearized (delete) or all zeros.
            nonlinear models:
                pytree, parameters at initialization.
        xs:
            jax.Array of shape (|D|, d_in), input feature matrix.
        ys:
            jax.Array of shape (|D|, d_out), target matrix.

        Returns
        -------
        opt_state:
            optimizer state returned by opt_init or opt_update.
        loss:
            float, empirical risk evaluated at the current parameters.
        grads:
            pytree, gradients of the empirical risk evaluated at the current parameters.
        """

        params = get_params(opt_state)
        loss, grads = value_and_grad(loss_fn)(params, params_p, xs, ys)
        opt_state = opt_update(step, grads, opt_state)

        return opt_state, loss, grads

    return update

def make_batched_grad(loss_fn, batch_size):
    """
    Build a function that computes the gradient of loss over full dataset in batches. This is used when parallel computation
    on the entire dataset poses memory issues.

    Parameters
    ----------
    loss_fn:
        callable, loss_fn(params, params_p, xs, ys) -> loss.
    batch_size:
        int, default=100, minibatch size used to compute full gradient. It does not necessarily have to be the same as
        the batch size used for training.

    Returns
    -------
    full_grad_fn:
        callable, full_grad_fn(params, params_p, xs, ys) -> grads, function that computes the full gradient of loss.
    """

    def batched_grad(params, params_p, xs, ys):

        N = xs.shape[0]
        num_steps = math.ceil(N / batch_size)
        pad = num_steps * batch_size - N

        # pad to fixed shapes for scan
        xs_pad = jnp.pad(xs, ((0, pad),) + ((0, 0),) * (xs.ndim - 1))
        ys_pad = jnp.pad(ys, ((0, pad),) + ((0, 0),) * (ys.ndim - 1))

        xs_pad = xs_pad.reshape(num_steps, batch_size, *xs.shape[1:])
        ys_pad = ys_pad.reshape(num_steps, batch_size, *ys.shape[1:])

        # mask to recover true batch sizes
        mask = jnp.pad(jnp.ones(N, dtype=jnp.float32), (0, pad))
        mask = mask.reshape(num_steps, batch_size)

        init = tree_map(jnp.zeros_like, params)

        def body(carry, inputs):
            x_b, y_b, m_b = inputs
            Nb = jnp.sum(m_b)

            grads_b = grad(loss_fn)(params, params_p, x_b, y_b)  # mean grad over batch
            grads_b = tree_map(lambda x: Nb * x, grads_b) # sum grad over batch

            carry = tree_map(lambda x, y: x + y, carry, grads_b)
            return carry, None

        grads_sum, _ = lax.scan(body, init, (xs_pad, ys_pad, mask))
        grads_full = tree_map(lambda x: x / N, grads_sum)

        return grads_full

    return batched_grad

def make_batched_apply(apply_fn, batch_size):
    """
    Return a batched version of
                                apply_fn(params, xs) -> logits
    to help reduce the peak memory of evaluating model at multiple data points when model and dataset are large.

    Parameters
    ----------
    apply_fn:
        callable, apply_fn(params, xs) -> logits
    batch_size:
        int, default=100, minibatch size for batched apply_fn.

    Returns
    -------
    batched_apply:
        callable, batched_apply(params, xs) -> logits
    """

    @jit
    def batched_apply(params, xs):
        """
        batched version of apply_fn(params, xs) -> logits

        Parameters:
        -----------
        params:
            pytree, model parameters.
        xs:
            jax.Array of shape (|D|, d_in), where d_in may be a scalar or a tuple, feature matrix.

        Returns:
        --------
        ys:
            jax.Array of shape (|D|, d_out), model output at xs evaluated in batches.
        """

        N = xs.shape[0]
        num_steps = math.ceil(N / batch_size)
        pad = num_steps * batch_size - N

        # Pad to make sure fixed shapes across iterations
        xs_padded = jnp.pad(xs, ((0, pad),) + ((0, 0),) * (xs.ndim - 1))
        xs_batched = xs_padded.reshape(num_steps, batch_size, *xs.shape[1:]) # shape (num_steps, batch_size, d_in)

        def body(_, x_batch):
            y_batch = apply_fn(params, x_batch)
            return None, y_batch

        _, ys_batched = lax.scan(body, None, xs_batched)

        # Merge batch axis
        ys = ys_batched.reshape(num_steps * batch_size, *ys_batched.shape[2:]) # shape: (num_steps * batch_size, d_out)

        # Remove padded outputs
        return ys[:N]

    return batched_apply

def make_kvp(K_ndim):
    """
    Build an NTK matrix-vector product function depend on the shape of K.

    Parameters
    ----------
    K_ndim:
        int, number of dimensions of K

    Returns
    -------
    callable
        kvp(v, K) -> Kv
    """

    if K_ndim == 2:

        def kvp(v, K_mat):
            """
            Compute the NTK matrix-vector product for a 2D NTK matrix.

            Parameters
            ----------
            v:
                jax.Array of shape (j, d_out).
            K_mat:
                jax.Array of shape (i, j), equivalent to K_math \otimes I_{d_out}.

            Returns
            -------
                jax.Array of shape (i, d_out).
            """
            return K_mat @ v

        return kvp

    elif K_ndim == 4:

        def kvp(v, K_mat):
            """
            Compute the NTK matrix-vector product for a 4D NTK matrix.

            Parameters
            ----------
            v:
                jax.Array of shape (j, b).
            K_mat:
                jax.Array of shape (i, j, a, b).

            Returns
            -------
                jax.Array of shape (i, a).
            """
            return jnp.einsum("ijab,jb->ia", K_mat, v)

        return kvp

    else:
        raise ValueError(f"K must be 2D or 4D, got K_ndim={K_ndim}")

def make_kgd_update_fn(loss_fnl, lam, f0=None, active_indices=None):
    """
    Build a JIT-compiled Kernal Gradient Descent (KGD) update function for a given loss functional.

    Parameters
    ----------
    loss_fnl:
        callable, loss_fnl(fs, ys) -> loss.
    lam:
        float, regularization constant.
    f0:
        jax.Array of shape (|D_t|, d_out), default=None, inital model outputs at all data points in a dataset D_t. If None,
        then f0 = 0.
    active_indices:
        jax.Array of shape (|D|,), default=None, indices of the data points whose loss is used to compute the
        gradient. If None, the gradient is computed using all entries of `f` and `t`.

    Returns
    -------
    kgd_update:
        callable, kgd_update(f, t, K, eta) -> f_new, loss, grads
    """

    @jit
    def kgd_update(f, t, K, eta):
        """
        Implement the KGD update rule:
                            f <- f - eta * (K @ grad_f loss + lam * (f - f0))

        Parameters
        ----------
        f:
            jax.Array of shape (|D_t|, d_out), current model outputs at all data points in a dataset D_t, where D_t need
            to be a superset of D_r.
        t:
            jax.Array of shape (|D_r|, d_out), target matrix of the training dataset, where D_r need to be a subset of D.
        K:
            jax.Array of shape (|D_t|, |D_r|, d_out, d_out) or shape (|D_t|, |D_r|), the NTK matrix. If it is 2D, it is
            equivalent to K_X_X \otimes I_{d_out}.
        eta:
            float, learning rate.

        Returns
        -------
        f_new:
            jax.Array of shape (|D_t|, d_out), updated model outputs at all data points in a dataset D_t, where D_t need
            to be a superset of D.
        loss:
            float, empirical risk evaluated at the current model outputs.
        grads:
            jax.Array of shape (|D_t|, d_out), gradients of the empirical risk evaluated at the current model outputs.

        """

        kvp = make_kvp(K.ndim)
        f0_local = jnp.zeros_like(f) if f0 is None else f0

        if active_indices is None:
            g = grad(loss_fnl)(f, t)
            kv = kvp(g, K)
        else:
            g = grad(loss_fnl)(f[active_indices], t[active_indices])
            kv = kvp(g, K)
            g = jnp.zeros_like(f).at[active_indices].set(g)

        update = kv + lam * (f - f0_local)
        return f - eta * update, loss_fnl(f, t), g

    return kgd_update

# Deprecated
# ======================================================================================================================

# def make_update_fn(loss_fn):
#     """
#     Build a JIT-compiled update function for a given empirical risk.
#
#     Parameters
#     ----------
#     loss_fn:
#         callable, a function with signature loss_fn(params, params_p, xs, ys) -> loss.
#
#     Returns
#     -------
#     update:
#         callable, a JIT-compiled function with signature update(params, params_p, xs, ys, eta) -> (new_params, loss, grads).
#     """
#
#     @jit
#     def update(params, params_p, xs, ys, eta):
#         """
#         Update the model parameters by batch gradient descent.
#
#         Parameters
#         ----------
#         params:
#             pytree, model parameters.
#         params_p:
#             pytree, parameters at which the model is linearized. For linear models, it should be a pytree full of zeros.
#         xs:
#             jax.Array, input feature matrix of shape (|D|, d_in).
#         ys:
#             jax.Array, target matrix of shape (|D|, d_out).
#         eta:
#             float, learning rate.
#
#         Returns
#         -------
#         new_params:
#             pytree, updated model parameters.
#         loss:
#             float, empirical risk evaluated at the current parameters.
#         grads:
#             pytree, gradients of the empirical risk evaluated at the current parameters.
#         """
#
#         loss, grads = value_and_grad(loss_fn)(params, params_p, xs, ys)
#         new_params = jax.tree_util.tree_map(lambda p, g: p - eta * g, params, grads)
#
#         return new_params, loss, grads
#
#     return update