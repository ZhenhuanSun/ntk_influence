import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
from utils.metrics import squared_l2_norm


def make_rls_loss_fn(apply_fn, lam=0.0):
    """
    Build a regularized least squares empirical risk for a given parameterized model and regularization constant.

    Parameters
    ----------
    apply_fn:
        callable, apply_fn(params, xs) -> logits.
    lam:
        float, default=0.0, regularization constant.

    Returns
    -------
    loss_fn:
        callable, loss_fn(params, params_p, xs, ys) -> loss.
    """

    def loss_fn(params, params_p, xs, ys):
        """
        Empirical risk with least squares loss and a regularization term that penalizes deviations from params_p.

        Parameters
        ----------
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
            jax.Array of shape (|D|, d_in), input feature matrix.
        ys:
            jax.Array of shape (|D|, d_out), target matrix.

        Returns
        -------
        loss:
            float, regularized least squares empirical risk evaluated at params.
        """

        preds = apply_fn(params, xs)
        sqr_err = (preds - ys) ** 2
        mse = 0.5 * jnp.mean(sqr_err)

        delta_params = tree_map(lambda p, q: p - q, params, params_p)
        reg = 0.5 * lam * squared_l2_norm(delta_params)

        return mse + reg

    return loss_fn

def make_rce_loss_fn(apply_fn, lam=0.0):
    """
    Build a regularized cross entropy empirical risk for a given parameterized model and regularization constant.

    Parameters
    ----------
    apply_fn:
        callable, apply_fn(params, xs) -> logits.
    lam:
        float, default=0.0, regularization constant.

    Returns
    -------
    loss_fn:
        callable, loss_fn(params, params_p, xs, ys) -> loss.
    """

    def loss_fn(params, params_p, xs, ys):
        """
        Empirical risk with cross entropy loss and a regularization term that penalizes deviations from params_p.

        Parameters
        ----------
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
            jax.Array, input feature matrix of shape (|D|, d_in).
        ys:
            jax.Array, target matrix of shape (|D|, d_out).

        Returns
        -------
        loss:
            float, regularized cross entropy empirical risk evaluated at params.
        """

        logits = apply_fn(params, xs)
        labels = jnp.argmax(ys, axis=-1)

        log_probs = jax.nn.log_softmax(logits)
        ce_loss = -jnp.mean(log_probs[jnp.arange(xs.shape[0]), labels])

        delta_params = tree_map(lambda p, q: p - q, params, params_p)
        reg = 0.5 * lam * squared_l2_norm(delta_params)

        return ce_loss + reg

    return loss_fn

def make_upweight_loss_fn(loss_fn, params_p, xs, ys, forget_indices):
    """
    Build an upweighted empirical risk.

    Parameters:
    -----------
    loss_fn:
        callable, loss_fn(params, params_p, xs, ys) -> loss.
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

    Returns:
    --------
    upweight_loss_fn:
        callable, upweight_loss_fn(params) -> loss.
    """

    N = xs.shape[0]
    Nf = len(forget_indices)
    Nr = N - Nf
    retain_indices = jnp.setdiff1d(jnp.arange(N), forget_indices)
    xrs, yrs = xs[retain_indices], ys[retain_indices]

    def upweight_loss_fn(params):
        """
        Upweighted empirical risk L_D(theta, epsilon) in the paper.

        Parameters:
        -----------
        params:
            pytree, model parameters.

        Returns:
        --------
            float, upweighted empirical risk evaluated at params
        """
        # loss_fn(params, params_p, xs, ys) - (Nf / N) * loss_fn(params, params_p, xfs, yfs)
        return (Nr / N) * loss_fn(params, params_p, xrs, yrs)

    return upweight_loss_fn

def difference_in_loss(loss_fn, params, params_r, params_p, xts, yts):
    """
    Compute the loss difference between the retrained and original models at all data points in a test set using loss
    function.

    Parameters:
    -----------
    loss_fn:
        callable, a function with signature loss_fn(params, params_p, xs, ys) -> loss.
    params:
        pytree, model parameters.
    params_r:
        pytree, retrained model parameters.
    params_p:
        linear models:
            pytree, all zeros.
        linearized models:
            pytree, parameters at which the model is linearized or all zeros.
        nonlinear models:
            pytree, parameters at initialization.
    xts:
        jax.Array of shape (|D_t|, d_in), test feature matrix.
    yts:
        jax.Array of shape (|D_t|, d_out), test target matrix.

    Returns:
    --------
         jax.Array of shape (|D_t|,), actual loss difference at all data points in test set.
    """

    def loss_diff_at_zt(xt, yt):
        """
        Compute the loss difference between the retrained and original models at one data point.

        Parameters:
        -----------
        xt:
            jax.Array of shape (d_in,), feature of a test data point.
        yt:
            jax.Array of shape (d_out,), label of a test data point.

        Returns:
        --------
            jax.Array of shape (), actual loss difference at the data point.
        """

        return (
            loss_fn(params_r, params_p, xt[None, ...], yt[None, ...])
            - loss_fn(params, params_p, xt[None, ...], yt[None, ...])
        )

    return jax.vmap(loss_diff_at_zt, in_axes=(0, 0))(xts, yts)

def difference_in_loss_fnl(loss_fnl, fts, fts_r, yts):
    """
    Compute the loss difference between the retrained and original models at all data points in a test set using loss
    functional.

    Parameters
    ----------
    loss_fnl:
        callable, a function with signature loss_fnl(fs, ys) -> scalar.
    fts:
        jax.Array of shape (|D_t|, d_out), original model's outputs at test data points
    fts_r:
        jax.Array of shape (|D_t|, d_out), retrained model's outputs at test data points.
    yts:
        jax.Array of shape (|D_t|, d_out), test target matrix.

    Returns
    -------
        jax.Array of shape (|D_t|,), loss difference at each test data point.
    """

    def loss_diff_at_zt(f, f_r, yt):
        """
        Compute the loss difference between the retrained and original models
        at one test data point.

        Parameters
        ----------
        f:
            jax.Array of shape (d_out,), original model output at one test point.
        f_r:
            jax.Array of shape (d_out,), retrained model output at one test point.
        yt:
            jax.Array of shape (d_out,), label of one test data point.

        Returns
        -------
            jax.Array of shape (), loss difference at the data point.
        """

        return (
            loss_fnl(f_r[None, ...], yt[None, ...])
            - loss_fnl(f[None, ...], yt[None, ...])
        )

    return jax.vmap(loss_diff_at_zt, in_axes=(0, 0, 0))(fts, fts_r, yts)