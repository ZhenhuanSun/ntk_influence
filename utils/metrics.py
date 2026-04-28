import jax.numpy as jnp
from jax import jit
from jax.tree_util import tree_map, tree_leaves


def squared_l2_norm(tree):
    """
    Compute the squared l2 norm of a pytree.

    Parameters
    ----------
    tree:
        pytree, model parameters.

    Returns
    -------
    squared_l2_norm:
        jax.Array of shape (), squared l2 norm of tree.
    """

    return jnp.sum(jnp.stack([jnp.sum(x ** 2) for x in tree_leaves(tree)]))

def l2_distance(tree1, tree2):
    """
    Compute the l2 distance between two pytrees.

    Parameters
    ----------
    tree1:
        pytree, model parameters.
    tree2:
        pytree, model parameters.

    Returns
    -------
    l2_distance:
        jax.Array of shape (), l2 distance between two trees.
    """

    diffs = tree_map(lambda a, b: a - b, tree1, tree2)

    return jnp.sqrt(squared_l2_norm(diffs))

def relative_l2_distance(tree1, tree2):
    """
    Compute the relative l2 distance between two pytrees, i.e., ||tree1 - tree2||_2 / ||tree1||_2.

    Parameters
    ----------
    tree1:
        pytree, model parameters.
    tree2:
        pytree, model parameters.

    Returns
    -------
    relative_l2_distance:
        jax.Array of shape (), relative l2 distance between two trees, normalized by the l2 norm of tree1.
    """

    return l2_distance(tree1, tree2) / jnp.sqrt(squared_l2_norm(tree1))

def linf_distance(tree1, tree2):
    """
    Compute the l-infinity distance between two pytrees.

    Parameters
    ----------
    tree1:
        pytree, model parameters.
    tree2:
        pytree, model parameters.

    Returns
    -------
    linf_distance:
        jax.Array of shape (), l-infinity distance between two trees.
    """

    diffs = tree_map(lambda a, b: jnp.abs(a - b), tree1, tree2)

    return jnp.max(jnp.stack([jnp.max(x) for x in tree_leaves(diffs)]))


def make_accuracy_fn(apply_fn):
    """
    Builds a JIT-compiled accuracy function for a given parameterized model.

    Parameters
    ----------
    apply_fn:
        callable, a function with signature apply_fn(params, xs) -> logits.
    Returns
    -------
    accuracy:
        callable, a JIT-compiled function with signature accuracy(params, xs, ys) -> accuracy.
    """

    @jit
    def accuracy(params, xs, ys):
        """
        Compute the classification accuracy for a given parameterized model.

        Parameters
        ----------
        params:
            pytree, model parameters.
        xs:
            jax.Array of shape (|D_t|, d_in), features of all data points in a dataset D_t..
        ys:
            jax.Array of shape (|D_t|, d_out), targets of all data points in a dataset D_t.

        Returns
        -------
        accuracy:
            jax.Array of shape (), classification accuracy.
        """

        logits = apply_fn(params, xs)

        # {-1, 1} encoding
        if ys.shape[1] == 1:
            target_class = ys
            predicted_class = jnp.where(logits >= 0, 1.0, -1.0)
        # One-hot encoding
        else:
            target_class = jnp.argmax(ys, axis=1)
            predicted_class = jnp.argmax(logits, axis=1)

        return jnp.mean(predicted_class == target_class)

    return accuracy

@jit
def accuracy_from_logits(fs, ys):
    """
    Compute classification accuracy from model outputs and labels.

    Parameters
    ----------
    fs:
        jax.Array of shape (|D_t|, d_out), model outputs at all data points in a dataset D_t.
    ys:
        jax.Array of shape (|D_t|, d_out), targets of all data points in a dataset D_t.

    Returns
    -------
    accuracy:
        jax.Array of shape (), classification accuracy.
    """

    # {-1, 1} encoding
    if ys.shape[1] == 1:
        target_class = ys
        predicted_class = jnp.where(fs >= 0, 1.0, -1.0)

    # One-hot encoding
    else:
        target_class = jnp.argmax(ys, axis=1)
        predicted_class = jnp.argmax(fs, axis=1)

    return jnp.mean(predicted_class == target_class)