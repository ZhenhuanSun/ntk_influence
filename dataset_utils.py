import jax
import jax.numpy as jnp
from jax import random

import numpy as np

import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
from tensorflow.keras import datasets

def one_hot(labels, classes):
    """
    Remap class labels to indices, e.g., {2, 5, 7} -> {0, 1, 2}, and then produces one-hot encodings.

    Parameters
    ----------
    labels:
        jax.Array of shape (num_samples,), target vector
    classes:
        list of int, list of class labels to include in the encoding.

    Returns
    -------
        jax.Array of shape (num_samples, num_classes), one-hot encoded target matrix.
    """

    num_classes = len(classes)
    # remap {class_a, class_b, …} → {0, 1, …, C‑1}
    label_map = {cls: i for i, cls in enumerate(classes)}
    remapped_labels = jnp.array([label_map[int(label)] for label in labels])

    return jax.nn.one_hot(remapped_labels, num_classes)

def binary(labels, classes):
    """
    Map the two classes of a binary classification problem to -1 and 1.

    Parameters
    ----------
    labels:
        jax.Array of shape (num_images,), target vector.
    classes:
        list of int, a list of exactly two class labels.

    Returns
    -------
        jax.Array of shape (num_images, 1), {-1, 1} encoded target matrix.
    """

    # The first class is mapped to -1, and the second class is mapped to +1.
    label_map = {classes[0]: -1, classes[1]: 1}
    labels = jnp.array([label_map[int(label)] for label in labels])

    return labels[:, None]

def build_subset(key, images, labels, classes, samples_per_class=None):
    """
    Build a subset of a dataset by selecting samples from specified classes.

    Parameters
    ----------
    key:
        jax PRNGKey, random key for shuffling and sampling.
    images:
        jax.Array of shape (num_samples, height, width, num_channels), image matrix.
    labels:
        jax.Array of shape (num_samples,), target vector.
    classes:
        list of int, list of class labels to include in the subset.
    samples_per_class:
        list of int or None, default=None, number of samples to take per class. It must have the same length as classes.
        If None or if the requested number exceeds the available samples, all samples for that class are included.

    Returns
    -------
    images_subset:
        jax.Array of shape (num_selected_samples, height, width, num_channels), image submatrix.
    labels_subset:
        jax.Array of shape (num_selected_samples,), target vector.
    """

    if len(classes) != len(samples_per_class):
        raise ValueError("`classes` and `samples_per_class` must have the same length.")

    selected_indices = []
    key_seq = random.split(key, len(classes) + 1)  # +1 for final shuffle
    num_pixels = int(jnp.prod(jnp.asarray(images.shape[1:])))

    if samples_per_class is None:
        # Take all samples for every class
        samples_per_class = [None] * len(classes)

    for cls, k, num_selected in zip(classes, key_seq[:-1], samples_per_class):
        cls_indices = jnp.where(labels == cls)[0]

        if (num_selected is None) or (num_selected >= len(cls_indices)):
            selected = cls_indices # select all of them
        else:
            permutation = random.permutation(k, len(cls_indices))
            selected = cls_indices[permutation[:num_selected]]

        selected_indices.append(selected)

    all_selected_indices = jnp.concatenate(selected_indices)
    # Final shuffle so classes are mixed
    all_selected_indices = all_selected_indices[random.permutation(key_seq[-1], len(all_selected_indices))]

    images_subset = images[all_selected_indices]
    labels_subset = labels[all_selected_indices]

    # images_subset.reshape(images_subset.shape[0], -1)
    return images_subset, labels_subset


def build_retain_and_forget(key, images, labels, classes, samples_per_class=None):
    """
    Build retain and forget datasets from the full dataset by randomly removing a specified number of samples per class.

    Parameters
    ----------
    key:
        jax PRNGKey, random key for sampling indices to remove.
    images:
        jax.Array of shape (num_samples, height, width, num_channels), image matrix.
    labels:
        jax.Array of shape (num_samples,), target vector.
    classes:
        list of int, list of class labels from which samples are removed.
    samples_per_class:
        list of int, number of samples to remove per class. It must have the same length as classes. Use 0 for any class
        from which no samples are removed.

    Returns
    -------
    retain_indices:
        jax.Array of shape (num_retain,), indices of retain samples.
    retain_images:
        jax.Array of shape (num_retain, height, width, num_channels), retain image matrix.
    retain_labels:
        jax.Array of shape (num_retain,), retain target vector.
    forget_indices:
        jax.Array of shape (num_forget,), indices of forget samples.
    forget_images:
        jax.Array of shape (num_forget, height, width, num_channels), forget image matrix.
    forget_labels:
        jax.Array of shape (num_forget,), forget target vector.
    """

    if len(classes) != len(samples_per_class):
        raise ValueError("`classes` and `samples_per_class` must have the same length.")

    num_samples = len(labels)
    forget_mask = jnp.zeros(num_samples, dtype=bool)
    key_seq = random.split(key, len(classes))

    for cls, k, num_remove in zip(classes, key_seq, samples_per_class):
        if num_remove == 0:
            continue

        cls_indices = jnp.where(labels == cls)[0]
        if num_remove > len(cls_indices):
            raise ValueError(f"Requested {num_remove} removals from class {cls}, "
                             f"but only {cls_indices.size} available.")

        permutation = random.permutation(k, len(cls_indices))
        forget_indices = cls_indices[permutation[:num_remove]]
        forget_mask = forget_mask.at[forget_indices].set(True)

    retain_mask = ~forget_mask
    retain_images = images[retain_mask]
    retain_labels = labels[retain_mask]
    forget_images = images[forget_mask]
    forget_labels = labels[forget_mask]

    retain_indices = jnp.where(retain_mask)[0]
    forget_indices = jnp.where(forget_mask)[0]

    return (retain_indices, retain_images, retain_labels), (forget_indices, forget_images, forget_labels)

def build_retain_and_forget_by_percentage(key, images, labels, classes, forget_percents):
    """
    Build retain and forget datasets by randomly removing a specified percentage of samples from each selected class.

    Parameters
    ----------
    key:
        jax PRNGKey, random key for sampling indices to remove.
    images:
        jax.Array of shape (num_samples, height, width, num_channels), image matrix.
    labels:
        jax.Array of shape (num_samples,), target vector.
    classes:
        list of int or None, classes from which samples are removed.
            - If None, use all unique classes in `labels`.
    forget_percents:
        float/int or list of float/int in [0, 100], percentages of samples to remove from each selected class.
            - If forget_percents is a scalar, the same forget percentage is applied to every class in `classes`.
            - If forget_percents is a list, it must have the same length as `classes`.

    Returns
    -------
    (retain_indices, retain_images, retain_labels):
        retain subset.
    (forget_indices, forget_images, forget_labels):
        forget subset.
    """

    num_samples = labels.shape[0]

    if classes is None:
        classes = jnp.unique(labels)
    else:
        classes = jnp.asarray(classes)

    num_classes = len(classes)

    if jnp.isscalar(forget_percents):
        forget_percents = [float(forget_percents)] * num_classes
    else:
        forget_percents = list(forget_percents)
        if len(forget_percents) != num_classes:
            raise ValueError(
                "`forget_percents` must be a scalar or have the same length as `classes`."
            )

    for p in forget_percents:
        if not (0 <= p <= 100):
            raise ValueError("Each entry of `forget_percents` must be in [0, 100].")

    forget_mask = jnp.zeros(num_samples, dtype=bool)
    key_seq = random.split(key, num_classes)

    for cls, k, pct in zip(classes, key_seq, forget_percents):
        if pct == 0:
            continue

        cls_indices = jnp.where(labels == cls)[0]
        cls_count = cls_indices.shape[0]
        num_remove = int(round((pct / 100.0) * cls_count))

        if num_remove == 0:
            continue

        permutation = random.permutation(k, cls_count)
        cls_forget_indices = cls_indices[permutation[:num_remove]]
        forget_mask = forget_mask.at[cls_forget_indices].set(True)

    retain_mask = ~forget_mask

    retain_images = images[retain_mask]
    retain_labels = labels[retain_mask]
    forget_images = images[forget_mask]
    forget_labels = labels[forget_mask]

    retain_indices = jnp.where(retain_mask)[0]
    forget_indices = jnp.where(forget_mask)[0]

    return (
        (retain_indices, retain_images, retain_labels),
        (forget_indices, forget_images, forget_labels),
    )

def load_dataset_normalized(dataset_name):
    """
    Load dataset train/test splits and normalize images to [0, 1].

    Supported dataset: {"mnist", "cifar10"}

    Returns
    -------
    train_images:
        np.ndarray of shape (50000, 28, 28) or (50000, 32, 32, 3), normalized train images.
    train_labels:
        np.ndarray of shape (50000,), train labels.
    test_images
        np.ndarray of shape (10000, 28, 28) or (10000, 32, 32, 3), normalized test images.
    test_labels:
        np.ndarray of shape (10000,), test labels.
    """

    if dataset_name == "mnist":
        (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    elif dataset_name == "cifar10":
        (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    else:
        raise ValueError(f"Unsupported dataset_name={dataset_name!r}. Use 'mnist' or 'cifar10'.")

    # Normalize pixel values to be between 0 and 1
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0

    # Flatten target vector
    train_labels = train_labels.astype(np.float32).flatten()
    test_labels = test_labels.astype(np.float32).flatten()

    return train_images, train_labels, test_images, test_labels

def build_datasets(dataset_name, dataset_cfg, key):
    """
    Build a subset of `dataset_name` dataset based on information given in dataset_cfg.

    Parameters
    ----------
    dataset_name:
        str, name of dataset.
    dataset_cfg:
        dict, dataset configuration dictionary specifying how many samples to draw per class, e.g.
        {
          "train_per_class": {3: 100, 8: 100},
          "test_per_class": {3: 1000, 8: 1000},
        }
    key:
        jax PRNGKey, random key used to sample the subsets.

    Returns
    -------
    A dict with keys:
        train_images:
            jax.Array of shape (train_per_class * num_classes, height, width, num_channels), a subset of `dataset_name` training dataset
        train_labels:
            if len(classes) == 2:
                jax.Array of shape (train_per_class * num_classes, 1), binary encoded labels for train_images
            else:
                jax.Array of shape (train_per_class * num_classes, num_classes), one-hot encoded labels for train_images.
        test_images:
            jax.Array of shape (test_per_class * num_classes, height, width, num_channels), a subset of `dataset_name` test dataset
        test_labels:
            if len(classes) == 2:
                jax.Array of shape (test_per_class * num_classes, 1), binary encoded labels for test_images
            else:
                jax.Array of shape (test_per_class * num_classes, num_classes), one-hot encoded labels for test_images.
        classes:
            list[int], sorted class labels used.
    """

    train_images, train_labels, test_images, test_labels = load_dataset_normalized(dataset_name)

    train_per_class = dict(dataset_cfg["train_per_class"])
    test_per_class = dict(dataset_cfg["test_per_class"])

    classes = sorted(train_per_class.keys())
    if set(test_per_class.keys()) != set(classes):
        raise ValueError("train_per_class and test_per_class must have same class keys")

    key, train_key, test_key = random.split(key, 3)

    # X_train, y_train = build_subset(
    #     train_key,
    #     train_images,
    #     train_labels,
    #     classes=classes,
    #     samples_per_class=[int(train_per_class[c]) for c in classes],
    # )
    # X_test, y_test = build_subset(
    #     test_key,
    #     test_images,
    #     test_labels,
    #     classes=classes,
    #     samples_per_class=[int(test_per_class[c]) for c in classes],
    # )

    X_train, y_train = build_subset(
        train_key,
        train_images,
        train_labels,
        classes=classes,
        samples_per_class=[train_per_class[c] for c in classes],
    )
    X_test, y_test = build_subset(
        test_key,
        test_images,
        test_labels,
        classes=classes,
        samples_per_class=[test_per_class[c] for c in classes],
    )

    return {
        "train_images": X_train,
        "train_labels": y_train,
        "test_images": X_test,
        "test_labels": y_test,
        "classes": classes,
    }

# Deprecated
########################################################################################################################
# def build_retain_and_forget_by_percentage(key, images, labels, forget_percent):
#     """
#     Randomly split a dataset into retain/forget subsets by forgetting a given percentage of the full dataset.
#
#     Parameters
#     ----------
#     key:
#         jax PRNGKey, random key for sampling indices to remove.
#     images:
#         jax.Array of shape (num_samples, height, width, num_channels), image matrix.
#     labels:
#         jax.Array of shape (num_samples,), target vector.
#     forget_percent:
#         float or int in [0, 100], percentage of the total samples to be forgotten. For example, 50 means forget 50% of
#         the dataset.
#
#     Returns
#     -------
#     retain_indices:
#         jax.Array of shape (num_retain,), indices of retain samples.
#     retain_images:
#         jax.Array of shape (num_retain, height, width, num_channels), retain image matrix.
#     retain_labels:
#         jax.Array of shape (num_retain,), retain target vector.
#     forget_indices:
#         jax.Array of shape (num_forget,), indices of forget samples.
#     forget_images:
#         jax.Array of shape (num_forget, height, width, num_channels), forget image matrix.
#     forget_labels:
#         jax.Array of shape (num_forget,), forget target vector.
#     """
#
#     if not (0 <= forget_percent <= 100):
#         raise ValueError("`forget_percent` must be in [0, 100].")
#
#     num_samples = labels.shape[0]
#     num_forget = int(round((forget_percent / 100.0) * num_samples))
#
#     perm = random.permutation(key, num_samples)
#     forget_indices = perm[:num_forget]
#
#     forget_mask = jnp.zeros(num_samples, dtype=bool).at[forget_indices].set(True)
#     retain_mask = ~forget_mask
#
#     retain_images = images[retain_mask]
#     retain_labels = labels[retain_mask]
#     forget_images = images[forget_mask]
#     forget_labels = labels[forget_mask]
#
#     # Indices in original dataset order
#     retain_indices = jnp.where(retain_mask)[0]
#     forget_indices = jnp.where(forget_mask)[0]
#
#     return (retain_indices, retain_images, retain_labels), (forget_indices, forget_images, forget_labels)