from jax import jit, random
import neural_tangents as nt
from neural_tangents import stax

def build_linear_model(model_cfg, key, d_in, d_out):
    """
    Build random ReLU features extractor + linear model.

    The model has structure: x -> Dense(num_rrf) -> ReLU -> Dense(d_out), where the first Dense+ReLU
    acts as a fixed random feature extractor and the final Dense is the trainable layer.

    Parameters
    ----------
    model_cfg:
        dict, model configuration dictionary with keys:
            num_rrf:
                int, number of random ReLU features to extract.
            W_std:
                float, default=2.0, standard deviation used to initialize Conv/Dense weights.
            b_std:
                float, default=0.1, standard deviation used to initialize Conv/Dense biases.
            parameterization:
                str, default="ntk", stax parameterization.
            linearize:
                bool, whether to linearize the model or not. This key is not used for linear model since linear models are
                already linearized.
        Example:
            {"num_rrf": 20000, "W_std": 2.0, "b_std": 0.1, "parameterization": "ntk", "linearize": True}
    key:
        jax PRNGKey, random key for initializing random Relu features extractor and linear model.
    d_in:
        int, input dimension of random feature extractor.
    d_out:
        int, output dimension of linear model.

    Returns
    -------
        A dict with keys:
            params_fixed:
                pytree, parameters of the fixed random feature extractor (first Dense layer + ReLU).
            extract_rrf:
                callable, extract_rrf(params_fixed, xs) -> random ReLU features of xs.
            params_0:
                pytree, initial parameters of the trainable output Dense layer.
            apply_fn:
                callable, apply_fn(params, xs) -> logits, linear model.
            kernel_fn:
                callable, kernel_fn(x1, x2, 'ntk') -> ntk value between x1 and x2, infinitely wide linear model induced by
                the given model architecture.
    """

    num_rrf = int(model_cfg["num_rrf"])
    W_std = float(model_cfg.get("W_std", 2.0))
    b_std = float(model_cfg.get("b_std", 0.1))
    parameterization = str(model_cfg.get("parameterization", "ntk"))

    init_rrf, extract_rrf, _ = stax.serial(
        stax.Dense(num_rrf, W_std=W_std, b_std=b_std, parameterization=parameterization),
        stax.Relu(),
    )

    key, k_rrf, k_model = random.split(key, 3)
    rrf_out_shape, params_fixed = init_rrf(k_rrf, (-1, d_in))
    extract_rrf = jit(extract_rrf)

    init_fn, apply_fn, kernel_fn = stax.Dense(d_out, W_std=W_std, b_std=b_std, parameterization=parameterization)
    _, params_0 = init_fn(k_model, rrf_out_shape)
    apply_fn = jit(apply_fn)

    return {
        "params_fixed": params_fixed,
        "extract_rrf": extract_rrf,
        "params_0": params_0,
        "apply_fn": apply_fn,
        "kernel_fn": kernel_fn,
    }

def build_fcnn_model(model_cfg, key, d_in, d_out):
    """
        Build a fully connected neural network model with ReLU activations.

        The model has structure:
            x -> Dense(width_1) -> ReLU
              -> ...
              -> Dense(width_L) -> ReLU
              -> Dense(d_out)

        Parameters
        ----------
        model_cfg:
            dict with keys:
                hidden_widths:
                    list[int], widths of hidden layers, e.g. [512, 512]
                W_std:
                    float, default=2.0, standard deviation used to initialize Conv/Dense weights.
                b_std:
                    float, default=0.1, standard deviation used to initialize Conv/Dense biases.
                parameterization:
                    str, default="ntk", stax parameterization.
                linearize:
                    bool, whether to linearize the model or not.
            Example:
                {"hidden_widths": [512, 512], "W_std": 2.0, "b_std": 0.1, "parameterization": "ntk", "linearize": True}
        key:
            jax PRNGKey, random key for initializing parameters of fcnn.
        d_in:
            int, input dimension of fcnn.
        d_out:
            int, output dimension of fcnn.

        Returns
        -------
        dict with keys:
            params_0:
                pytree, initial parameters of the fcnn.
            apply_fn:
                callable, apply_fn(params, xs) -> logits, fcnn model or linearized fcnn model around params_0
            kernel_fn:
                callable, kernel_fn(x1, x2, 'ntk') -> ntk value between x1 and x2, infinitely wide fcnn model induced by
                the given model architecture.
        """

    hidden_widths = list(model_cfg["hidden_widths"])
    W_std = float(model_cfg.get("W_std", 2.0))
    b_std = float(model_cfg.get("b_std", 0.1))
    parameterization = str(model_cfg.get("parameterization", "ntk"))
    linearize = bool(model_cfg["linearize"])

    # Build FCNN layers
    layers = []
    for w in hidden_widths:
        layers += [
            stax.Dense(int(w), W_std=W_std, b_std=b_std, parameterization=parameterization),
            stax.Relu(),
        ]
    layers += [stax.Dense(d_out, W_std=W_std, b_std=b_std, parameterization=parameterization)]

    init_fn, apply_fn, kernel_fn = stax.serial(*layers)

    key, k_init = random.split(key)
    _, params_0 = init_fn(k_init, (-1, int(d_in)))

    if linearize:
        apply_fn = jit(nt.linearize(apply_fn, params_0))
    else:
        apply_fn = jit(apply_fn)

    return {
        "params_0": params_0,
        "apply_fn": apply_fn,
        "kernel_fn": kernel_fn,
    }

def build_cnn_model(model_cfg, key, d_in, d_out):
    """
    Build a convolutional neural network model with ReLU activations and average pooling.

    The model has structure:
        x -> Conv(c1) -> ReLU -> AvgPool
          -> Conv(c2) -> ReLU -> AvgPool
          -> ...
          -> Flatten -> Dense(d_out)

    Parameters
    ----------
    model_cfg:
        dict with keys:
            channels:
                list[int], number of output channels for each convolution layer, e.g., [128, 128, 128].
            kernel_size:
                tuple[int, int], convolution kernel size, e.g., (3, 3).
            pool_window:
                tuple[int, int], default=(2, 2), pooling window shape.
            pool_strides:
                tuple[int, int], default=(2, 2), pooling strides.
            conv_padding:
                str, default="SAME", padding used in convolution layers.
            pool_padding:
                str, default="SAME" padding used in pooling layers.
            W_std:
                float, default=2.0, standard deviation used to initialize Conv/Dense weights.
            b_std:
                float, default=0.1, standard deviation used to initialize Conv/Dense biases.
            parameterization:
                str, default="ntk", stax parameterization.
            linearize:
                bool, whether to linearize the model around params_0.
        Example:
            {
                "channels": [128, 128, 128],
                "kernel_size": (3, 3),
                "pool_window": (2, 2),
                "pool_strides": (2, 2),
                "conv_padding": "SAME",
                "pool_padding": "SAME",
                "W_std": 2.0,
                "b_std": 0.1,
                "parameterization": "ntk",
                "linearize": True,
            }
    key:
        jax PRNGKey, random key for initializing parameters of cnn.
    d_in:
        tuple[int, int, int], input dimension of cnn, e.g. (32, 32, 3).
    d_out:
        int, output dimension of cnn.

    Returns
    -------
    dict with keys:
        params_0:
            pytree, initial parameters of the cnn.
        apply_fn:
            callable, apply_fn(params, xs) -> logits, cnn model or linearized cnn model around params_0.
        kernel_fn:
            callable, kernel_fn(x1, x2, 'ntk') -> ntk value between x1 and x2, infinitely wide cnn model induced by
            the given model architecture.
    """

    channels = list(model_cfg["channels"])
    kernel_size = tuple(model_cfg["kernel_size"])
    pool_window = tuple(model_cfg.get("pool_window", (2, 2)))
    pool_strides = tuple(model_cfg.get("pool_strides", (2, 2)))
    conv_padding = str(model_cfg.get("conv_padding", "SAME"))
    pool_padding = str(model_cfg.get("pool_padding", "SAME"))
    W_std = float(model_cfg.get("W_std", 2.0))
    b_std = float(model_cfg.get("b_std", 0.1))
    parameterization = str(model_cfg.get("parameterization", "ntk"))
    linearize = bool(model_cfg["linearize"])

    # Build CNN layers
    layers = []
    for c in channels:
        layers += [
            stax.Conv(
                int(c),
                kernel_size,
                padding=conv_padding,
                W_std=W_std,
                b_std=b_std,
                parameterization=parameterization,
            ),
            stax.Relu(),
            stax.AvgPool(
                pool_window,
                strides=pool_strides,
                padding=pool_padding,
            ),
        ]

    layers += [
        stax.Flatten(),
        stax.Dense(
            int(d_out),
            W_std=W_std,
            b_std=b_std,
            parameterization=parameterization,
        ),
    ]

    init_fn, apply_fn, kernel_fn = stax.serial(*layers)

    key, k_init = random.split(key)
    _, params_0 = init_fn(k_init, (-1, *d_in))

    if linearize:
        apply_fn = jit(nt.linearize(apply_fn, params_0))
    else:
        apply_fn = jit(apply_fn)

    return {
        "params_0": params_0,
        "apply_fn": apply_fn,
        "kernel_fn": kernel_fn,
    }