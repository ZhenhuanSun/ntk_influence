import numpy as np

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