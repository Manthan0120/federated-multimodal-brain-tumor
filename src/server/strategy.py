from typing import List, Dict, Any
import numpy as np


def fedavg(weights: List[List[np.ndarray]]) -> List[np.ndarray]:
    avg_weights = []
    for weights_list_tuple in zip(*weights):
        layer_weights = np.array(weights_list_tuple)
        avg_weights.append(layer_weights.mean(axis=0))
    return avg_weights


def weighted_average(metrics: List[tuple]) -> Dict[str, Any]:
    """
    Compute weighted average for all numeric metric keys across clients.
    `metrics` is a list of tuples `(num_examples, metrics_dict)`.
    Returns a dict mapping metric_name -> weighted_average.
    """
    # Collect total examples
    total_examples = sum([num for num, _ in metrics])
    if total_examples == 0:
        return {}

    # Gather all metric keys
    keys = set()
    for _, m in metrics:
        if isinstance(m, dict):
            keys.update(m.keys())

    agg: Dict[str, float] = {}
    for k in keys:
        weighted_sum = 0.0
        for num, m in metrics:
            if isinstance(m, dict) and k in m:
                try:
                    weighted_sum += float(m[k]) * float(num)
                except Exception:
                    # Non-numeric metric; skip
                    pass
        agg[k] = weighted_sum / total_examples

    return agg