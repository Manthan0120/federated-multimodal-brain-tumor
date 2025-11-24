import os
import time

from src.fl.server import get_strategy


def test_learning_curve_filename_contains_head(tmp_path):
    out_dir = str(tmp_path)
    cfg = {
        "learning_curve_dir": out_dir,
        "client": {"local_classifier": "torch_mlp"},
        "fl": {"num_clients": 2, "fraction_fit": 1.0, "rounds": 1, "local_epochs": 1},
    }

    strategy = get_strategy(cfg)

    # call evaluate function to write files
    metrics = [(10, {"val_accuracy": 0.9}), (10, {"val_accuracy": 0.8})]
    # call the strategy's metric aggregator (it should write files into out_dir)
    agg = strategy.evaluate_metrics_aggregation_fn(metrics)

    # allow a short window for ts generated in the code
    files = os.listdir(out_dir)
    found = False
    for f in files:
        if f.startswith("val_accuracy_torch_mlp_") and f.endswith(".json"):
            found = True
            break

    assert found, f"Expected a learning-curve file with 'torch_mlp' in the filename but found {files}"
