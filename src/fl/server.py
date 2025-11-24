import flwr as fl
from typing import Dict, List, Tuple
import numpy as np
import os
import json
import time
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .strategy import weighted_average  # Keep your existing weighted_average

def get_strategy(config):
    min_fit = int(config["fl"]["num_clients"] * config["fl"]["fraction_fit"])
    # Prepare output directory for learning curves
    out_dir = config.get("learning_curve_dir", os.path.join("outputs", "learning_curves"))
    os.makedirs(out_dir, exist_ok=True)
    # Use a timestamped filename and include which local classifier/head was used
    ts = int(time.time())
    head_name = config.get("client", {}).get("local_classifier", "unknown")
    # sanitize head_name for filenames
    if head_name is None:
        head_name = "unknown"
    safe_head = str(head_name).lower().replace(" ", "_")
    outfile = os.path.join(out_dir, f"val_accuracy_{safe_head}_{ts}.json")

    # Internal list and counter to track per-round aggregated accuracy and per-client metrics
    aggregated_list: List[dict] = []
    round_counter = {"r": 0}

    def evaluate_and_save(metrics: List[tuple]) -> dict:
        # Called by Flower after evaluating clients each round.
        round_counter["r"] += 1
        agg = weighted_average(metrics)

        # Prepare per-client metrics list (num_examples and any reported metrics)
        clients: List[dict] = []
        for num_examples, m in metrics:
            client_entry: dict = {"num_examples": int(num_examples)}
            if isinstance(m, dict):
                for k, v in m.items():
                    try:
                        client_entry[k] = float(v)
                    except Exception:
                        client_entry[k] = v
            clients.append(client_entry)

        # Build entry including all aggregated metrics
        entry = {"round": round_counter["r"], "clients": clients}
        # Copy aggregated metrics into entry (ensure JSON serializable)
        for k, v in agg.items():
            try:
                entry[k] = float(v)
            except Exception:
                entry[k] = v

        aggregated_list.append(entry)

        # Write JSON
        try:
            with open(outfile, "w") as f:
                json.dump(aggregated_list, f, indent=2)
        except Exception:
            pass

        # Write CSV with dynamic aggregated metric columns
        csv_file = os.path.splitext(outfile)[0] + ".csv"
        try:
            # Determine aggregated metric keys (excluding 'round' and 'clients')
            agg_keys = [k for k in aggregated_list[0].keys() if k not in ("round", "clients")]
            header_cols = ["round"] + agg_keys + ["client_accuracies", "client_examples"]
            write_header = not os.path.exists(csv_file)
            with open(csv_file, "a") as f:
                if write_header:
                    f.write(",".join(header_cols) + "\n")
                # Build row values
                client_accs = ";".join([str(c.get("val_accuracy", "")) if "val_accuracy" in c or "accuracy" in c else "" for c in clients])
                client_examples = ";".join([str(c.get("num_examples", "")) for c in clients])
                row_vals = [str(entry.get("round", ""))]
                for k in agg_keys:
                    row_vals.append(str(entry.get(k, "")))
                row_vals.extend([client_accs, client_examples])
                f.write(",".join(row_vals) + "\n")
        except Exception:
            pass

        # Save/overwrite PNG plot of aggregated metrics over rounds
        try:
            rounds = [e["round"] for e in aggregated_list]
            # Identify numeric aggregated metric keys to plot
            plot_keys = [k for k in aggregated_list[0].keys() if k not in ("round", "clients")]
            plt.figure(figsize=(7, 4))
            for k in plot_keys:
                try:
                    vals = [float(e.get(k, float('nan'))) for e in aggregated_list]
                    plt.plot(rounds, vals, marker="o", label=k)
                except Exception:
                    continue
            plt.title("Aggregated Metrics per Round")
            plt.xlabel("Round")
            plt.ylabel("Value")
            plt.grid(True)
            if plot_keys:
                plt.legend()
            png_file = os.path.splitext(outfile)[0] + ".png"
            plt.savefig(png_file, bbox_inches="tight")
            plt.close()
        except Exception:
            pass

        return agg

    return fl.server.strategy.FedAvg(
        fraction_fit=config["fl"]["fraction_fit"],
        fraction_evaluate=1.0,
        min_fit_clients=min_fit,
        min_available_clients=config["fl"]["num_clients"],
        evaluate_metrics_aggregation_fn=evaluate_and_save,
        on_fit_config_fn=lambda server_round: {
            "local_epochs": config["fl"]["local_epochs"]
        },
    )