# src/server/server.py — NOW COLLECTS TRAIN + VAL ACCURACY PER ROUND
import flwr as fl
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt

def get_strategy(config):
    out_dir = Path("outputs/learning_curves")
    out_dir.mkdir(parents=True, exist_ok=True)

    head_name = config.get("client", {}).get("local_classifier", "rf")
    timestamp = int(time.time())
    json_path = out_dir / f"metrics_{head_name}_{timestamp}.json"
    csv_path  = out_dir / f"metrics_{head_name}_{timestamp}.csv"
    png_path  = out_dir / f"learning_curve_{head_name}_{timestamp}.png"

    history = []

    def on_evaluate(metrics):
        if not metrics:
            return {}

        # Aggregate val accuracy
        val_accs = [m["val_accuracy"] * n for n, m in metrics]
        train_accs = [m.get("train_accuracy", 0.0) * n for n, m in metrics]
        total_n = sum(n for n, _ in metrics)

        val_acc = sum(val_accs) / total_n if total_n > 0 else 0.0
        train_acc = sum(train_accs) / total_n if total_n > 0 else 0.0

        round_num = len(history) + 1
        entry = {
            "round": round_num,
            "train_accuracy": round(train_acc, 4),
            "val_accuracy": round(val_acc, 4),
            "n_clients": len(metrics)
        }
        history.append(entry)

        # Save JSON + CSV
        with open(json_path, "w") as f:
            json.dump(history, f, indent=2)

        with open(csv_path, "w") as f:
            f.write("round,train_accuracy,val_accuracy,n_clients\n")
            for h in history:
                f.write(f"{h['round']},{h['train_accuracy']},{h['val_accuracy']},{h['n_clients']}\n")

        # Plot both curves
        rounds = [h["round"] for h in history]
        train_curve = [h["train_accuracy"] for h in history]
        val_curve = [h["val_accuracy"] for h in history]

        plt.figure(figsize=(7, 4.5))
        plt.plot(rounds, train_curve, "o-", label="Train Accuracy", color="tab:green")
        plt.plot(rounds, val_curve, "s-", label="Validation Accuracy", color="tab:blue")
        plt.fill_between(rounds, train_curve, val_curve, alpha=0.15, color="gray")
        plt.title(f"Federated Learning Convergence\n{head_name.upper()} heads | {config['fl']['num_clients']} clients")
        plt.xlabel("Communication Round")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0.65, 1.0)
        plt.tight_layout()
        plt.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"Round {round_num:2d} | Clients: {len(metrics):2d} | Train: {train_acc:.4f} | Val: {val_acc:.4f} | Gap: {train_acc - val_acc:.4f}")

        return {"val_accuracy": val_acc}

    return fl.server.strategy.FedAvg(
        fraction_fit=config["fl"].get("fraction_fit", 0.25),
        fraction_evaluate=config["fl"].get("fraction_evaluate", 0.2),
        min_fit_clients=2,
        min_available_clients=config["fl"]["num_clients"],
        evaluate_metrics_aggregation_fn=on_evaluate,
    )