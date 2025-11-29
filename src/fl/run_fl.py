# src/fl/run_fl.py — FINAL WORKING VERSION
import yaml
from pathlib import Path
import flwr as fl
from src.models.client import FLClient
from src.server.server import get_strategy
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def load_config():
    path = Path(__file__).parents[2] / "configs/default.yaml"
    with open(path) as f:
        return yaml.safe_load(f"{f.read()}")

def main():
    config = load_config()
    log.info("Starting Multimodal FL Simulation")
    log.info(f"  Clients: {config['fl']['num_clients']} | Rounds: {config['fl']['rounds']}")

    strategy = get_strategy(config)

    # CORRECT CLIENT MAPPING
    def client_fn(cid: str):
        client_id = int(cid)
        # First half = CT, second half = MRI
        total = config["fl"]["num_clients"]
        if client_id < total // 2:
            modality = "ct"
            local_client_id = client_id
        else:
            modality = "mri"
            local_client_id = client_id - total // 2
        return FLClient(local_client_id, modality, config).to_client()

    hist = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config["fl"]["num_clients"],
        client_resources={"num_cpus": 2, "num_gpus": 0.0},
        config=fl.server.ServerConfig(num_rounds=config["fl"]["rounds"]),
        strategy=strategy,
    )

    log.info("Simulation finished!")

    if "val_accuracy" in hist.metrics_centralized and hist.metrics_centralized["val_accuracy"]:
        final_val_acc = hist.metrics_centralized["val_accuracy"][-1][1]
        log.info(f"FINAL VALIDATION ACCURACY: {final_val_acc:.4f}")
    else:
        log.warning("No validation accuracy recorded!")
        final_val_acc = 0.0

    # Also print train accuracy if available
    if "train_accuracy" in hist.metrics_centralized and hist.metrics_centralized["train_accuracy"]:
        final_train_acc = hist.metrics_centralized["train_accuracy"][-1][1]
        log.info(f"FINAL TRAIN ACCURACY:     {final_train_acc:.4f}")

    log.info("Run 'python main.py eval' to get FINAL TEST accuracy on held-out test set")

if __name__ == "__main__":
    main()