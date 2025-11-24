import argparse
import yaml
import logging
from tqdm import tqdm
import time
import subprocess
import sys
import warnings
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)

import flwr as fl
from src.fl.client import FLClient
from src.fl.server import get_strategy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

def load_config(path="configs/default.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def run_simulation(config, head_choice=None):
    log.info("Starting MULTIMODAL FL SIMULATION")
    log.info(f"  Clients: {config['fl']['num_clients']} (CT + MRI)")
    log.info(f"  Rounds: {config['fl']['rounds']}")
    log.info(f"  Local epochs: {config['fl']['local_epochs']}")

    # Start server (pass head choice so server includes it in filenames)
    server_cmd = [sys.executable, "-m", "src.main", "--server"]
    if head_choice:
        server_cmd += ["--head", head_choice]
    server_proc = subprocess.Popen(server_cmd)
    time.sleep(5)

    # Start clients
    client_procs = []
    modalities = ["ct", "ct", "mri", "mri"]
    for i, mod in enumerate(modalities):
        cmd = [
            sys.executable, "-m", "src.main",
            "--client-id", str(i),
            "--modality", mod
        ]
        # include head option so spawned clients use the same head choice
        if head_choice:
            cmd += ["--head", head_choice]
        proc = subprocess.Popen(cmd)
        client_procs.append(proc)
        time.sleep(1)

    # Wait for completion
    for proc in tqdm(client_procs, desc="Clients Running", leave=False):
        proc.wait()

    server_proc.terminate()
    log.info("SIMULATION COMPLETE!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--client-id", type=int)
    parser.add_argument("--modality", choices=["ct", "mri"])
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--head", choices=["random_forest", "decision_tree", "sklearn_mlp", "torch_mlp"], help="Which local classifier head to use")
    args = parser.parse_args()

    config = load_config(args.config)

    os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

    head_choice = args.head or config.get("client", {}).get("local_classifier")
    # Ensure config reflects the head choice so server and other code can read it
    if "client" not in config:
        config["client"] = {}
    config["client"]["local_classifier"] = head_choice
    if args.simulate:
        run_simulation(config, head_choice=head_choice)
    elif args.server:
        strategy = get_strategy(config)
        log.info(f"Server starting | Rounds: {config['fl']['rounds']}")
        fl.server.start_server(
            server_address="0.0.0.0:8090",
            config=fl.server.ServerConfig(num_rounds=config["fl"]["rounds"]),
            strategy=strategy
        )
    else:
        if args.client_id is None or args.modality is None:
            parser.error("--client-id and --modality required")
        client = FLClient(args.client_id, args.modality, config, head_name=head_choice)
        fl.client.start_client(
            server_address="localhost:8090",
            client=client.to_client()
        )

if __name__ == "__main__":
    main()