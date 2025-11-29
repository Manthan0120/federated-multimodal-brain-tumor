# main.py — FINAL VERSION THAT ALWAYS WORKS
import argparse
import subprocess
import sys
from pathlib import Path

def run(cmd):
    print(f"\n→ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=Path(__file__).parent)
    if result.returncode != 0:
        print("FAILED")
        sys.exit(1)
    print("DONE")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", nargs="?", default="all", choices=["all", "clients", "fl", "eval"])
    args = parser.parse_args()

    # Add src to Python path automatically
    import os
    src_path = Path(__file__).parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    if args.task in ["all", "clients"]:
        print("Generating 60 realistic clients...")
        from src.data.simulate_clients import main as gen_clients
        gen_clients()

    if args.task in ["all", "fl"]:
        print("Starting Federated Learning (60 clients)...")
        from src.fl.run_fl import main as run_fl
        run_fl()

    if args.task in ["all", "eval"]:
        print("Running FINAL evaluation (your paper result)...")
        from src.fl.evaluate_final import main as final_eval
        final_eval()

    if args.task == "all":
        print("\nSUCCESS! ALL DONE!")
        print("Your paper-ready result is ready.")
        print("Check: outputs/final_results.json")

if __name__ == "__main__":
    main()