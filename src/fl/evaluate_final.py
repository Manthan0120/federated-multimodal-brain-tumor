# src/fl/evaluate_final.py — FINAL VERSION THAT SAVES TRAIN/VAL/TEST
import torch
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from src.models.multimodal import MultimodalFLModel
from src.models.local_heads import create_head
from src.data.preprocess import load_client_data
import yaml
import pandas as pd

def load_config():
    path = Path(__file__).parents[2] / "default.yaml"
    with open(path) as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load final global model
    model_path = Path("outputs/final_global_model.pt")
    if not model_path.exists():
        print("Run FL first! Missing final_global_model.pt")
        return

    state_dict = torch.load(model_path, map_location=device)
    model = MultimodalFLModel(config).to(device)
    model.body.load_state_dict(state_dict)
    model.eval()

    results = []
    print("Computing train / val / test accuracy per client...")

    for modality in ["ct", "mri"]:
        pkl_path = Path("data/processed_clients") / modality / "clients.pkl"
        with open(pkl_path, "rb") as f:
            clients = pickle.load(f)

        for client_idx, df in enumerate(tqdm(clients, desc=f"{modality.upper()} clients")):
            if len(df) < 30:
                continue

            # Split: 70% train, 15% val, 15% test
            n = len(df)
            train_df = df.iloc[:int(0.70*n)]
            val_df   = df.iloc[int(0.70*n):int(0.85*n)]
            test_df  = df.iloc[int(0.85*n):]

            head = create_head(config["client"]["local_classifier"], cfg=config.get("client", {}))

            # Helper to get features + labels
            def get_features_labels(subset_df):
                feats, labels = [], []
                for path, label in zip(subset_df["filepath"], subset_df["label_id"]):
                    try:
                        from PIL import Image
                        from torchvision import transforms
                        img = Image.open(path).convert("RGB")
                        img = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])(img).unsqueeze(0).to(device)

                        with torch.no_grad():
                            feat = model.get_features(img)
                        feats.append(feat.cpu())
                        labels.append(label)
                    except:
                        continue
                return torch.cat(feats), torch.tensor(labels)

            # Train on train set
            train_feats, train_labels = get_features = get_features_labels(train_df)
            head.fit(train_feats, train_labels)

            # Evaluate on all three splits
            for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
                if len(split_df) == 0: continue
                feats, labels = get_features_labels(split_df)
                preds = head.predict_class(feats)
                acc = accuracy_score(labels.numpy(), preds.cpu().numpy())

                results.append({
                    "client_id": f"{modality}_{client_idx}",
                    "modality": modality.upper(),
                    "split": split_name,
                    "accuracy": round(acc, 4),
                    "n_samples": len(split_df)
                })

    # Save everything
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    df_results = pd.DataFrame(results)
    csv_path = out_dir / "train_val_test_per_client.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"Saved detailed results → {csv_path}")

    # Summary
    test_acc = df_results[df_results["split"] == "test"]["accuracy"].mean()
    print(f"\nFINAL TEST ACCURACY (average over all clients): {test_acc:.4f}")

    summary = {
        "final_test_accuracy": round(float(test_acc), 4),
        "n_clients": len(df_results["client_id"].unique()),
        "results_csv": str(csv_path)
    }
    with open(out_dir / "final_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("All done! Use outputs/train_val_test_per_client.csv for any plot you want.")

if __name__ == "__main__":
    main()