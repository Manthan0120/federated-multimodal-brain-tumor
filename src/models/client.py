# src/models/client.py — FINAL FIXED VERSION (WORKS WITH ALL HEADS)
import torch
from sklearn.metrics import accuracy_score
import flwr as fl
from pathlib import Path

from src.models.multimodal import MultimodalFLModel
from src.models.local_heads import create_head
from src.data.preprocess import create_dataloaders


class FLClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, modality: str, config):
        self.client_id = client_id
        self.modality = modality.lower()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load data
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(client_id, self.modality)

        # Shared frozen backbone
        self.model = MultimodalFLModel(config).to(self.device)
        self.model.eval()

        # Local head (RF, MLP, etc.)
        head_name = config["client"].get("local_classifier", "random_forest")
        self.local_head = create_head(
            name=head_name,
            in_features=self.model.feature_dim,
            cfg=config.get("client", {}),
            device=self.device
        )
        self.model.set_local_head(self.local_head)

        # === CRITICAL FIX: Universal fitted flag that works for ALL heads ===
        self.local_head.is_fitted = False  # We add this attribute ourselves

        # Feature cache
        self.features_cache_dir = Path("outputs/features_cache")
        self.features_cache_dir.mkdir(parents=True, exist_ok=True)
        self.train_features_path = self.features_cache_dir / f"client_{client_id}_{modality}_train.pt"
        self.val_features_path   = self.features_cache_dir / f"client_{client_id}_{modality}_val.pt"
        self.test_features_path  = self.features_cache_dir / f"client_{client_id}_{modality}_test.pt"

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.body.state_dict().values()]

    def set_parameters(self, parameters):
        if not parameters:
            return
        state_dict = self.model.body.state_dict()
        new_state_dict = {}
        for (name, _), ndarray in zip(state_dict.items(), parameters):
            new_state_dict[name] = torch.tensor(ndarray, device=self.device)
        self.model.body.load_state_dict(new_state_dict, strict=True)

    def fit(self, parameters, config):
        """Receive global backbone — train local head ONLY ONCE (first round)"""
        self.set_parameters(parameters)

        # Train local head only once, the very first time
        if not self.local_head.is_fitted:
            print(f"Client {self.client_id} ({self.modality.upper()}) → Training local head (first round)")
            train_feats, train_labels = self._extract_features(self.train_loader)
            self.local_head.fit(train_feats, train_labels)
            self.local_head.is_fitted = True  # Mark as permanently trained

        num_examples = len(self.train_loader.dataset)
        return self.get_parameters({}), num_examples, {}

    def evaluate(self, parameters, config):
        """Use the SAME trained local head — just update backbone"""
        self.set_parameters(parameters)

        # Extract fresh features using the newest global backbone
        train_feats, train_labels = self._extract_features(self.train_loader)
        val_feats,   val_labels   = self._extract_features(self.val_loader)

        # Safety net (should never trigger)
        if not self.local_head.is_fitted:
            print(f"WARNING: Client {self.client_id} head not fitted — fixing now")
            self.local_head.fit(train_feats, train_labels)
            self.local_head.is_fitted = True

        with torch.no_grad():
            train_preds = self.local_head.predict_class(train_feats)
            val_preds   = self.local_head.predict_class(val_feats)

            train_acc = accuracy_score(train_labels.cpu().numpy(), train_preds.cpu().numpy())
            val_acc   = accuracy_score(val_labels.cpu().numpy(),   val_preds.cpu().numpy())

        gap = train_acc - val_acc
        print(f"Client {self.client_id:2d} ({self.modality.upper():3s}) | "
              f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | Gap: {gap:.4f}")

        return 0.0, len(val_labels), {
            "val_accuracy": float(val_acc),
            "train_accuracy": float(train_acc)
        }

    def _extract_features(self, loader):
        features_list = []
        labels_list = []
        self.model.eval()
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                feats = self.model.get_features(images)
                features_list.append(feats.cpu())
                labels_list.append(labels)
        return torch.cat(features_list), torch.cat(labels_list)

    def save_features_for_final_eval(self):
        train_f, train_l = self._extract_features(self.train_loader)
        val_f,   val_l   = self._extract_features(self.val_loader)
        test_f,  test_l  = self._extract_features(self.test_loader)

        torch.save({"feats": train_f, "labels": train_l}, self.train_features_path)
        torch.save({"feats": val_f,   "labels": val_l},   self.val_features_path)
        torch.save({"feats": test_f,  "labels": test_l},  self.test_features_path)
        print(f"Saved cached features for client {self.client_id} ({self.modality})")