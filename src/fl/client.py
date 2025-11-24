import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import flwr as fl
import numpy as np
from typing import List

from src.models.local_heads import create_head
from src.data.preprocess import create_dataloaders
from src.models import get_model


class FLClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, modality: str, config, head_name: str = None):
        self.client_id = client_id
        self.modality = modality.upper()
        self.config = config
        # Enable MPS for Apple Silicon
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(modality)

        self.model = get_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        client_cfg = config["client"]
        # choose head from CLI or config
        head_choice = (head_name or client_cfg.get("local_classifier", "random_forest"))
        # create_head handles selecting correct class and reading cfg
        self.local_head = create_head(
            head_choice,
            in_features=self.model.feature_dim,
            cfg=client_cfg,
            device=self.device,
        )
        self.model.set_local_head(self.local_head)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.body.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]):
        state_dict = self.model.body.state_dict()
        new_state_dict = {}
        for (name, _), ndarray in zip(state_dict.items(), parameters):
            new_state_dict[name] = torch.tensor(ndarray).to(self.device)
        self.model.body.load_state_dict(new_state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        optimizer = optim.AdamW(self.model.get_gradient_params(), lr=1e-3)
        epochs = config.get("local_epochs", 3)

        print(f"\nClient {self.client_id} ({self.modality}) | Training {epochs} epochs...")
        for epoch in tqdm(range(epochs), desc=f"C{self.client_id} Train", leave=False):
            for images, labels in self.train_loader:
                images = images.repeat(1, 3, 1, 1)  # Grayscale to RGB
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                features = self.model(images)
                loss = self.criterion(features, labels)
                loss.backward()
                optimizer.step()

        print(f"Client {self.client_id} | Fitting local RF head...")
        self._train_local_rf()

        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def _train_local_rf(self):
        self.model.eval()
        features_list, labels_list = [], []
        with torch.no_grad():
            for images, labels in tqdm(self.train_loader, desc=f"C{self.client_id} RF", leave=False):
                images = images.repeat(1, 3, 1, 1)  # Grayscale to RGB
                images = images.to(self.device)
                feats = self.model.get_features(images)
                features_list.append(feats.cpu())
                labels_list.append(labels)
        features = torch.cat(features_list)
        labels = torch.cat(labels_list)
        self.local_head.fit(features, labels)

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        # Evaluate on validation set
        val_features_list, val_labels_list = [], []
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.repeat(1, 3, 1, 1)  # Grayscale to RGB
                images = images.to(self.device)
                feats = self.model.get_features(images)
                val_features_list.append(feats.cpu())
                val_labels_list.append(labels)

        val_features = torch.cat(val_features_list)
        val_labels = torch.cat(val_labels_list)
        val_preds = self.local_head.predict_class(val_features)
        val_acc = accuracy_score(val_labels.numpy(), val_preds)

        # Evaluate on test set
        test_features_list, test_labels_list = [], []
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.repeat(1, 3, 1, 1)
                images = images.to(self.device)
                feats = self.model.get_features(images)
                test_features_list.append(feats.cpu())
                test_labels_list.append(labels)

        test_features = torch.cat(test_features_list)
        test_labels = torch.cat(test_labels_list)
        test_preds = self.local_head.predict_class(test_features)
        test_acc = accuracy_score(test_labels.numpy(), test_preds)

        # Evaluate on training set as well (report training accuracy)
        train_features_list, train_labels_list = [], []
        with torch.no_grad():
            for images, labels in self.train_loader:
                images = images.repeat(1, 3, 1, 1)
                images = images.to(self.device)
                feats = self.model.get_features(images)
                train_features_list.append(feats.cpu())
                train_labels_list.append(labels)

        train_features = torch.cat(train_features_list)
        train_labels = torch.cat(train_labels_list)
        train_preds = self.local_head.predict_class(train_features)
        train_acc = accuracy_score(train_labels.numpy(), train_preds)

        print(f"Client {self.client_id} ({self.modality}) | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")
        # Return metrics including train accuracy so server can aggregate them
        metrics = {
            "train_accuracy": float(train_acc),
            "val_accuracy": float(val_acc),
            "test_accuracy": float(test_acc),
        }
        return 0.0, len(val_labels), metrics