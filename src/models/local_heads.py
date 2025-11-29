# src/models/local_heads.py
import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from typing import Optional


class RandomForestHead:
    def __init__(self, n_estimators=200, max_depth=None, max_samples=0.6, random_state=42):
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_samples=max_samples,      # ← Critical: prevents overfitting + speeds up
            random_state=random_state,
            n_jobs=-1,
            warm_start=False
        )
        self.is_fitted = False

    def fit(self, features: torch.Tensor, labels: torch.Tensor):
        X = features.detach().cpu().numpy()
        y = labels.cpu().numpy().ravel()

        # Optional: subsample if too big (robustness)
        if len(X) > 5000:
            idx = np.random.choice(len(X), 5000, replace=False)
            X, y = X[idx], y[idx]

        self.rf.fit(X, y)
        self.is_fitted = True

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("Head not fitted")
        X = features.detach().cpu().numpy()
        proba = self.rf.predict_proba(X)[:, 1]
        return torch.from_numpy(proba).float().to(features.device)

    def predict_class(self, features: torch.Tensor) -> torch.Tensor:
        X = features.detach().cpu().numpy()
        pred = self.rf.predict(X)
        return torch.from_numpy(pred).long().to(features.device)


class DecisionTreeHead:
    def __init__(self, max_depth=10, random_state=42):
        self.dt = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        self.is_fitted = False

    def fit(self, features: torch.Tensor, labels: torch.Tensor):
        X = features.detach().cpu().numpy()
        y = labels.cpu().numpy().ravel()
        self.dt.fit(X, y)
        self.is_fitted = True

    def predict_class(self, features: torch.Tensor) -> torch.Tensor:
        X = features.detach().cpu().numpy()
        return torch.from_numpy(self.dt.predict(X)).long().to(features.device)


class SklearnMLPHead:
    def __init__(self, hidden_layer_sizes=(100, 50), max_iter=300, random_state=42):
        self.mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=random_state
        )
        self.is_fitted = False

    def fit(self, features: torch.Tensor, labels: torch.Tensor):
        X = features.detach().cpu().numpy()
        y = labels.cpu().numpy().ravel()
        self.mlp.fit(X, y)
        self.is_fitted = True

    def predict_class(self, features: torch.Tensor) -> torch.Tensor:
        X = features.detach().cpu().numpy()
        return torch.from_numpy(self.mlp.predict(X)).long().to(features.device)


def create_head(name: str, in_features: int = None, cfg: dict = None, device=None):
    cfg = cfg or {}
    name = name.lower()

    if name in ("random_forest", "rf"):
        return RandomForestHead(
            n_estimators=cfg.get("rf_n_estimators", 200),
            max_depth=cfg.get("rf_max_depth", None),
            max_samples=cfg.get("rf_max_samples", 0.6),
        )
    elif name in ("decision_tree", "dt"):
        return DecisionTreeHead(max_depth=cfg.get("dt_max_depth", 12))
    elif name in ("sklearn_mlp", "mlp"):
        return SklearnMLPHead(
            hidden_layer_sizes=cfg.get("mlp_hidden", (100, 50)),
            max_iter=cfg.get("mlp_max_iter", 300)
        )
    else:
        raise ValueError(f"Unknown head: {name}")