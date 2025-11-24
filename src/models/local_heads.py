import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import torch


class RandomForestHead:
    def __init__(self, n_estimators=100, max_depth=None):
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=1
        )
        self.is_fitted = False

    def fit(self, features: torch.Tensor, labels: torch.Tensor):
        X = features.detach().cpu().numpy()
        y = labels.cpu().numpy()
        self.rf.fit(X, y)
        self.is_fitted = True

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("RF head not fitted.")
        X = features.detach().cpu().numpy()
        proba = self.rf.predict_proba(X)[:, 1]
        return torch.from_numpy(proba).to(features.device)

    def predict_class(self, features: torch.Tensor) -> torch.Tensor:
        X = features.detach().cpu().numpy()
        pred = self.rf.predict(X)
        return torch.from_numpy(pred).to(features.device)


class DecisionTreeHead:
    """Simple Decision Tree head wrapper around sklearn.tree.DecisionTreeClassifier

    Methods mirror RandomForestHead: fit, predict (probabilities), predict_class
    """

    def __init__(self, max_depth=None):
        self.dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        self.is_fitted = False

    def fit(self, features: torch.Tensor, labels: torch.Tensor):
        X = features.detach().cpu().numpy()
        y = labels.cpu().numpy()
        self.dt.fit(X, y)
        self.is_fitted = True

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("Decision tree head not fitted.")
        X = features.detach().cpu().numpy()
        # predict_proba returns shape (n_samples, n_classes)
        proba = self.dt.predict_proba(X)[:, 1]
        return torch.from_numpy(proba).to(features.device)

    def predict_class(self, features: torch.Tensor) -> torch.Tensor:
        X = features.detach().cpu().numpy()
        pred = self.dt.predict(X)
        return torch.from_numpy(pred).to(features.device)


class SklearnMLPHead:
    """Wrapper for sklearn's MLPClassifier for a lightweight neural head.

    This head uses scikit-learn's MLPClassifier (good for small feature vectors)
    and exposes fit, predict and predict_class methods consistent with other heads.
    """

    def __init__(self, hidden_layer_sizes=(100,), max_iter=200, random_state=42):
        # Ensure probability estimates are available
        self.clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
        )
        self.is_fitted = False

    def fit(self, features: torch.Tensor, labels: torch.Tensor):
        X = features.detach().cpu().numpy()
        y = labels.cpu().numpy()
        self.clf.fit(X, y)
        self.is_fitted = True

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("Sklearn MLP head not fitted.")
        X = features.detach().cpu().numpy()
        proba = self.clf.predict_proba(X)[:, 1]
        return torch.from_numpy(proba).to(features.device)

    def predict_class(self, features: torch.Tensor) -> torch.Tensor:
        X = features.detach().cpu().numpy()
        pred = self.clf.predict(X)
        return torch.from_numpy(pred).to(features.device)


class TorchMLPHead:
    """A small PyTorch MLP head useful as a trainable local head.

    Provides a fit loop (basic), predict (probabilities) and predict_class.
    This is useful when you want a trainable NN head unlike scikit-learn models.
    """

    def __init__(self, in_features, hidden_units=64, lr=1e-3, epochs=10, device=None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_units),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_units, 1),
            torch.nn.Sigmoid(),
        ).to(self.device)

        self.epochs = epochs
        self.lr = lr
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.is_fitted = False

    def fit(self, features: torch.Tensor, labels: torch.Tensor):
        X = features.to(self.device).float()
        y = labels.to(self.device).float().view(-1, 1)

        self.model.train()
        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            out = self.model(X)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()

        self.is_fitted = True

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("Torch MLP head not fitted.")
        self.model.eval()
        with torch.no_grad():
            X = features.to(self.device).float()
            out = self.model(X).view(-1)
        return out.to(features.device)

    def predict_class(self, features: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        proba = self.predict(features)
        return (proba >= threshold).long()


def create_head(name: str, in_features: int = None, cfg: dict = None, device=None):
    """Factory to create a local classifier head by name.

    name: one of ['random_forest', 'decision_tree', 'sklearn_mlp', 'torch_mlp']
    in_features: required for torch_mlp
    cfg: optional configuration dict (client section from config)
    device: torch device for TorchMLPHead
    """
    cfg = cfg or {}
    name = (name or "random_forest").lower()
    if name in ("random_forest", "rf"):
        return RandomForestHead(
            n_estimators=cfg.get("rf_n_estimators", 100),
            max_depth=cfg.get("rf_max_depth", None),
        )
    elif name in ("decision_tree", "dt"):
        return DecisionTreeHead(max_depth=cfg.get("dt_max_depth", None))
    elif name in ("sklearn_mlp", "sklearn", "mlp_sk"):
        return SklearnMLPHead(
            hidden_layer_sizes=tuple(cfg.get("sklearn_mlp_hidden", (100,))),
            max_iter=cfg.get("sklearn_mlp_max_iter", 200),
            random_state=cfg.get("random_state", 42),
        )
    elif name in ("torch_mlp", "torch", "tmlp"):
        if in_features is None:
            raise ValueError("in_features must be provided for torch_mlp head")
        return TorchMLPHead(
            in_features=in_features,
            hidden_units=cfg.get("torch_mlp_hidden_units", 64),
            lr=cfg.get("torch_mlp_lr", 1e-3),
            epochs=cfg.get("torch_mlp_epochs", 10),
            device=device,
        )
    else:
        raise ValueError(f"Unknown head name: {name}")