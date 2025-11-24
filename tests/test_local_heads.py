import torch
import numpy as np

from src.models.local_heads import (
    RandomForestHead,
    DecisionTreeHead,
    SklearnMLPHead,
    TorchMLPHead,
    create_head,
)


def _make_dummy_data(n=100, dim=16):
    X = torch.randn(n, dim)
    # synthetic labels: simple linear separable
    w = torch.randn(dim)
    logits = X @ w
    y = (logits > 0).long()
    return X, y


def test_sklearn_heads_fit_predict():
    X, y = _make_dummy_data(n=100, dim=16)

    # RandomForest
    rf = RandomForestHead(n_estimators=3, max_depth=3)
    rf.fit(X, y)
    p = rf.predict(X)
    assert p.shape[0] == X.shape[0]

    # Decision Tree
    dt = DecisionTreeHead(max_depth=3)
    dt.fit(X, y)
    p = dt.predict(X)
    assert p.shape[0] == X.shape[0]

    # Sklearn MLP
    mlp = SklearnMLPHead(hidden_layer_sizes=(32,), max_iter=100)
    mlp.fit(X, y)
    p = mlp.predict(X)
    assert p.shape[0] == X.shape[0]


def test_torch_mlp_head_fit_predict():
    X, y = _make_dummy_data(n=80, dim=16)

    tmlp = TorchMLPHead(in_features=16, hidden_units=32, epochs=5)
    tmlp.fit(X, y)
    p = tmlp.predict(X)
    assert p.shape[0] == X.shape[0]
    c = tmlp.predict_class(X)
    assert c.shape[0] == X.shape[0]


def test_create_head_factory_returns_correct_type():
    # RF
    h = create_head("random_forest", in_features=16, cfg={"rf_n_estimators": 3, "rf_max_depth": 3})
    assert isinstance(h, RandomForestHead)

    # DT
    h = create_head("decision_tree", in_features=16, cfg={"dt_max_depth": 4})
    assert isinstance(h, DecisionTreeHead)

    # Sklearn MLP
    h = create_head("sklearn_mlp", in_features=16, cfg={"sklearn_mlp_hidden": (16,), "sklearn_mlp_max_iter": 10})
    assert isinstance(h, SklearnMLPHead)

    # Torch MLP
    h = create_head("torch_mlp", in_features=16, cfg={"torch_mlp_hidden_units": 8, "torch_mlp_epochs": 1})
    assert isinstance(h, TorchMLPHead)
