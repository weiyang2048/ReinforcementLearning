import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from .models import DNNModel
from .utils import calc_ks_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


def train_eval_pytorch_dnn(
    X_train, y_train, X_val, y_val, epochs=5, batch_size=64, lr=1e-3, device="cpu"
):
    """
    Train and evaluate a PyTorch DNN model.

    Args:
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        X_val (array-like): Validation features.
        y_val (array-like): Validation labels.
        epochs (int, optional): Number of training epochs. Defaults to 5.
        batch_size (int, optional): Batch size for training. Defaults to 64.
        lr (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
        device (str, optional): Device to use for training ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        tuple: Contains:
            - model: The trained PyTorch model.
            - auc (float): Area under ROC curve score on validation set.
            - ks (float): KS statistic score on validation set.
            - val_preds (array): Predicted probabilities on validation set.
    """
    model = DNNModel(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)

    dataset_size = len(X_train_t)
    n_batches = (dataset_size // batch_size) + 1

    for epoch in range(epochs):
        indices = torch.randperm(dataset_size)
        X_train_t = X_train_t[indices]
        y_train_t = y_train_t[indices]

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            if start_idx >= dataset_size:
                break

            x_batch = X_train_t[start_idx:end_idx]
            y_batch = y_train_t[start_idx:end_idx]

            preds = model(x_batch)
            loss = criterion(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        val_preds = model(X_val_t).cpu().numpy().ravel()

    auc = roc_auc_score(y_val, val_preds)
    ks = calc_ks_score(y_val, val_preds)
    return model, auc, ks, val_preds


def train_eval_model(model_name, X_train, y_train, X_val, y_val, device="cpu"):
    """
    Train and evaluate a model based on the specified model name.

    Args:
        model_name (str): Name of the model to train. Must be one of: "xgb", "lgbm", "rf", "dnn".
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        X_val (array-like): Validation features.
        y_val (array-like): Validation labels.
        device (str, optional): Device to use for DNN training. Defaults to "cpu".

    Returns:
        tuple: Contains:
            - model: The trained model object.
            - auc (float): Area under ROC curve score on validation set.
            - ks (float): KS statistic score on validation set.
            - y_prob (array): Predicted probabilities on validation set.

    Raises:
        ValueError: If model_name is not one of the supported models.
    """
    if model_name == "xgb":
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        ks = calc_ks_score(y_val, y_prob)
        return model, auc, ks, y_prob

    elif model_name == "lgbm":
        model = LGBMClassifier()
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        ks = calc_ks_score(y_val, y_prob)
        return model, auc, ks, y_prob

    elif model_name == "rf":
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        ks = calc_ks_score(y_val, y_prob)
        return model, auc, ks, y_prob

    elif model_name == "dnn":
        model, auc, ks, y_prob = train_eval_pytorch_dnn(
            X_train.values, y_train.values, X_val.values, y_val.values, device=device
        )
        return model, auc, ks, y_prob

    else:
        raise ValueError(f"Unknown model name: {model_name}")
