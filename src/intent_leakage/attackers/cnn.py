"""Optional real 1D convolutional traffic-sequence attacker.

PyTorch is imported only when this attacker is fitted or queried.  If the
optional dependency is missing, this module raises a clear error and does not
substitute a dummy or a different model.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import OptionalDependencyError, TraceAttacker, stable_label_order, validate_labels


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised where torch is absent
        raise OptionalDependencyError(
            "CNN1DTraceAttacker requires the optional 'torch' dependency"
        ) from error
    return torch


def _sequence_tensor(X: Any) -> np.ndarray:
    values = np.asarray(X, dtype=float)
    if values.ndim == 2:
        values = values[:, :, None]
    if values.ndim != 3 or values.shape[0] == 0 or values.shape[1] == 0 or values.shape[2] == 0:
        raise ValueError("X must have shape (samples, length) or (samples, length, features)")
    if not np.all(np.isfinite(values)):
        raise ValueError("sequence features must all be finite")
    return values.astype(np.float32, copy=False)


class CNN1DTraceAttacker(TraceAttacker):
    """Small configurable 1D CNN over signed-size/inter-arrival sequences.

    Widths, epochs, optimizer settings and early stopping are engineering
    configuration because the paper intentionally leaves them for validation.
    CPU is the default to avoid requiring a GPU and improve reproducibility.
    """

    def __init__(
        self,
        *,
        hidden_channels: tuple[int, int] = (16, 32),
        kernel_size: int = 3,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 32,
        epochs: int = 20,
        patience: int | None = 5,
        random_state: int = 0,
        device: str = "cpu",
    ) -> None:
        if len(hidden_channels) != 2 or any(channel <= 0 for channel in hidden_channels):
            raise ValueError("hidden_channels must contain two positive widths")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if learning_rate <= 0.0 or weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative")
        if batch_size <= 0 or epochs <= 0:
            raise ValueError("batch_size and epochs must be positive")
        if patience is not None and patience <= 0:
            raise ValueError("patience must be positive or None")
        self.hidden_channels = tuple(int(value) for value in hidden_channels)
        self.kernel_size = int(kernel_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.patience = patience
        self.random_state = int(random_state)
        self.device = str(device)

    def _encode_labels(self, labels: np.ndarray) -> np.ndarray:
        result = np.empty(labels.size, dtype=np.int64)
        for index, label in enumerate(labels.tolist()):
            matches = np.flatnonzero(self.classes_ == label)
            if matches.size == 0:
                raise ValueError(f"validation label was absent from training data: {label!r}")
            result[index] = int(matches[0])
        return result

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        X_validation: Any | None = None,
        y_validation: Any | None = None,
    ) -> "CNN1DTraceAttacker":
        torch = _torch()
        nn = torch.nn
        values = _sequence_tensor(X)
        labels = validate_labels(y, n_samples=values.shape[0])
        self.classes_ = stable_label_order(labels)
        encoded = self._encode_labels(labels)
        self.n_sequence_features_ = values.shape[2]
        self.sequence_length_ = values.shape[1]

        if (X_validation is None) != (y_validation is None):
            raise ValueError("X_validation and y_validation must be supplied together")
        validation_values: np.ndarray | None = None
        validation_encoded: np.ndarray | None = None
        if X_validation is not None:
            validation_values = _sequence_tensor(X_validation)
            if validation_values.shape[1:] != values.shape[1:]:
                raise ValueError("training and validation sequence shapes must match")
            validation_labels = validate_labels(
                y_validation, n_samples=validation_values.shape[0]
            )
            validation_encoded = self._encode_labels(validation_labels)

        torch.manual_seed(self.random_state)
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)
        device = torch.device(self.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but unavailable")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS device requested but unavailable")

        first_width, second_width = self.hidden_channels
        model = nn.Sequential(
            nn.Conv1d(
                self.n_sequence_features_,
                first_width,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                first_width,
                second_width,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(second_width, self.classes_.size),
        ).to(device)

        inputs = torch.as_tensor(np.transpose(values, (0, 2, 1)), dtype=torch.float32)
        targets = torch.as_tensor(encoded, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(inputs, targets)
        generator = torch.Generator().manual_seed(self.random_state)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=True,
            generator=generator,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        loss_function = nn.CrossEntropyLoss()
        best_state: dict[str, Any] | None = None
        best_validation_loss = np.inf
        epochs_without_improvement = 0
        history: list[dict[str, float]] = []

        for epoch in range(self.epochs):
            model.train()
            total_loss = 0.0
            total_examples = 0
            for batch_inputs, batch_targets in loader:
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_inputs)
                loss = loss_function(logits, batch_targets)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * batch_targets.shape[0]
                total_examples += batch_targets.shape[0]
            record = {"epoch": float(epoch + 1), "train_loss": total_loss / total_examples}

            if validation_values is not None and validation_encoded is not None:
                model.eval()
                with torch.no_grad():
                    validation_inputs = torch.as_tensor(
                        np.transpose(validation_values, (0, 2, 1)),
                        dtype=torch.float32,
                        device=device,
                    )
                    validation_targets = torch.as_tensor(
                        validation_encoded, dtype=torch.long, device=device
                    )
                    validation_loss = float(
                        loss_function(model(validation_inputs), validation_targets).cpu()
                    )
                record["validation_loss"] = validation_loss
                if validation_loss < best_validation_loss - 1e-12:
                    best_validation_loss = validation_loss
                    best_state = {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in model.state_dict().items()
                    }
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if self.patience is not None and epochs_without_improvement >= self.patience:
                        history.append(record)
                        break
            history.append(record)

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model_ = model
        self.device_ = device
        self.training_history_ = tuple(history)
        return self

    def _prepare_predict(self, X: Any) -> Any:
        if not hasattr(self, "model_"):
            raise RuntimeError("attacker must be fitted before prediction")
        values = _sequence_tensor(X)
        if values.shape[1] != self.sequence_length_ or values.shape[2] != self.n_sequence_features_:
            raise ValueError(
                "prediction sequence length and feature count must match training data"
            )
        torch = _torch()
        return torch.as_tensor(
            np.transpose(values, (0, 2, 1)), dtype=torch.float32, device=self.device_
        )

    def predict_proba(self, X: Any) -> np.ndarray:
        torch = _torch()
        inputs = self._prepare_predict(X)
        self.model_.eval()
        with torch.no_grad():
            probabilities = torch.softmax(self.model_(inputs), dim=1)
        return probabilities.cpu().numpy()

    def predict(self, X: Any) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


# Descriptive alias used by the paper-to-code map.
CNNTraceAttacker = CNN1DTraceAttacker
