from abc import ABC

import numpy as np

from lib.load_dataset_raw import EurosatRawDataset

# TODO add Dropout to layers?

dataset = EurosatRawDataset()

print("Dataset: ", dataset.X_train.shape)
NUM_INPUTS_PARAMETERS = dataset.X_train.shape[1]


def bce(y_pred: np.ndarray, y_true: np.ndarray):
    n = len(y_pred)
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

    if len(y_true.shape) == 1:
        # 1D -> row of labels i guess
        correct_confidences = y_pred_clipped[range(n), y_true]
    elif len(y_true.shape) == 2:
        # 2D -> means onehot?
        correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
    else:
        raise ValueError("y_true shape not implemented")

    # neg log likelyhood becuase of simplification of softmax
    return -np.log(correct_confidences)


class LayerDense:
    """Layer Dense Inspired by the implementation in Neural Networks from Scratch"""

    def __init__(self, n_inputs: int, n_neurons: int):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        z = inputs @ self.weights + self.biases

        self.output = z
        return self.output

    def backward(self, dout):
        self.dweights = self.inputs.T @ dout
        self.dbiases = np.sum(dout, axis=0, keepdims=True)
        self.dinputs = dout @ self.weights.T
        return self.dinputs


class ActivationRelu:
    def forward(self, inputs):
        self.mask = inputs > 0
        return np.where(self.mask, inputs, 0)

    def backward(self, dout):
        return dout * self.mask


class LossSoftmaxCrossEntropy:
    """
    Fused: numerically stable softmax + CE. Backward is just
    (probs - one_hot(y)).

    """

    # TODO ai generated i dont fully understand this so have to get back to it
    def forward(self, logits, y_true):
        shift = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shift)
        self.probs = exp / exp.sum(axis=1, keepdims=True)
        self.y_true = y_true
        # average CE over batch
        return -np.log(self.probs[np.arange(len(y_true)), y_true] + 1e-12).mean()

    def backward(self):
        n = len(self.y_true)
        d = self.probs.copy()
        d[np.arange(n), self.y_true] -= 1
        return d / n


class Model:
    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self):
        d = self.loss.backward()
        for layer in reversed(self.layers):
            d = layer.backward(d)

    def params(self):
        return [
            layer for layer in self.layers if hasattr(layer, "weights")
        ]  # TODO watch hasattr if changes are made


class SGD:
    def __init__(self, learning_rate=1e-2):
        self.learning_rate = learning_rate

    def step(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases


def main():

    # Model Architecture as a Model class
    model = Model(
        [
            LayerDense(NUM_INPUTS_PARAMETERS, 256),
            ActivationRelu(),
            LayerDense(256, 128),
            ActivationRelu(),
            LayerDense(128, 10),
        ],
        LossSoftmaxCrossEntropy(),
    )
    opt = SGD(learning_rate=1e-2)

    BATCH_SIZE = 64
    N_EPOCHS = 10

    X_train = dataset.X_train
    y_train = dataset.y_train
    X_test = dataset.X_test
    y_test = dataset.y_test

    def accuracy(logits: np.ndarray, y: np.ndarray) -> float:
        """Fraction of samples where the model's top-class prediction matches y."""
        return float((logits.argmax(axis=1) == y).mean())

    def evaluate(model: Model, X: np.ndarray, y: np.ndarray, batch_size: int = 256):
        """Run a forward-only pass over (X, y) and return mean loss + accuracy."""
        total_loss = 0.0
        total_correct = 0
        for i in range(0, len(X), batch_size):
            Xb, yb = X[i : i + batch_size], y[i : i + batch_size]
            logits = model.forward(Xb)
            loss = model.loss.forward(logits, yb)
            # loss is the batch-mean — weight by batch size to average across the dataset
            total_loss += loss * len(yb)
            total_correct += int((logits.argmax(axis=1) == yb).sum())
        return total_loss / len(X), total_correct / len(X)

    history: list[dict] = []  # collect per-epoch metrics for later plotting

    for epoch in range(N_EPOCHS):
        perm = np.random.permutation(len(X_train))

        running_loss = 0.0
        running_correct = 0
        seen = 0

        for i in range(0, len(perm), BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            Xb, yb = X_train[idx], y_train[idx]

            logits = model.forward(Xb)
            loss = model.loss.forward(logits, yb)
            model.backward()
            for layer in model.params():
                opt.step(layer)

            running_loss += loss * len(yb)
            running_correct += int((logits.argmax(axis=1) == yb).sum())
            seen += len(yb)
            # Live progress: running averages within the epoch.
            print(
                f"\r  epoch {epoch + 1}/{N_EPOCHS}  "
                f"batch {seen}/{len(X_train)}  "
                f"train_loss {running_loss / seen:.4f}  "
                f"train_acc {running_correct / seen:.4f}",
                end="",
            )

        train_loss = running_loss / seen
        train_acc = running_correct / seen
        test_loss, test_acc = evaluate(model, X_test, y_test)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )
        print(
            f"\repoch {epoch + 1}/{N_EPOCHS}  "
            f"train_loss {train_loss:.4f}  train_acc {train_acc:.4f}  "
            f"test_loss {test_loss:.4f}  test_acc {test_acc:.4f}"
        )


if __name__ == "__main__":
    main()
