"""Train a CNN on MNIST and save it as digit_model.keras.

Uses the local mnist.npz in this folder if present (avoids re-downloading).
Run:  python3 train_model.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

MODEL_PATH = "digit_model.keras"
EPOCHS = 5


def load_data():
    # Load mnist.npz directly (Keras 3.x no longer accepts a path argument).
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist.npz")
    path = local if os.path.exists(local) else "mnist.npz"
    with np.load(path) as f:
        X_train, y_train = f["x_train"], f["y_train"]
        X_test, y_test = f["x_test"], f["y_test"]

    # Normalize and add channel dim: (N, 28, 28) -> (N, 28, 28, 1)
    X_train = (X_train / 255.0).astype("float32")[..., np.newaxis]
    X_test = (X_test / 255.0).astype("float32")[..., np.newaxis]
    return X_train, y_train, X_test, y_test


def build_model():
    return keras.Sequential(
        [
            keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )


def main():
    X_train, y_train, X_test, y_test = load_data()

    model = build_model()
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    model.fit(X_train, y_train, epochs=EPOCHS, validation_split=0.1, batch_size=128)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {acc:.4f}  (loss {loss:.4f})")

    model.save(MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
