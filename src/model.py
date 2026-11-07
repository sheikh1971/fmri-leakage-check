"""3D CNN with the architecture described in the paper (Table II).

Input: 64 x 64 x 64 x 1 volumes. Two Conv3D layers (32 and 64 filters, 3x3x3,
ReLU), each followed by 2x2x2 max pooling, then Dense(256, ReLU), dropout 0.5
and a softmax output. About 45 M parameters, most of them in the dense layer.

This is a re-implementation written from the paper's description; the
original training code was not preserved.
"""
from tensorflow.keras import layers, models, optimizers


def build_3d_cnn(input_shape=(64, 64, 64, 1), num_classes=4):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv3D(32, 3, activation="relu"),
        layers.MaxPooling3D(2),
        layers.Conv3D(64, 3, activation="relu"),
        layers.MaxPooling3D(2),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    build_3d_cnn().summary()
