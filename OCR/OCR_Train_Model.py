import tensorflow as tf
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Rescaling, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.losses import SparseCategoricalCrossentropy
from keras import regularizers


# Load data from train and test sets
def load_data(data_path):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=data_path + "train",
        image_size=(128, 128),
        seed=123,
        batch_size=16,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory=data_path + "val", image_size=(128, 128), seed=123, batch_size=16
    )

    return train_ds, val_ds


train_ds, val_ds = load_data("./Train Character Data/")

class_names = train_ds.class_names

num_classes = len(class_names)

# Define model
model = tf.keras.Sequential(
    [
        Rescaling(1.0 / 255),
        Conv2D(32, 5, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
        MaxPooling2D(pool_size=2, strides=2),
        Conv2D(64, 5, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
        MaxPooling2D(pool_size=2, strides=2),
        Dropout(0.2),
        Conv2D(128, 5, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
        MaxPooling2D(pool_size=2, strides=2),
        # Conv2D(256, 5, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
        # MaxPooling2D(pool_size=2, strides=2),
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.2),
        Dense(100, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ]
)

checkpoint_path = "./Character Recognition Weights/model_on_target_data_8.hdf5"
checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=True,
    monitor="val_accuracy",
    verbose=1,
)

early_stopping = EarlyStopping(
    monitor="val_accuracy", patience=10, verbose=1, mode="max"
)


model.compile(
    optimizer="adam",
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)


model.fit(
    train_ds, validation_data=val_ds, epochs=20, callbacks=[checkpoint, early_stopping]
)
