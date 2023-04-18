import tensorflow as tf
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Rescaling
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.losses import SparseCategoricalCrossentropy


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
        Conv2D(32, 5, activation="relu"),
        MaxPooling2D(pool_size=2, strides=2),
        Conv2D(64, 5, activation="relu"),
        MaxPooling2D(pool_size=2, strides=2),
        Conv2D(128, 5, activation="relu"),
        MaxPooling2D(pool_size=2, strides=2),
        Conv2D(256, 5, activation="relu"),
        MaxPooling2D(pool_size=2, strides=2),
        Flatten(),
        Dense(256, activation="relu"),
        Dense(100, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ]
)

checkpoint_path = "./Character Recognition Weights/model_on_target_data_4.hdf5"
checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=True,
    monitor="val_loss",
    verbose=1,
)

early_stopping = EarlyStopping(monitor="val_loss", patience=2, verbose=1)


model.compile(
    optimizer="adam",
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)


model.fit(
    train_ds, validation_data=val_ds, epochs=15, callbacks=[checkpoint, early_stopping]
)
