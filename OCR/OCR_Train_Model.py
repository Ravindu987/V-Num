import tensorflow as tf
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Rescaling
from keras.callbacks import ModelCheckpoint
from keras.losses import SparseCategoricalCrossentropy


# Load data from train and test sets
def load_data(data_path):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_path+"train",
        image_size=(128, 128),
        seed=123,
        batch_size=16
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_path+"val",
        image_size=(128, 128),
        seed=123,
        batch_size=16
    )

    return train_ds, val_ds


train_ds, val_ds = load_data("./Train Character Data/")

class_names = train_ds.class_names

print(class_names)
# show_data_sample(train_ds, class_names)

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

num_classes = len(class_names)

# Define model
model = tf.keras.Sequential([
    Rescaling(1./255),
    Conv2D(32, 3, activation='relu'),
    MaxPooling2D(pool_size=2, strides=2),
    Conv2D(64, 3, activation='relu'),
    MaxPooling2D(pool_size=2, strides=2),
    Conv2D(128, 3, activation='relu'),
    MaxPooling2D(pool_size=2, strides=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])


model.compile(
    optimizer='adam',
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)


checkpoint_path = "./Character Recognition Weights/model_mixed_3.hdf5"
callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=True,
    verbose=1
)


model.fit(train_ds, validation_data=val_ds, epochs=5, callbacks=[callback])
