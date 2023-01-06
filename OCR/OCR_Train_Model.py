import tensorflow as tf

# Load data from train and test sets
def load_data(data_path):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_path+"train",
        image_size=(128,128),
        seed=123,
        batch_size=16
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_path+"val",
        image_size=(128,128),
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
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
  tf.keras.layers.Conv2D(64, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
  tf.keras.layers.Conv2D(128, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes, activation='softmax')
])


model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
    )


checkpoint_path = "./Character Recognition Weights/model_mixed_2.hdf5"
callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=True,
    verbose=1
    )


model.fit(train_ds, validation_data=val_ds, epochs=5, callbacks=[callback])

