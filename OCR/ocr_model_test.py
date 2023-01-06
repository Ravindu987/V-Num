import tensorflow as tf
import matplotlib.pyplot as plt


def show_data_sample(train_ds, class_names):
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
        plt.show()


def load_data(data_path):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        "./Train Digit Data split/train",
        # validation_split=0.1,
        image_size=(128,128),
        # subset="training",
        seed=123,
        batch_size=16
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        # data_path,
        "./Train Digit Data split/val",
        # validation_split=0.1,
        image_size=(128,128),
        # subset="validation",
        seed=123,
        batch_size=16
    )

    return train_ds, val_ds

train_ds, val_ds = load_data("./Train Letter Data All/")

class_names = train_ds.class_names

print(class_names)
# show_data_sample(train_ds, class_names)

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x,y: (normalization_layer(x),y))


# AUTOTUNE = tf.data.AUTOTUNE

# train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
  tf.keras.layers.Conv2D(64, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
  tf.keras.layers.Conv2D(128, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
#   tf.keras.layers.Conv2D(32, 3, activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Conv2D(32, 3, activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
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


checkpoint_path = "./CNN letter Dataset/model_mixed_2.hdf5"
callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=True,
    verbose=1
    )


model.fit(train_ds, validation_data=val_ds, epochs=5, callbacks=[callback])

