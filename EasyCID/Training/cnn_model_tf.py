import tensorflow as tf


def cnn_model(x_train):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(input_shape=(x_train.shape[1], 1), filters=32, kernel_size=(3), strides=(2),
                               padding='SAME', kernel_initializer=tf.keras.initializers.he_normal(), use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool1D(pool_size=(2)),
        tf.keras.layers.Conv1D(input_shape=(x_train.shape[1] / 2, 32), filters=64, kernel_size=(3), strides=(2),
                               padding='SAME', kernel_initializer=tf.keras.initializers.he_normal(), use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool1D(pool_size=(2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu', kernel_initializer=tf.keras.initializers.he_normal()),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])