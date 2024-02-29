import tensorflow as tf
from data_splitting import num_classes, input_size


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01), input_shape=(input_size,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(80, activation='relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(50, activation='relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
