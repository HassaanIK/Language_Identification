import tensorflow as tf


def lr_scheduler(epoch, lr):
    if epoch < 3:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
    

early_stopping = tf.keras.callbacksEarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
lr_scheduler_callback = tf.keras.callbacksLearningRateScheduler(lr_scheduler)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)