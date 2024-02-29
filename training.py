from model import model
from data_splitting import num_classes, X_train, X_val, input_size
from data_preparing import y_train_encoded, y_val_encoded
from model_callbacks import optimizer, early_stopping, lr_scheduler_callback

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(X_train, y_train_encoded, epochs=10, batch_size=32, validation_data=(X_val, y_val_encoded), callbacks=[early_stopping,lr_scheduler_callback])

model.save('full_language_identifcation_model1.h5')