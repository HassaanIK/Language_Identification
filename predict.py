from data_cleaning import clean_text
import numpy as np


def predict_language(text, model, cv, le):
    cleaned_text = clean_text(text)
    text_vectorized = cv.transform([cleaned_text])
    prediction = model.predict(text_vectorized)
    predicted_label = le.inverse_transform([np.argmax(prediction)])[0]  # Get the first element of the list
    return predicted_label
