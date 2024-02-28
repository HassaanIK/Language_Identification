# Language Identifier
---

## OVERVIEW
This project is a Flask web application that identifies the language of input text. It uses a machine learning model trained on text data to make predictions. The user inputs text into a form on the web app, and the app returns the predicted language.

## SPECIFICATIONS
- The data used for training is taken from Kaggle. It has 22 different languages.
- The text in the dataset has tokenization, non alphanumeric characters removal and vectorization applied to it.
- The model used for training has 4 layers with 27M params which is enough for getting high accuracy. Complex architectures couldn’t be used because of not sufficient GPUs.
- Techniques like early stopping, learning rate decay and weight decay are used while training to get the most accurate results.
- The metrics used for evaluation is accuracy, 97.89% of which is achieved.
- I usually use Pytorch but this time I used Tensorflow because converting tokens into tensors crashed the GPU constantly.
- The project uses Flask, a lightweight web framework for Python, to create the web application.
- The input text is preprocessed before being fed into the model for prediction.
  
## USAGE

```python
def predict_language(text, model, cv, le):
    cleaned_text = clean_text(text)
    text_vectorized = cv.transform([cleaned_text])
    prediction = model.predict(text_vectorized)
    predicted_label = le.inverse_transform([np.argmax(prediction)])[0]  # Get the first element of the list
    return predicted_label
