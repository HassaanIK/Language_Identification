from flask import Flask, request, render_template, jsonify
from predict import predict_language
import joblib
import tensorflow as tf
import h5py

model = tf.keras.models.load_model('models\\full_language_identifcation_modelf.h5')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
CountVectorizer = joblib.load('models\\cv.joblib')
LabelEncoder = joblib.load('models\\le.joblib')


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        prediction = predict_language(text, model, CountVectorizer, LabelEncoder)  # Call your prediction function
        return render_template('result.html', prediction=prediction, text=text)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)