# Language Identification Web App

### Overview
This project is a Flask web application that identifies the language of input text. It uses a machine learning model trained on text data to make predictions. The user inputs text into a form on the web app, and the app returns the predicted language.

### Steps
- Data Collection: The data used for training is taken from [Kaggle]([URL](https://www.kaggle.com/datasets/zarajamshaid/language-identification-datasst)). It has 22 different languages.
- Data Preprocessing: The input text is preprocessed to remove non alphanumeric characters, its tokenized and vectorized using `word_tokenize` and `CountVectorizer` from `nltk` and `scikit-learn`.
- Model Training: The model used for training has 4 layers with 27M parameters. Complex architectures couldn’t be used because of insufficient GPUs.
- Model Evaluation: The metrics for evaluation of the model used is accuracy, 97.89% of which is achieved.
- Web App Development: The Flask web application is developed to take user input and display the predicted language.

### Techniques Used
- Tokenization: Splitting the input text into tokens.
- Vectorization: Converting tokens into its vectors.
- Model Training: Using a machine learning model to learn patterns in the text data.
- Model Callbacks: Using `EarlyStopping` and `lr_schedular` from tensorflow to get efficient training.
- Flask Web Framework: Creating a web application to interact with the model using `Flask`.

### Functions
- `clean_text(text)`: Preprocesses the input text by removing non-alphanumeric characters and creating tokens.
- `predict_language(text, model, cv, le)`: Uses the trained model to predict the language of the input text.
  
### Usage
- Install the dependencies `pip install -r requirements.txt`
- Run the Flask app: `python app.py`
- Open a web browser and go to `http://localhost:5000`
- Enter text into the form and submit to see the predicted language.

### Web App
![Screenshot (27)](https://github.com/HassaanIK/Language_Identification/assets/139614780/1e59ecc8-f8ea-4f63-9da5-ff5a06fa6381)
![Screenshot (28)](https://github.com/HassaanIK/Language_Identification/assets/139614780/2b883da4-f0ca-40a1-a551-9284d657b730)

