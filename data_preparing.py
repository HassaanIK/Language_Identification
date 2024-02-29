from sklearn.feature_extraction.text import CountVectorizer
from data_analysis import df
from sklearn.preprocessing import LabelEncoder
from data_splitting import y_train, y_val
import tensorflow as tf

#Input Variable
# vectorizing input varible 'clean_text' into a matrix 
features = df['clean_text']

cv = CountVectorizer() # ngram_range=(1,2)
features = cv.fit_transform(features)

# changing the datatype of the number into uint8 to consume less memory
features = features.astype('uint8') # uint8 and float32


# defining target variable
# using LabelEncoder to get placeholder number values for categorical variabel 'language'
le = LabelEncoder()
df['language_encoded'] = le.fit_transform(df['language'])

targets = df['language_encoded']

y_train_encoded = tf.keras.utils.to_categorical(y_train, num_classes=22)
y_val_encoded = tf.keras.utils.to_categorical(y_val, num_classes=22)