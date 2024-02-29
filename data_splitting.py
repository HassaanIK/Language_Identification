from sklearn.model_selection import train_test_split
from data_preparing import features, targets
from data_preparing import le
# splitting data into training and testing datasets

X_train, X_val, y_train, y_val = train_test_split(features, targets, test_size=0.2, random_state=2007)

X_train = X_train.toarray()
X_val = X_val.toarray()

input_size = X_train.shape[1]
num_classes = len(le.classes_)
