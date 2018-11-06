from sklearn.base import BaseEstimator, ClassifierMixin
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.layers.convolutional import MaxPooling2D
from keras.utils import to_categorical
from keras import optimizers
from keras.layers.core import Flatten
from keras.layers import Dropout
from keras.callbacks import EarlyStopping

import pickle

class ImageClassifier(BaseEstimator, ClassifierMixin):  
    """An example of classifier"""

    def __init__(self, batch_size, epochs):
        """
        Called when initializing the classifier
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None

    def fit(self, X, y=None):
        model = Sequential()
        model.add(Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(12, activation = 'softmax'))
        model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=0.01),
              metrics=['accuracy'])
        model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs, validation_split=0.1)
        self.model = model
    
    def predict(self, X):
        if(self.model is None):
            print("You should train your model before predicting")
            return None
        y_pred = self.model.predict(X)
        return y_pred
    
    def save_model(self, path_model):
        """
        Save model and possible classes into pickle files
        params:
         - path_model, path_classes (string): path where the 2 files will be saved
        """
        self.model.save(path_model)

    def load_model(self, path_model):
        """
        Load model and classes, used in predict mode
        params:
         - path_model, path_classes (string): path where the 2 files are saved
        """
        try:
            self.model = load_model(path_model)

        except Exception as e:
            print('Unexpected error:', e)

        
        
        