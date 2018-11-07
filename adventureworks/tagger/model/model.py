from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import load_model
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
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128,128,3)))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3)))
        model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        #model.add(tf.keras.layers.Dense(12, activation='softmax'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(12))
        model.add(tf.keras.layers.Activation("softmax"))
        
        model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(lr=0.01),
              metrics=['accuracy'])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        model.fit(X_train, y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  verbose=True,
                  validation_data=(X_test, y_test))
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

    def load_neural_model(self, path_model):
        """
        Load model and classes, used in predict mode
        params:
         - path_model, path_classes (string): path where the 2 files are saved
        """
        try:
            self.model = load_model(path_model)

        except Exception as e:
            print('Unexpected error:', e)

        
        
        