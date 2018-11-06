import numpy as np
from sklearn.preprocessing import LabelBinarizer
import pickle

def scale_matrix(images_array):
    '''
    Function used to scale images array
    
    input:
        - images_array (numpy.array): Should contain all images and should also contain the 3 channels
    output:
        - images_array_scaled (numpy.array): Rescaled array
    '''
    return images_array / 255

def get_label_binarizer(y_train):
    '''
    Function used to get the label binarizer that will be used later on for re-encoding train and test labels
    
    input:
        - y_train (list) should contain labels as a list of strings
    output:
        - lb_bin (sklearn.preprocessing.LabelBinarizer) a Binarizer object that can be applied later on.
    '''
    lb_bin = LabelBinarizer()
    lb_bin.fit(y_train)
    return lb_bin

def transform_y_binarizer(y, lb_bin):
    '''
    Function used to transform y values (either train or test) into a proper format for neural nets.
    
    Input:
        - y (list) should contain labels as a list of strings
        - lb_bin (sklearn.preprocessing.LabelBinarizer) a Binarizer object output from get_label_binarizer function
    Output:
        - binarized_y (list of numpy.array): list of arrays corresponding to  of each class
    '''
    binarized_y = lb_bin.transform(y)
    return binarized_y

def save_binarizer(path_to_binarizer, binarizer):
    '''
    Need a docstring!!!
    '''
    path_to_binarizer = open(path_to_binarizer, 'rb')
    pickle.dump(binarizer, path_to_binarizer)
    path_to_binarizer.close()
    
def load_binarizer(path_to_binarizer):
    '''
    Need a docstring!!!
    '''
    path_to_binarizer = open(path_to_binarizer, 'rb')
    binarizer = pickle.load(path_to_binarizer)
    path_classes.close()
    return binarizer

    
    