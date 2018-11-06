from tagger.data_cleaning.data_cleaning import load_resize_image
from tagger.data_cleaning.data_cleaning import exagerate_contrast
from tagger.data_cleaning.data_cleaning import exagerate_contrast_2

from tagger.feature_engineering.feature_engineering import scale_matrix
from tagger.feature_engineering.feature_engineering import get_label_binarizer
from tagger.feature_engineering.feature_engineering import transform_y_binarizer
from tagger.feature_engineering.feature_engineering import save_binarizer

from tagger.model.model import ImageClassifier

import os
from PIL import Image
import numpy as np

def workflow_training(path_to_images, path_to_saves):
    '''
    Need a docstring!!
    Assuming that data is in classed into directories for each label
    '''
    # Load Data
    labels = []
    list_imgs = []
    for element_type in os.listdir(path_to_images):
        for pic in os.listdir(path_to_images + '/' + element_type):
            img = load_resize_image(path_to_images + '/' + element_type + '/' + pic)
            img = exagerate_contrast_2(img)
            img = np.array(img)
            list_imgs.append(img)
            labels.append(element_type)

    train_images = np.array(list_imgs)
    
    lb_bin = get_label_binarizer(labels)
    y = transform_y_binarizer(labels, lb_bin)
    
    X = scale_matrix(train_images)
    
    model = ImageClassifier(epochs=30, batch_size=128)
    model.fit(X, y)
    
    # Save model
    model.save_model(path_to_saves + '/model.pkl')
    save_binarizer(path_to_saves + '/binarizer.pkl', lb_bin)
    
    
    
    
            
    
    


