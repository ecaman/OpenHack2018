from tagger.data_cleaning.data_cleaning import load_resize_image
from tagger.data_cleaning.data_cleaning import exagerate_contrast
from tagger.data_cleaning.data_cleaning import exagerate_contrast_2

from tagger.feature_engineering.feature_engineering import scale_matrix
from tagger.feature_engineering.feature_engineering import get_label_binarizer
from tagger.feature_engineering.feature_engineering import transform_y_binarizer
from tagger.feature_engineering.feature_engineering import load_binarizer

from tagger.model.model import ImageClassifier

import os
from PIL import Image
import numpy as np

def workflow_predicting(path_to_images, path_to_saves):
    '''
    Need a docstring!!
    Assuming that data is not classed into directories for each label
    '''
    # Load Data
    list_imgs = []
    for pic in os.listdir(path_to_images):
        img = load_resize_image(path_to_images + '/' + pic)
        img = exagerate_contrast(img)
        img = np.array(img)
        list_imgs.append(img)
        labels.append(element_type)

    pred_images = np.array(list_imgs)
    
    # Load Model
    model = ImageClassifier(epochs=30, batch_size=128)
    
    model.load_neural_model(path_to_saves + '/model.pkl')
    lb_bin = load_binarizer(path_to_saves + '/binarizer.pkl')
    
    # Predicting
    
    X = scale_matrix(pred_images)
    y_pred = model.predict(X)
    final_pred = lb_bin.invert_transform(y_pred, 0.5)
    print(final_pred)
    
    
    
    
    
    
            
    
    



