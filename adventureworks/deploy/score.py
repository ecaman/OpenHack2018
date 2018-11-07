import json
import numpy as np
import os
import pickle
from PIL import Image
from keras.models import load_model
#from tagger.model.model import ImageClassifier
#from tagger.data_cleaning.data_cleaning import load_resize_image, exagerate_contrast
#from tagger.feature_engineering.feature_engineering import load_binarizer
from sklearn.preprocessing import LabelBinarizer

from azureml.core.model import Model

def init():
    global model
    global binarizer
    # retreive the path to the model file using the model name
    model_path = Model.get_model_path('neural_model.pkl')
    binarizer_path = Model.get_model_path('binarizer.pkl')
    
    # Load binarizer
    path_to_binarizer = open(binarizer_path, 'rb')
    binarizer = pickle.load(path_to_binarizer)
    path_to_binarizer.close()
    
    # Load model
    model = load_model(model_path)

def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    
    img = Image.fromarray(np.uint8(data))
    
    # Resize image
    if img.size[0] > img.size[1]:
        white = np.zeros((img.size[0] - img.size[1], img.size[0],  3)) + 255
        img = np.vstack((np.array(img)[:,:,:3], white))
    elif img.size[0] < img.size[1]:
        white = np.zeros((img.size[1], img.size[1] - img.size[0], 3)) + 255
        img = np.hstack((np.array(img)[:,:,:3], white))
    else:
        img = np.array(img)[:,:,:3]
    basewidth = 128
    img = Image.fromarray(np.uint8(img))
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    
    # Exagerate colors
    hist,bins = np.histogram(np.array(img).flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    img2 = cdf[img]

    img = Image.fromarray(img2)
    
    img = np.array(img)
    
    # make prediction
    y_hat = model.predict(img)
    y_hat = binarizer.invert_transform(y_hat, 0.5)
    return json.dumps(y_hat.tolist())
    
    
    
    

    