import numpy as np
from PIL import Image


def process_image(image):
    
    size = 256, 256
    image.thumbnail(size)
    
    width, height = image.size # Get dimensions

    new_width = 224
    new_height = 224

    left = (width - new_width)/2.
    top = (height - new_height)/2.
    right = (width + new_width)/2.
    bottom = (height + new_height)/2.

    image = image.crop((left, top, right, bottom))
    
    image = np.array(image)
    image =  image.astype(np.float64)
    
    min_values = np.array([0, 0, 0])
    max_values = np.array([255, 255, 255])    
    
    image -= min_values 
    image /= max_values - min_values
    
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    
    image -= means 
    image /= stds 
        
    # we make the current last axis, and make it the first axis.
    image = image.transpose(2, 0, 1)
    
    
    return image 
    


def recover_key(d, value):
    for k, v in d.items():
        if v == value: 
            return k

        