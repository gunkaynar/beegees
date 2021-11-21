import os
import glob
import numpy as np
from PIL import Image


def load_image(path):
    image = Image.open(path)
    data = np.asarray(image, dtype="int32")
    return data


def load_all_images():
    root = "C:/Users/tanay/Documents/Github/beegees/data/"   # Rootu kendinize göre ayarlayın
    
    b_array = np.array([])
    g_array = np.array([])
    r_array = np.array([])
    
    for filename in glob.iglob(os.path.normpath(os.path.join(root, "**/*.jpg")), recursive = True):
        image_data = load_image(filename)
        if (len(image_data.shape) == 3):
            b, g, r = image_data[:, :, 0], image_data[:, :, 1], image_data[:, :, 2]
            b_array = np.append(b_array, b)
            g_array = np.append(g_array, g)
            r_array = np.append(r_array, r)
        print(filename)
    
    return r_array, g_array, b_array

load_all_images()