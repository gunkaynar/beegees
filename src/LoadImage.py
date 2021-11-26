import os
import glob
import numpy as np
from PIL import Image

# The old version creates an array with shape (mxnxk,), for all R, G, B pixels of all k images with shape (mxn).
# But we need an array with shape (mxn,1,k), for all R, G, B pixels of all k images with shape (mxn).

def load_image(path):
    image = Image.open(path)
    data = np.asarray(image, dtype="int32")
    return data


def load_all_images():
    root = '/Users/fatmanur/beegees-main/data'   # Rootu kendinize göre ayarlayın
    
    b_list = []
    g_list = []
    r_list = []
    
    for filename in glob.iglob(os.path.normpath(os.path.join(root, "**/*.jpg")), recursive = True):
        image_data = load_image(filename)
        if (len(image_data.shape) == 3):
            b, g, r = image_data[:, :, 0], image_data[:, :, 1], image_data[:, :, 2]
            
            b_list.append([b.flatten()])
            g_list.append([g.flatten()])
            r_list.append([r.flatten()])
            
    _1D_b_array = np.array(b_list)
    _1D_g_array = np.array(g_list)
    _1D_r_array = np.array(r_list)
    
    r_array = _1D_r_array.transpose()
    g_array = _1D_g_array.transpose()
    b_array = _1D_b_array.transpose()
    
    return r_array, g_array, b_array

def label():
    root = '/Users/fatmanur/beegees-main/data'
    label_list = [] 
    for foldername in glob.iglob(os.path.normpath(os.path.join(root,"*")), recursive = True):
        for filename in glob.iglob(os.path.normpath(os.path.join(foldername, "*.jpg")), recursive = True):
            if (foldername == root + '/bee1' or foldername == root + '/bee2'):
                label_list.append(1)
            else : 
                label_list.append(0) 
    label = np.array(label_list)
    return label
