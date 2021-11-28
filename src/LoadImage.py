import numpy as np
import random
import os
import os.path
import glob
from PIL import Image

# Load images as arrays
def load_image(path):
    image = Image.open(path)
    data = np.asarray(image, dtype="int32")
    return data

# Resize dataset to all have same size (nxm)
def resize():
    root = '/Users/fatmanur/beegees-main/data'
    for filename in glob.iglob(os.path.normpath(os.path.join(root, "**/*.jpg")), recursive = True):
        image = Image.open(filename)
        image = image.resize((224,224))
        image.save(filename)

# Split data into Training and Test Sets        
def data_split():
    root = '/Users/fatmanur/beegees-main/data'
    filenames = []
    for filename in glob.iglob(os.path.normpath(os.path.join(root, "**/*.jpg")), recursive = True):
        filenames.append(filename)
    filenames.sort()  
    random.seed(230)
    random.shuffle(filenames) 

    split = int(0.8 * len(filenames))
    train_filenames = filenames[:split]
    test_filenames = filenames[split:]
    return train_filenames, test_filenames

# Exctract RGB layers from image arrays and create RGB matrix. 
# Normalize RGB matrix to have GrayScale matrix by using linear approximation of gamma and perceptual luminance corrected
# Label datasets accourding to folder name
# Obtain training features & labels and testing features & labels
def load_data():
    resize()
    
    train_filenames, test_filenames = data_split()
    
    # Create RGB matrix for Train Dataset
    b_train_list = []
    g_train_list = []
    r_train_list = []
    
    for filename in train_filenames:
        image_data = load_image(filename)
        if (len(image_data.shape) == 3):
            b, g, r = image_data[:, :, 0], image_data[:, :, 1], image_data[:, :, 2]
            b_train_list.append([b.flatten()])
            g_train_list.append([g.flatten()])
            r_train_list.append([r.flatten()])
    
    b_tr_arr = np.array(b_train_list)
    g_tr_arr = np.array(g_train_list)
    r_tr_arr = np.array(r_train_list)
    b_train_array = b_tr_arr.transpose()
    g_train_array = g_tr_arr.transpose()
    r_train_array = r_tr_arr.transpose()
    
    RGBmatrix_train = np.array([r_train_array,g_train_array,b_train_array])
    
    # Creating Gray Scale Matrix from RGB Matrix of Train Dataset
    train_x_orig = np.empty_like(r_train_array)
    for i in range (np.shape(RGBmatrix_train)[1]):
        for j in range (np.shape(RGBmatrix_train)[3]):
            train_x_orig[i][0][j] = r_train_array[i][0][j]*0.299 + g_train_array[i][0][j]*0.587 + b_train_array[i][0][j]*0.114
    
    # Labelling Train Dataset
    y_train_list = []
    for filename in train_filenames:
        path = os.path.dirname(filename)
        name = os.path.basename(path)
        if (name == "bee1" or name == "bee2" ):
            y_train_list.append([1])
        else :
            y_train_list.append([0])
    
    y_tr_arr = np.array(y_train_list) 
    train_y = y_tr_arr.transpose()
    
    # Create RGB matrix for Test Dataset
    b_test_list = []
    g_test_list = []
    r_test_list = []
    for filename in test_filenames:
        image_data = load_image(filename)
        if (len(image_data.shape) == 3):
            b, g, r = image_data[:, :, 0], image_data[:, :, 1], image_data[:, :, 2]
            b_test_list.append([b.flatten()])
            g_test_list.append([g.flatten()])
            r_test_list.append([r.flatten()])
    
    b_ts_arr = np.array(b_test_list)
    g_ts_arr = np.array(g_test_list)
    r_ts_arr = np.array(r_test_list)
    
    b_test_array = b_ts_arr.transpose()
    g_test_array = g_ts_arr.transpose()
    r_test_array = r_ts_arr.transpose()
    
    RGBmatrix_test = np.array([r_test_array, g_test_array, b_test_array])
    
    # Creating Gray Scale Matrix from RGB Matrix of Test Dataset
    test_x_orig = np.empty_like(r_test_array)
    for i in range (np.shape(RGBmatrix_test)[1]):
        for j in range (np.shape(RGBmatrix_test)[3]):
            test_x_orig[i][0][j] = r_test_array[i][0][j]*0.299 + g_test_array[i][0][j]*0.587 + b_test_array[i][0][j]*0.114
            
    # Labelling Test Dataset    
    y_test_list = []
    for filename in test_filenames:
        path = os.path.dirname(filename)
        name = os.path.basename(path)
        if (name == "bee1" or name == "bee2" ):
            y_test_list.append([1])
        else :
            y_test_list.append([0])
    
    y_ts_arr = np.array(y_test_list) 
    test_y = y_ts_arr.transpose()   
    
        
    return train_x_orig, train_y, test_x_orig, test_y
