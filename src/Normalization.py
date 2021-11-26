import numpy as np
from LoadImage import load_all_images, label
from sklearn.model_selection import train_test_split

def normalization():
    r_array, g_array, b_array = load_all_images()
    
    RGBmatrix = np.array([r_array,g_array,b_array])
    
    gray = np.empty_like(r_array)
    for i in range (np.shape(RGBmatrix)[1]):
        for j in range (np.shape(RGBmatrix)[3]):
            gray[i][0][j] = r_array[i][0][j]*0.299 + g_array[i][0][j]*0.587 + b_array[i][0][j]*0.114

    return gray

def load_data():
    
