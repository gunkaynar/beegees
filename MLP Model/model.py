
import numpy as np
import matplotlib.pyplot as plt
from dnn import *


# Gerek kalmadi
# from LoadImage import load_data
# train_x_orig, train_y, test_x_orig, test_y = load_data()

# Collected arrays
train_x_orig = np.load('x_train.npy')
train_y = np.load('y_train.npy')
test_x_orig = np.load('x_test.npy')
test_y = np.load('y_test.npy')

# Bu kisimda reshape değerleri sadece b2 ve w1 için çalıştırılan modeldekilerle yapılmış, genelleştirdim.
train_x = np.reshape(train_x_orig,(np.shape(train_x_orig)[0],np.shape(train_x_orig)[2]))
train_x = train_x.T
test_x = np.reshape(test_x_orig,(np.shape(test_x_orig)[0],np.shape(test_x_orig)[2]))

test_x = test_x.T
m_train = train_x.shape[0]
num_px = train_x.shape[1]
m_test = test_x.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x.shape))
print ("test_y shape: " + str(test_y.shape))


  
def plot_costs(costs, learning_rate=0.0075):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()



layers_dims = [50176, 20, 7, 5, 1]  #adjust this



def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):


    np.random.seed(1)
    costs = []                         
    

    parameters = initialize_parameters_deep(layers_dims)
    

    for i in range(0, num_iterations):


        AL, caches = L_model_forward(X, parameters)

        cost = compute_cost(AL, Y)
        

        grads = L_model_backward(AL, Y, caches)
        

        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)
    
    return parameters, costs





print("Cost after first iteration: " + str(costs[0]))


parameters, costs = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)

plot_costs(costs, learning_rate)

