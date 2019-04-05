
# coding: utf-8

import numpy as np

def perceptron(X, y, epochs, r=1):
    """
    Uses the perceptron algorithm to predict a linear classifier.
    
    Input:
    -X:  a numpy array of shape [no_samples, no_attributes + 1] representing the dataset
    -y:  a numpy array of shape [no_samples] that has the labels {-1, 1} for the dataset
    -epochs:  an int that represents the number of epochs to perform.
    -r:  A float representing the learning rate to be used in calculating the gradient.
    
    Returns:
    -w:  a numpy array of shape [no_attributes + bias] representing the set of weights learned in the algorithm.
    """

    import numpy as np

    w = np.zeros(X.shape[1]) # initialize the weights as zero
    for epoch in range(epochs):
        for i in range(X.shape[0]):
            y_pred = np.sign(w.dot(X[i]))
            if y_pred != y[i]:
                w = w + r * y[i] * X[i]
    
    return w
    


# In[179]:


def voted_perceptron(X, y, epochs, r=1):
    """
    Uses the voted perceptron algorithm to predict a linear classifier.
    
    Input:
    -X:  a numpy array of shape [no_samples, no_attributes + 1] representing the dataset
    -y:  a numpy array of shape [no_samples] that has the labels {-1, 1} for the dataset
    -epochs:  an int that represents the number of epochs to perform.
    -r:  A float representing the learning rate to be used in calculating the gradient.
    
    Returns:
    -weights:  a 2D numpy array of shape [k, no_attributes + bias] representing the list
    of weights learned in each k iteration of the algorithm.
    -counts:  a numpy array of shape [k] representing the counts for each iteration
    """

    import numpy as np

    w = np.zeros(X.shape[1]) # initialize the weights as zero
    weight_list = []
    count_list = []
    correct = 1
    for epoch in range(epochs):
        for i in range(X.shape[0]):
            y_pred = np.sign(w.dot(X[i]))
            if y_pred != y[i]:
                weight_list.append(w)
                count_list.append(correct)
                w = w + r * y[i] * X[i]
                correct = 1
            else:
                correct +=1
    
    weights = np.asarray(weight_list)
    counts = np.asarray(count_list)
    
    return weights, counts
    


# In[180]:


def averaged_perceptron(X, y, epochs, r=1):
    """
    Uses the averaged perceptron algorithm to predict a linear classifier.
    
    Input:
    -X:  a numpy array of shape [no_samples, no_attributes + 1] representing the dataset
    -y:  a numpy array of shape [no_samples] that has the labels {-1, 1} for the dataset
    -epochs:  an int that represents the number of epochs to perform.
    -r:  A float representing the learning rate to be used in calculating the gradient.
    
    Returns:
    -a:  a numpy array of shape [no_attributes + 1], which represents an averaged set of weights
    """

    import numpy as np

    w = a = np.zeros(X.shape[1]) # initialize the weights as zero
    for epoch in range(epochs):
        for i in range(X.shape[0]):
            y_pred = np.sign(w.dot(X[i]))
            if y_pred != y[i]:
                w = w + r * y[i] * X[i]
            a += w
    
    return a
    


# In[181]:


def predict(X, y, w):
    """
    Computes the average prediction error of a dataset X and given weight vector w.
    
    Inputs:
    -X:  a numpy array of shape [no_samples, no_attributes + 1] representing the dataset
    -y:  a numpy array of shape [no_samples] that has the labels for the dataset
    -w:  a numpy array of shape [no_attributes + bias] representing the set of weights to predict the labels.
    
    Returns:
    -error:  A float representing the average prediction error of for the dataset X.
    """
    
    incorrect = 0
    for i in range(X.shape[0]):
        if np.sign(w.dot(X[i])) != y[i]:
            incorrect += 1
           
    print("The total incorrect is " + str(incorrect))
    return float(incorrect) / X.shape[0]


# In[173]:


def predict_voted(X, y, w, c):
    """
    Computes the average prediction error of a dataset X and given weight vector w for the voted perceptron algorithm.
    
    Inputs:
    -X:  a numpy array of shape [no_samples, no_attributes + 1] representing the dataset
    -y:  a numpy array of shape [no_samples] that has the labels for the dataset
    -w:  a 2d numpy array of shape [k, no_attributes + bias] representing a list of weights.
    -c:  a numpy array of shape[k] representing the associated counts for the weights
    
    Returns:
    -error:  A float representing the average prediction error of for the dataset X.
    """
    
    incorrect = 0
    for i in range(X.shape[0]):
        weighted_sum = 0;
        for k in range(w.shape[0]):
            weighted_sum += c[k] * np.sign(w[k].dot(X[i]))
        predict = 1
        if weighted_sum <= 0:
            predict = -1
        if predict != y[i]:
            incorrect += 1
            
    print("The total incorrect is " + str(incorrect))
    return float(incorrect) / X.shape[0]


# In[182]:


def load_data(path):
    """
    Loads and processes the bank note data set
    
    Inputs:
    -path:  string representing the path of the file
    
    Returns:
    -X:  a numpy array of shape [no_samples, no_attributes + 1]
    -y:  a numpy array of shape [no_samples] that represents the labels {-1, 1} for the dataset X
    """
    
    import numpy as np
    
    data = []
    with open(path, 'r') as f:
        for line in f:
            example = line.strip().split(',')
            if len(example) > 0:
                example = [float(i) for i in example]
                data.append(example)
    X = np.array(data, dtype=np.float64)
    y = X[:,-1]
    y = y.astype(int)
    new_y = y[y == 0] = -1
    X = X[:,:-1]
    bias = np.ones((X.shape[0],1),dtype=np.float64) 
    X = np.hstack((X,bias))

    return X, y


# In[183]:


#Test the Implementations
X, y = load_data('train.csv')
X_test, y_test = load_data("test.csv")

#Naive Implementation
w = perceptron(X,y,10)
print("The weights for the naive perceptron implementation are: ")
print(w)
print('\n')
print("The average prediction error for naive perceptron is " + str(predict(X_test, y_test, w)))
print('\n')

#Voted Implementation
w, c = voted_perceptron(X,y,10)
print("The list of weights for the voted perceptron implementation are: ")
print(*w,sep='\n')
print("The list of counts for the voted implementation are: ")
print(c)
print('\n')
print("The average prediction error for voted perceptron is " + str(predict_voted(X_test, y_test, w, c)))
print('\n')

#Averaged Implementation
a = averaged_perceptron(X,y,10)
print("The list of weights for the averaged perceptron implementation are: ")
print(a)
print('\n')
print("The average prediction error for averaged perceptron is " + str(predict(X_test, y_test, a)))



