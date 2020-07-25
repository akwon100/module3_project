import numpy as np
import pandas as pd

def splitData(df, fraction):
    """
    parameters: df <pandas.dataframe> the percentage used for training
    returns: randomly split data frame of test and train data
    """
    df_train = df.sample(frac = fraction) 
    df_test = pd.concat([df, df_train, df_train]).drop_duplicates(keep=False)
    return (df_train,df_test)

def weightInitialization(k_features):
    """
    parameters: k_features <int> the number of variables 
    returns: a zero vector of size k_features for weights and zero for bias 
    """
    w = np.zeros(k_features)
    b = 0
    return w,b

def sigmoid(x):
    """
    parameters: x <float> a real number. 
    returns: sigmoid function evaluated at x.
    """
    return 1/(1+np.exp(-x))

def model_helper(w, b, X, Y):
    """
    parameters: w <array> weights, b <int> bias, X <matrix> dependent variables, Y <array> indepdent variable.
    returns: gradient and cost. 
    """
    N = X.shape[0] 
    cost = (-1/N)*(np.sum((Y.T*np.log(sigmoid(np.dot(w,X.T)+b))) + ((1-Y.T)*(np.log(1-sigmoid(np.dot(w,X.T)+b))))))
   
    #Gradient calculation
    m = X.shape[1]
    dw = (1/m)*(np.dot(X.T,(sigmoid(np.dot(w,X.T)+b)-Y.T).T))
    db = (1/m)*(np.sum(sigmoid(np.dot(w,X.T)+b)-Y.T))
    gradient = {"dw": dw, "db": db}    
    return gradient, cost

def model_fit(w, b, X, Y, learning_rate = 0.001, num_iterations = 1000):
    """
    parameters: w <array> weights, b <int> bias, X <matrix> dependent variables, Y <array> indepdent variable,
    learning_rate <float> default is 0.001, num_iterations <int> default is 1000
    
    returns: best fit coefficients, gradient and costs
    """
    costs = []
    for i in range(num_iterations):
        gradient, cost = model_helper(w,b,X,Y)
        dw = gradient["dw"]
        db = gradient["db"]        
        w = w - (learning_rate * (dw.T))
        b = b - (learning_rate * db)        
        if (i % 100 == 0):
            costs.append(cost)
            #print("Cost after %s iteration is %s" %(i, cost)) 
    coeff = {"w": w, "b": b}
    gradient = {"dw": dw, "db": db}
    return coeff, gradient, costs


def log_reg_predict(X, n, w, b, threshold = 0.5):
    """
    parameters: X <matrix> test independent variables, n <int> number of observations, w <array> weights, b <float> bias, 
    threshold <float> default set to 0.5
    
    returns: predictions using Logistic Regression
    """
    final_pred = sigmoid(np.dot(w,X.T)+b)
    y_pred = np.zeros(n)
    for i in range(len(final_pred)):
        if final_pred[i] >= 0.5:
            y_pred[i] = 1
        elif final_pred[i] < 0.5:
            y_pred[i] = 0
    return y_pred

def percentage_correct(true, predicted):
    """
    parameters: true <list> list of actual values or dependent variable values in test data, 
    predicted <list> list of predicted values from test independent variables.
    returns: the percentage of times prediction has been true
    """
    correct = 0
    for i in range(len(true)):
        if true[i] == predicted[i]:
            correct = correct + 1
    return correct/int(len(true))*100






