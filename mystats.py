import numpy as np
import pandas as pd

#log regression
def perfect_separation_indicator(df,x,y):
    """
    parameters: df <pandas.dataframe> dataset, y <string> dependent variable, x <string> independent variable.
    returns: checks for perfect separation: false if no perfect separation, true if perfect separation
    """
    X, Y = list(df[x]), list(df[y])
    count_false = 0
    all_pairs = []
    for x_i, y_i in zip(X,Y):
        all_pairs.append((x_i, y_i))
    for i in range(len(all_pairs)):
        for pair in all_pairs:
            if all_pairs[i][0] == pair[0] and all_pairs[i][1] != pair[1]:
                count_false +=1
    if count_false >1:
        return False
    else:
        return True
    
def stats_invertible(df, y):
    '''
    parameters: df <pandas.dataframe> dataset, y <string> log dependent variable
    returns: wether or not the hessian is invertible. 
    If not invertible then you will get nan values on stats log reg summary.
    '''
    X = df.drop(y, axis =1)
    X = sm.tools.add_constant(X)
    Y = df[y]
    logit_model = sm.Logit(Y, X)
    result = logit_model.fit()
    Matrix = np.diag(result.cov_params())
    return Matrix

        
import statsmodels.api as sm
def stats_logreg_summary(df, y):
    '''
    parameters: df <pandas.dataframe> dataset, y <string> dependent variable.
    returns: logistic regression summary via statsmodels. 
    '''
    X = df.drop(y, axis =1)
    X = sm.tools.add_constant(X)
    Y = df[y]
    logit_model = sm.Logit(Y, X)
    result = logit_model.fit()
    print(result.summary())
    

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def split(df, y, testSize):
    X = df.drop(y, axis =1)
    Y = df[y]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=testSize, random_state=101)
    return X_train, X_test, y_train, y_test

def sklearn_logreg_predict(df, y, testSize):
    '''
    parameters: df <pandas.dataframe> dataset, y <string> dependent variable y, testSize <float> test size.
    retturns: predicted train values, train values, predicted test values, test values.
    '''
    X_train, X_test, y_train, y_test = split(df, y, testSize)
    logreg = LogisticRegression(fit_intercept=False, C=1e12, solver='liblinear')
    logreg.fit(X_train, y_train)
    y_hat_train = logreg.predict(X_train)
    y_hat_test = logreg.predict(X_test)
    return y_hat_train, y_train, y_hat_test, y_test



def correct_percentage(y, y_hat):
    """
    parameters: y <list> actual values, y_hat <list> predicted values.
    returns: the number of correctly predicted values v. incorrect, percentage of correctly predicted values v. incorrect.
    """
    residuals = np.abs(y - y_hat)
    print(pd.Series(residuals).value_counts())
    print('------------------------------------')
    print(pd.Series(residuals).value_counts(normalize=True))



from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

def sklearn_summary(y_train, y_hat_train, y_test, y_hat_test):
    """
    parameters: y_train <list> actual values, y_hat_train <list> predicted values.
    retturns: precision, recall, accuracy, F1-Score of training and test data.
    """

    print('Training Precision: ', precision_score(y_train, y_hat_train))
    print('Testing Precision: ', precision_score(y_test, y_hat_test))
    print('--------------------------------------------------------------------')

    print('Training Recall: ', recall_score(y_train, y_hat_train))
    print('Testing Recall: ', recall_score(y_test, y_hat_test))
    print('--------------------------------------------------------------------')

    print('Training Accuracy: ', accuracy_score(y_train, y_hat_train))
    print('Testing Accuracy: ', accuracy_score(y_test, y_hat_test))
    print('--------------------------------------------------------------------')

    print('Training F1-Score: ', f1_score(y_train, y_hat_train))
    print('Testing F1-Score: ', f1_score(y_test, y_hat_test))

from sklearn.metrics import confusion_matrix

def cnf_matrix(y_test, y_hat_test):
    cnf_matrix = confusion_matrix(y_test, y_hat_test)
    print ('Confusion Matrix:\n', cnf_matrix)

def tuning_threshold(df, y, testSize):
    """
    parameters: df <pandas.dataframe> dataset, y <string> dependent variable y, testSize <float> test size.
    returns: testing accuracy and confusion matrix given different thresholds.
    """
    X_train, X_test, y_train, y_test = split(df, y, testSize)
    logreg = LogisticRegression(fit_intercept=False, C=1e12, solver='liblinear')
    model_log = logreg.fit(X_train, y_train)
    pred_proba = pd.DataFrame(model_log.predict_proba(X_test))
    
    threshold_list = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,.7,.75,.8,.85,.9,.95,.99]
    
    for i in threshold_list:
        print ('\n******** For threshold = {} ******'.format(i))
        Y_test_pred = pred_proba.applymap(lambda x: 1 if x>i else 0)
        
        test_accuracy = accuracy_score(y_test.to_numpy().reshape(y_test.to_numpy().size,1),
                                           Y_test_pred.iloc[:,1].to_numpy().reshape(Y_test_pred.iloc[:,1].to_numpy().size,1))
        test_precision = precision_score(y_test.to_numpy().reshape(y_test.to_numpy().size,1),
                                           Y_test_pred.iloc[:,1].to_numpy().reshape(Y_test_pred.iloc[:,1].to_numpy().size,1))
        test_recall = recall_score(y_test.to_numpy().reshape(y_test.to_numpy().size,1),
                                           Y_test_pred.iloc[:,1].to_numpy().reshape(Y_test_pred.iloc[:,1].to_numpy().size,1))
        test_f1score = f1_score(y_test.to_numpy().reshape(y_test.to_numpy().size,1),
                                           Y_test_pred.iloc[:,1].to_numpy().reshape(Y_test_pred.iloc[:,1].to_numpy().size,1))
        
        print('Our testing accuracy is: ', test_accuracy)
        print('Our testing precision is: ', test_precision)
        print('Our testing recall is: ', test_recall)
        print('Our testing F1 score is: ', test_f1score)

        print(confusion_matrix(y_test.to_numpy().reshape(y_test.to_numpy().size,1),
                           Y_test_pred.iloc[:,1].to_numpy().reshape(Y_test_pred.iloc[:,1].to_numpy().size,1)))


#kfold cross validation score
def kFoldCVS(df,k,y, model):
    '''
    parameters:<pandas.dataframe> df, <int> k number of folds, <string> y dependent variable
    returns: average cross validation score
    '''
    X = df.drop(y, axis =1)
    Y = df[y]
    kf = KFold(n_splits = 10, shuffle = True, random_state = 2)
    cvs = cross_val_score(model, X, Y, cv = kf)
    return cvs.mean()


#KNN
from sklearn.preprocessing import StandardScaler

def scaler(df, y):
    '''
    parameters: df <pd dataframe> dataset, y <string> independent variable
    returns: scaled df using standard scaler
    '''
    X = df.drop(y, axis =1)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X)
    scaled_df = pd.DataFrame(scaled_data, columns= X.columns)
    return scaled_df

def find_best_k_f1(X_train, y_train, X_test, y_test, min_k=1, max_k=25):
    '''
    parameters: X_train, y_train <list or array> training set, X_test, y_test <list or array> testing set,
    min_k = 1, max_k = 25 (default)
    returns: best fit for k using f1 score
    '''
    best_k = 0
    best_score = 0.0
    for k in range(min_k, max_k+1, 2):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        f1 = f1_score(y_test, preds)
        if f1 > best_score:
            best_k = k
            best_score = f1
    return best_k

def find_best_k_accuracy(X_train, y_train, X_test, y_test, min_k=1, max_k=25):
    '''
    parameters:X_train, y_train <list or array> training set, X_test, y_test <list or array> testing set,
    min_k = 1, max_k = 25 (default)
    returns: best fit for k using accuracy
    '''
    best_k = 0
    best_score = 0.0
    for k in range(min_k, max_k+1, 2):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        if accuracy > best_score:
            best_k = k
            best_score = accuracy
    return best_k

def find_best_k_precision(X_train, y_train, X_test, y_test, min_k=1, max_k=25):
    '''
    parameters:X_train, y_train <list or array> training set, X_test, y_test <list or array> testing set,
    min_k = 1, max_k = 25 (default)
    returns: best fit for k using precision
    '''
    best_k = 0
    best_score = 0.0
    for k in range(min_k, max_k+1, 2):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        precision = precision_score(y_test, preds)
        if precision > best_score:
            best_k = k
            best_score = precision
    return best_k

def find_best_k_recall(X_train, y_train, X_test, y_test, min_k=1, max_k=25):
    '''
    parameters:X_train, y_train <list or array> training set, X_test, y_test <list or array> testing set,
    min_k = 1, max_k = 25 (default)
    returns: best fit for k using recall
    '''
    best_k = 0
    best_score = 0.0
    for k in range(min_k, max_k+1, 2):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        recall = recall_score(y_test, preds)
        if recall > best_score:
            best_k = k
            best_score = recall
    return best_k
    
from sklearn.neighbors import KNeighborsClassifier

def sklearn_knn_predictions(df, y, testSize, k):
    '''
    parameters: df <pd dataframe> dataset, y <string> independent variable, testSize <float> test size, k <int> 
    returns: returns predictions using knn model
    '''
    X_train, X_test, y_train, y_test = split(df, y, testSize)
    clf = KNeighborsClassifier(n_neighbors = k)
    clf.fit(X_train, y_train)
    y_hat_test = clf.predict(X_test)
    y_hat_train = clf.predict(X_train)
    return y_hat_test, y_test, y_hat_train, y_train

# Decision Trees

from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import roc_curve, auc
from sklearn.tree import export_graphviz
from IPython.display import Image  
from pydotplus import graph_from_dot_data

def sklearn_dt_prediction(X_train, y_train, X_test):
    '''
    parameters: X_train, y_train <list or array> training data, X_test <list or array> testing variables.
    returns: y predictions using decision tree model.
    '''
    classifier = DecisionTreeClassifier(random_state=10, criterion='entropy')  
    classifier.fit(X_train, y_train)
    y_hat_test = classifier.predict(X_test)
    y_hat_train = classifier.predict(X_train) 
    return y_hat_test, y_hat_train

def sklearn_AUC(y_test, y_hat_test):
    '''
    parameters: y_test, y_hat_test <list or array> test values and prediction values. 
    returns: AUC 
    '''
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_hat_test)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print('\nAUC is :{0}'.format(round(roc_auc, 2)))

def tree(df, y, X_train, y_train):
    '''
    parameters:df <pd dataframe> dataset, y <string> independent variable,
    X_train, y_train <list or array> training data
    returns: decision tree graph
    '''
    classifier = DecisionTreeClassifier(random_state=10, criterion='entropy')  
    classifier.fit(X_train, y_train)
    y = df[y]
    dot_data = export_graphviz(classifier, out_file=None, 
                           feature_names=X_train.columns,  
                           class_names=np.unique(y).astype('str'), 
                           filled=True, rounded=True, special_characters=True)
    graph = graph_from_dot_data(dot_data)
    return Image(graph.create_png())
   

def max_depth(max_depths, X_test, y_test, X_train, y_train, SEED=1):
    '''
    parameters: max_depths <nd array>
    X_train, y_train <list or array> training data, X_test <list or array> testing variables,
    SEED=1 (default)
    returns: train and test results for different values of max depth
    '''
    train_results = []
    test_results = []
    for max_depth in max_depths:
       dt = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=SEED)
       dt.fit(X_train, y_train)
       train_pred = dt.predict(X_train)
       false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
       roc_auc = auc(false_positive_rate, true_positive_rate)
       # Add auc score to previous train results
       train_results.append(roc_auc)
       y_pred = dt.predict(X_test)
       false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
       roc_auc = auc(false_positive_rate, true_positive_rate)
       # Add auc score to previous test results
       test_results.append(roc_auc)
    return train_results, test_results 

def min_sample_split(min_samples_splits, X_train, y_train, X_test, y_test, SEED=1):
    '''
    parameters:min_sample_splits <nd array>
    X_train, y_train <list or array> training data, X_test <list or array> testing variables,
    SEED=1 (default)
    returns: train and test results for different values of minimum sample split.
    '''
    train_results = []
    test_results = []
    for min_samples_split in min_samples_splits:
       dt = DecisionTreeClassifier(criterion='entropy', min_samples_split=min_samples_split, random_state=SEED)
       dt.fit(X_train, y_train)
       train_pred = dt.predict(X_train)
       false_positive_rate, true_positive_rate, thresholds =    roc_curve(y_train, train_pred)
       roc_auc = auc(false_positive_rate, true_positive_rate)
       train_results.append(roc_auc)
       y_pred = dt.predict(X_test)
       false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
       roc_auc = auc(false_positive_rate, true_positive_rate)
       test_results.append(roc_auc)
    return train_results, test_results

def min_sample_leaf(min_samples_leafs, X_train, y_train, X_test, y_test, SEED=1):
    '''
    parameters:min_samples_leafs <nd array>
    X_train, y_train <list or array> training data, X_test <list or array> testing variables,
    SEED=1 (default)
    returns: train and test results for different values of minimum sample leaf.
    '''
    train_results = []
    test_results = []
    for min_samples_leaf in min_samples_leafs:
       dt = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=min_samples_leaf, random_state=SEED)
       dt.fit(X_train, y_train)
       train_pred = dt.predict(X_train)
       false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
       roc_auc = auc(false_positive_rate, true_positive_rate)
       train_results.append(roc_auc)
       y_pred = dt.predict(X_test)
       false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
       roc_auc = auc(false_positive_rate, true_positive_rate)
       test_results.append(roc_auc)
    return train_results, test_results


def max_feature(X_train, y_train, X_test, y_test, SEED=1):
    '''
    parameters:X_train, y_train <list or array> training data, X_test <list or array> testing variables,
    SEED=1 (default)
    returns: train and test results for different features. 
    '''
    max_features = list(range(1, X_train.shape[1]))
    train_results = []
    test_results = []
    for max_feature in max_features:
       dt = DecisionTreeClassifier(criterion='entropy', max_features=max_feature, random_state=SEED)
       dt.fit(X_train, y_train)
       train_pred = dt.predict(X_train)
       false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
       roc_auc = auc(false_positive_rate, true_positive_rate)
       train_results.append(roc_auc)
       y_pred = dt.predict(X_test)
       false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
       roc_auc = auc(false_positive_rate, true_positive_rate)
       test_results.append(roc_auc)
    return train_results, test_results


 

    
    
        
        