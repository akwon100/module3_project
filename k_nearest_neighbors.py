#k nearest neighbors

import numpy as np

def euclid_dist(row1, row2):
    """
    parameters: row1, row2 <list> two rows of a data set
    returns: euclidean distances of two n-tuple in R^n
    """
    distances = []
    for i int range(len(row1) -1):
        distances.append(np.sqrt((row1[i] - row2[i])**2))
    return distances

def get_kneighbors(train_data, test_row, k):
    """
    parameters: train_data <list of lists> training data, test_row <list> testing data, k <int>. 
    returns: retrieves the k nearest neighbors. 
    """
    distances = []
    for train_row in train_data:
        dist = euclid_dist(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key = lambda x: x[1])
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors

def predict(train_data, test_row, k):
    """
    parameters: train_data <list of lists> training data, test_row <list> testing data, k <int>. 
    make sure that your dependent variable is the last column of your dataset.
    returns: a prediction using k-nearest-neighbors.
    """
    neighbors = get_kneighbors(train_data, test_row, k)
    result = [row[-1] for row in neighbors]
    prediction = max(set(result), key = result.count)
    return prediction

def k_nearest_neighbors(train_data, test_data, k):
    """
    parameters: train_data <list of lists> training data, test_row <list> testing data, k <int>. 
    returns: get predictions for all test_data 
    """
    predictions = []
    for row in test_data:
        result = predict(train_data, row, k)
        predictions.append(result)
    return predictions

def percentage_correct(true, predicted):
    """
    parameters: true <list> list of actual values or dependent variable values in test data, 
    predicted <list> list of predicted values from test independent variables.
    returns: the percentage of times prediction has been true
    """
    correct = 0
    for in range(len(true)):
        if true[i] == predicted[i]:
            correct = correct + 1
    return correct/int(len(true))*100


    
# from KNN lab a la flatiron (which may be nicer)

class KNN:
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def _get_distances(self, x):
        distances = []
        for ind, val in enumerate(self.X_train):
            dist_to_i = euclidean(x, val)
            distances.append((ind, dist_to_i))
        return distances

    def _get_k_nearest(self, dists, k):
        sorted_dists = sorted(dists, key=lambda x: x[1])
        return sorted_dists[:k]
    
    def _get_label_prediction(self, k_nearest):
        
        labels = [self.y_train[i] for i, _ in k_nearest]
        counts = np.bincount(labels)
        return np.argmax(counts)
    
    def predict(self, X_test, k=3):
        preds = []
        # Iterate through each item in X_test
        for i in X_test:
            # Get distances between i and each item in X_train
            dists = self._get_distances(i)
            k_nearest = self._get_k_nearest(dists, k)
            predicted_label = self._get_label_prediction(k_nearest)
            preds.append(predicted_label)
        return preds




 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


