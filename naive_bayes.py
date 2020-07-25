#Naive bayes 

def prob_true(rows, true_obs):
    count_true = 0
    for row in rows:
        if row[-1] == true_obs:
            count_true += 1
    return count_true/len(rows)

def prob_false(rows, false_obs):
    count_false = 0
    for row in rows:
        if row[-1] == false_obs:
            count_false +=1
    return count_false/len(rows)

def prob_obs(rows, obs):
    count = 0
    for row in rows:
        for i in range(len(row)-1):
            if row[i] == obs[i]:
                count += 1
    return count/ len(rows)

def prob_given_true(rows, obs, true_obs):
    count_true = 0
    for row in rows:
        if row[:-1] == obs and row[-1] == true_obs:
                count_true += 1
    return count_true/len(rows)

def prob_given_false(rows, obs, false_obs):
    count_false = 0
    for row in rows:
        if row[:-1] == obs and row[-1] == false_obs:
                count_true += 1
    return count_false/len(rows)

def bayes_true(rows, obs, true_obs):
    probability_true = prob_true(rows, true_obs)* prob_given_true(rows, obs, true_obs) / prob_obs(rows, row)
    return probability_true

def bayes_false(rows, obs, false_obs):
    probability_false = prob_false(rows, false_obs)* prob_given_false(rows, obs, false_obs) / prob_obs(rows, row)
    return probability_false

def predict(rows, X_test, true_obs, false_obs):
    predictions = []
    for row in X_test:
        if bayes_true(rows, row, true_obs) > bayes_false(rows, row, false_obs):
            predictions.append(true_obs)
        else: 
            predictions.append(false_obs)
               
            
        




