#decision trees CART

def splitData(df, fraction):
    '''
    parameters: <pandas.dataframe> df, the percentage used for training.
    returns: randomly split data frame of test and train data.
    '''
    df_train = df.sample(frac = fraction) 
    df_test = pd.concat([df, df_train, df_train]).drop_duplicates(keep=False)
    return (df_train,df_test)

def changeDataframeToList(df):
    """
    parameters: df <pandas.dataframe> dataset.
    returns: data frame into list of lists, each list is a row of the dataframe.
    """
    return df.values.tolist()

def variables(df):
    """
    parameters: df <pandas.dataframe> dataset
    returns: variables of df
    """
    return list(df.columns)

class Question:
    """
    parameters: df <pandas.dataframe> dataset.
    A Question is used to partition a dataset.

    This class just records a column number and a column value of a dataset. 
    The 'match' method is used to compare the feature value in an example to the feature value stored in the
    question.
    """

    def __init__(self, col_num, value):
        self.column = col_num
        self.value = value

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        if isinstance(val,float) or isinstance(val,int):
            return val >= self.value
        else:
            return val == self.value
    
    def __repr__(self):
        condition = "=="
        if isinstance(self.value, float) or isinstance(self.value, int):
            condition = ">="
        return "Is var_col index %s %s %s ?" % (str(self.column), condition, str(self.value))
        
def partition(rows, question):
    """
    parameters: rows <list of lists> rows of your dataset, question <tuple> (column number, value)
    returns: partitions a dataset into true or false.
    """
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

def label_counts(rows):
    """
    parameters: rows <list of lists> rows of your dataset. Make sure your y column is last list in rows
    returns: <dict> Counts for each label of column of dependent variable"""
    counts = {}  
    for row in rows:
        #labels are dependent variable y
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def gini(rows):
    """
    parameters: rows <list of lists> rows of your dataset, index_of_y <string> column of dependent variable y.
    returns: the calculated Gini Impurity for a list of rows.
    """
    counts = label_counts(rows)
    impurity = 1
    for label in counts:
        prob_of_label = counts[label] / float(len(rows))
        impurity = impurity - prob_of_label**2
    return impurity

def info_gain(left, right, current_uncertainty):
    """
    parameters: left, right <list> subset of dataset that is true, false resp. 
    current_uncertainty <list> gini(dataset) 
    returns: the uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)

def find_best_split(rows):
    """
    parameters: rows <list of lists> rows of your dataset, index_of_y <string> column of dependent variable y.
    returns: best info_gain and best question.
    """
    best_gain = 0
    best_question = None
    current_uncertainty = gini(rows)
    
    for col in range(len(rows[0])-1):
        unique_values = set([row[col] for row in rows])
        for values in unique_values:
            question = Question(col,values)
            true_rows, false_rows = partition(rows, question)
        
        if len(true_rows) == 0 or len(false_rows) == 0:
            continue
        
        gain = info_gain(true_rows, false_rows, current_uncertainty)
        if gain >= best_gain:
            best_gain, best_question = gain, question
    return best_gain, best_question

class Leaf:
    """
    A Leaf node is the last child node in a decision tree.

    This holds a dictionary of class, 
    i.e. number of times it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, rows):
        self.predictions = label_counts(rows)
        
class Decision_Node:
    """A Decision Node asks a question. It is not a root node or a leaf node.

    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch
        

def build_tree(rows):
    """
    parameters: rows <list of lists> rows of your dataset, index_of_y <string> column of dependent variable y.
    returns: builds a decision tree and stores decision nodes.
    """
    gain, question = find_best_split(rows)
    if gain == 0:
        return Leaf(rows)
    
    true_rows, false_rows = partition(rows, question)
    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)
    
    return Decision_Node(question, true_branch, false_branch)

def print_tree(node, spacing=""):
    """
    parameters: node = tree
    returns: printed tree
    """

    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return
    print ('----------------------------------------')
    print (spacing + str(node.question))
    print ('----------------------------------------')
    print (spacing + 'True:')
    print_tree(node.true_branch, spacing + "  ")
    print ('----------------------------------------')
    print (spacing + 'False:')
    print_tree(node.false_branch, spacing + "  ")
    
def print_leaf_helper(row, node):
    """
    parameters: row <list> one row of our dataset, node of tree
    returns: cassifies by comparing the feature/ value stored in the node to the example we are considering
    """
    if isinstance(node, Leaf):
        return node.predictions
    
    if node.question.match(row):
        return print_leaf_helper(row, node.true_branch)
    else:
        return print_leaf_helper(row, node.false_branch)

def print_leaf(row,node):
    '''
    parameters: row <list> one row of our dataset, node of tree
    returns: <dict> classification with percentage of confidence.
    '''
    counts = print_leaf_helper(row,node)
    total = sum(counts.values())
    prob = {}
    for label in counts.keys():
        prob[label] = str(int(counts[label] / total*100)) + '%'
    return prob

def _traverse_tree(self, x, node):
    if node.is_leaf_node():
        return node.value

           
    
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
