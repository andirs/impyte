from impyte import Imputer
import pandas as pd
import numpy as np
from tools.data_prep import remove_random

# Import data sets
from sklearn import datasets

## generate toy data set
data_toy = pd.DataFrame(np.random.randint(low=0, high=10, size=(4, 4)), columns=['W', 'X', 'Y', 'Z'])

## load iris_data set
raw_data = datasets.load_iris()
iris_X = raw_data.data
iris_y = raw_data.target
data_iris = pd.DataFrame(iris_X)
data_iris['target'] = iris_y
data_iris['target'] = data_iris['target'].astype('category')

# Shuffle data and replace 20 % of first column with NaN values

data_iris = remove_random(data_iris, .2)
data_iris.columns = ['Test 1', 'Test 2', 'Test 3', 'Test 4', 'target']

imputer = Imputer(data=data_toy)
print imputer
imputer = Imputer(np.random.randint(low=0, high=10, size=(4, 4)))
print imputer
imputer = Imputer(data_iris)
print imputer
#imputer = Imputer("String")
imputer.load_model("anything")

# Tests for is_nan()
nan_test = ['test', '', None, '5', 7]

# [False, True, True, False, False]
print imputer.is_nan(nan_test)

nan_test2 = [['test', '', None, '5', 7], None, 'Test']

# [[False, True, True, False, False], True, False]
print imputer.is_nan(nan_test2)

imputer.load_data(pd.DataFrame(np.random.randint(low=0, high=10, size=(4, 4)), columns=['W', 'X', 'Y', 'Z']))

## nan_check
