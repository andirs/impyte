from impyte import Imputer
import pandas as pd
import numpy as np

# Import data sets
from sklearn import datasets

raw_data = datasets.load_iris()
iris_X = raw_data.data
iris_y = raw_data.target
data = pd.DataFrame(iris_X)
data['target'] = iris_y
data['target'] = data['target'].astype('category')

# Shuffle data and replace 20 % of first column with NaN values
data = data.sample(frac=1).reset_index(drop=True)

# Adding 20 % of NaN Values
frac = int(.2 * len(data)) / 3

data.loc[0:frac-1,0] = None
data.loc[frac:frac*2-1,1] = None
data.loc[frac*2:frac*3-1,2] = None
data.loc[frac*2+5:frac*3-1+5,3] = None

data.columns = ['Test 1', 'Test 2', 'Test 3', 'Test 4', 'target']

imputer = Imputer(data=pd.DataFrame(np.random.randint(low=0, high=10, size=(4, 4)), columns=['W', 'X', 'Y', 'Z']))
print imputer
imputer = Imputer(np.random.randint(low=0, high=10, size=(4, 4)))
print imputer
imputer = Imputer(data)
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
