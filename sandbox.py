from impyte import Imputer
import pandas as pd
import numpy as np


#imputer = Imputer(data=pd.DataFrame(np.random.randint(low=0, high=10, size=(4, 4)), columns=['W', 'X', 'Y', 'Z']))
imputer = Imputer(np.random.randint(low=0, high=10, size=(4, 4)))

print imputer