"""
Module to impute missing values using machine learning algorithms.
author: Andreas Rubin-Schwarz
"""

import math
import numpy as np
import pandas as pd
from collections import Counter
from datetime import date
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, \
    GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor, SGDClassifier, BayesianRidge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.base import clone


class NanChecker:
    """
    Class that checks data set, lists or single
    values for NaN occurrence. 
    """
    @staticmethod
    def is_nan(data,
               nan_vals=None,
               recursive=True):
        """
        Detect missing values (NaN in numeric arrays, empty strings in string arrays).

        Parameters
        ----------
        data : ndarray
        nan_vals : array of values that count as NaN values - if empty, "" and None are being used
        recursive : handle lists in recursive manner

        Returns
        -------
        result : array-like of bool or bool
            Array or bool indicating whether an object is null or if an array is
            given which of the element is null.
        """

        # Use immutable object as standard object
        if nan_vals is None:
            nan_vals = ["", None]

        result = []
        # if data is a list, evaluate all objects
        if isinstance(data, list) or hasattr(data, '__array__') and not np.isscalar(data):
            for item in data:
                # if item is a list, call function recursively
                if isinstance(item, list) or hasattr(item, '__array__'):
                    if recursive:
                        result.append(NanChecker.is_nan(item))
                    else:
                        raise ValueError("List in a list detected. Set recursive to True.")
                # If item is string, evaluate if empty
                elif isinstance(item, basestring):
                    if item in nan_vals:
                        result.append(True)
                    else:
                        result.append(False)
                # Check for None value explicitly
                elif item is None:
                    result.append(True)
                # Check if NaN when float
                elif isinstance(item, float) and math.isnan(item):
                    result.append(True)
                else:
                    result.append(False)
            # if result is not a list, turn into a list
            if isinstance(result, list):
                return result
            else:
                return np.array(result)
        # if data is not a list, evaluate single value
        elif isinstance(data, basestring):
            return data in nan_vals
        elif data is None:
            return True
        else:
            return math.isnan(data)


class Pattern:
    """
    Class that calculates, stores and visualizes 
    NaN patterns and their indices. 
    """
    def __init__(self):
        self.nan_checker = NanChecker()
        self.pattern_store = {}
        self.pattern_index_store = {}
        self.discrete_variables = []
        self.continuous_variables = []
        self.unique_instances = 10
        # tryout
        self.missing_per_column = None
        self.column_names = []
        self.result_pattern_temp = {}
        self.tuple_counter_dict_temp = {}
        self.pattern_index_store_temp = {}
        self.tuple_counter_temp = 0
        self.pattern_store_temp = {}
        self.pattern_col_names = {}
        self.store_tuple_columns = {}
        self.easy_access = {}
        self.complete_idx = None
        self.tuple_dict = {}

    def __str__(self):
        """
        String representation of Pattern class.
        :return: stored DataFrame
        """
        if self.pattern_store:
            return str(self.pattern_store["result"])

    def _check_complete_row(self, row):
        """
        Determines whether a row consists out of only 1s.
        
        Parameters
        ----------
        row: row of pandas DataFrame (i.e. results table)
        
        Returns
        -------
        int: index if all 0; -1 otherwise
        """
        counter = 0
        break_val = len(row) - 1
        for val in row:
            if val == 1:
                counter += 1
        if counter == break_val:
            return int(row.name)
        return -1

    def _compute_pattern(self, data, nan_values="", verbose=False, unique_instances=10):
        """
        Function that checks for missing values and prints out 
        a quick table of a summary of missing values.
        Includes pattern overview and counts of missing values by column  

        Parameters
        ----------
        data: pandas DataFrame

        Returns
        -------
        return_dict: dict with keys 'table' and 'indices'
                    'table': pandas DataFrame with pattern overview
                    'indices': dict with indices list
        """
        # NaN Values
        nan_vals = [""]

        # Add additional custom NaN Values
        # Check first if entry is list, otherwise turn into list
        if not isinstance(nan_values, list):
            nan_values = [nan_values]
        # Iterate over nan_values parameter and add to list of nan-values
        for nv in nan_values:
            nan_vals.append(nv)

        # Store column names of data set for later use
        self.column_names = data.columns
        self.missing_per_column = [0] * len(self.column_names)

        # Iteration via apply - stores results in self.result_pattern_temp
        data.apply(self.row_nan_pattern, axis=1)

        # Beautification of result
        result_pattern = pd.DataFrame.from_dict(self.result_pattern_temp, orient='index')
        final_result = []
        index_list = []
        for tuple_val in result_pattern.index:
            # Get index label from tuple_counter_dict
            index_list.append(self.tuple_counter_dict_temp[tuple_val])
            # Store pattern as list per column
            final_result.append(list(tuple_val))
        final_result = pd.DataFrame(final_result)
        final_result.columns = data.columns
        final_result["Count"] = result_pattern.values
        final_result.index = index_list
        final_result.sort_values("Count", ascending=False, inplace=True)
        old_indices = self.pattern_index_store_temp
        new_indices = {}
        new_tuple_dict = {}
        # Rearrange values for better ordering (from 0 to n)
        tuple_list = [tuple(x) for x in final_result.drop('Count', axis=1).values]
        for old, new in zip(final_result.index, range(len(final_result))):
            new_indices[new] = old_indices[old]
            new_tuple_dict[new] = tuple_list[new]

        self.tuple_dict = new_tuple_dict
        self.pattern_index_store_temp = new_indices
        final_result.reset_index(inplace=True, drop=True)

        # Store result in object
        self.pattern_store_temp["result"] = final_result

        return_dict = {"table": final_result,
                       "indices": self.pattern_index_store_temp}

        variable_store = self._get_discrete_and_continuous(data, unique_instances)
        self.discrete_variables, self.continuous_variables = variable_store['discrete'], variable_store['continuous']

        return return_dict

    def _compute_pattern_old(self, data, nan_values="", verbose=False):
        """
        Function that checks for missing values and prints out 
        a quick table of a summary of missing values.
        Includes pattern overview and counts of missing values by column  

        Parameters
        ----------
        data: pandas DataFrame

        Returns
        -------
        return_dict: dict with keys 'table' and 'indices'
                    'table': pandas DataFrame with pattern overview
                    'indices': dict with indices list
        """
        data_cols = data.columns

        # Stores the NaN pattern
        result_pattern = {}

        # Stores the count of NaN values per column
        result_columns = {}

        # Initialize storage for col
        for column in data_cols:
            result_columns[column] = 0

        # NaN Values
        nan_vals = [""]

        # Add additional custom NaN Values
        # Check first if entry is list, otherwise turn into list
        if not isinstance(nan_values, list):
            nan_values = [nan_values]
        # Iterate over nan_values parameter and add to list of nan-values
        for nv in nan_values:
            nan_vals.append(nv)

        pattern_index_store = {}
        tuple_counter = 0
        tuple_counter_dict = {}

        # Iterate over every row
        # TODO: Work with apply and row_nan_pattern function
        #data.apply(self.row_nan_pattern)

        for row_idx, row in data.iterrows():
            tmp_label = []
            for idx, value in enumerate(row):
                # For each value, check if NaN
                if self.nan_checker.is_nan(value):
                    # Add true-indicator to label
                    tmp_label.append('NaN')
                    # Count appearance for column
                    result_columns[data_cols[idx]] += 1
                else:
                    # Add false-indicator to label
                    tmp_label.append(1)
            # Check if tuple already exists, if so: increase count for label
            if tuple(tmp_label) in result_pattern:
                result_pattern[tuple(tmp_label)] += 1
                # Get corresponding label number from dict
                tuple_label = tuple_counter_dict[tuple(tmp_label)]
                pattern_index_store[tuple_label].append(row_idx)
            # else: tuple hasn't been seen yet
            else:
                result_pattern[tuple(tmp_label)] = 1
                tuple_counter_dict[tuple(tmp_label)] = tuple_counter
                # Add first row id to pattern_index_store
                pattern_index_store[tuple_counter] = [row_idx]
                tuple_counter += 1

        # Transform dict into DataFrame
        result_pattern = pd.DataFrame.from_dict(result_pattern, orient='index')
        final_result = []
        index_list = []
        for tuple_val in result_pattern.index:
            # Get index label from tuple_counter_dict
            index_list.append(tuple_counter_dict[tuple_val])
            # Store pattern as list per column
            final_result.append(list(tuple_val))
        final_result = pd.DataFrame(final_result)
        final_result.columns = data_cols
        final_result["Count"] = result_pattern.values
        final_result.index = index_list
        final_result.sort_values("Count", ascending=False, inplace=True)

        old_index = index_list

        final_result.reset_index(drop=True, inplace=True)
        new_index = range(len(final_result))

        print old_index
        print new_index

        new_pattern_index_store = {}
        # Transform pattern index store
        for old, new in zip(old_index, new_index):
            new_pattern_index_store[new] = pattern_index_store[old]
        # print pattern_index_store.keys()

        if verbose:
            print final_result
            print "\n"
            print "Column \t\t NaN Count"
            print "-" * 30
            for col in data_cols:
                print "{}: \t {}".format(col, result_columns[col])

        # Store in object
        self.pattern_store["result"] = final_result
        self.pattern_store["columns"] = result_columns
        self.pattern_index_store = new_pattern_index_store

        return_dict = {"table": final_result,
                       "indices": new_pattern_index_store}

        variable_store = self._get_discrete_and_continuous(data)
        self.discrete_variables, self.continuous_variables = variable_store['discrete'], variable_store['continuous']

        return return_dict

    @staticmethod
    def _is_discrete(tmpdata, unique_instances):
        """
        Determines on dtype and by counting unique instances whether a column 
        contains categorical or continuous values.
        """
        if tmpdata.dtypes == 'object':
            return True
        elif len(tmpdata.unique()) < unique_instances:
            return True
        else:
            return False

    @staticmethod
    def _get_discrete_and_continuous(tmpdata, unique_instances):
        discrete_selector = []
        continuous_selector = []
        for col in tmpdata.columns:
            if Pattern._is_discrete(tmpdata[col], unique_instances):
                discrete_selector.append(col)
            else:
                continuous_selector.append(col)

        return {'discrete': discrete_selector,
                'continuous': continuous_selector}

    @staticmethod
    def _get_index_and_pattern(row):
        tmplabel = []
        rowidx = row.index
        for cell_idx, cell_value in enumerate(row):
            print cell_idx, cell_value
            # For each value, check if NaN
            if NanChecker.is_nan(cell_value):
                # Add true-indicator to label
                tmplabel.append('NaN')
                # Count appearance for column (not needed right now)
                # result_columns[data_cols[cell_idx]] += 1
            else:
                # Add false-indicator to label
                tmplabel.append(1)
        return rowidx[0], tmplabel

    def _get_missing_value_percentage(self, data, importance_filter=False):
        return_table = pd.DataFrame(self.missing_per_column)
        return_table.index = self.column_names
        return_table.columns = ["Missing"]
        return_table["Complete"] = len(data) - return_table["Missing"]
        return_table["Percentage"] = (return_table["Missing"] / len(data))
        return_table["Percentage"] = pd.Series(
            ["{0:.2f} %".format(val * 100) for val in return_table["Percentage"]], index=return_table.index)
        return_table.sort_values("Missing", inplace=True)
        if importance_filter:
            return_table = return_table[return_table["Missing"] > 0]
        return return_table[["Complete", "Missing", "Percentage"]]

    def _store_tuple(self, tup, row_idx, tmp_col_names):
        if tup in self.result_pattern_temp:
            self.result_pattern_temp[tup] += 1
            # Get corresponding label number from dict
            tuple_label = self.tuple_counter_dict_temp[tup]
            self.pattern_index_store_temp[tuple_label].append(row_idx)
        # else: tuple hasn't been seen yet
        else:
            # Enter only if variable exists
            if tmp_col_names:
                self.easy_access[tup] = tmp_col_names
            self.result_pattern_temp[tup] = 1
            self.tuple_counter_dict_temp[tup] = self.tuple_counter_temp
            # Add first row id to pattern_index_store
            self.pattern_index_store_temp[self.tuple_counter_temp] = [row_idx]
            self.tuple_counter_temp += 1

    def get_complete_id(self):
        """
        Returns all ids that are complete.
        :return: list - indices of complete cases
        """
        return self.complete_idx

    def get_column_name(self, patter_no):
        """
        Return the column name(s) of a certain NaN-pattern.
        :param patter_no: int - index of pattern
        :return: list
        """
        return self.easy_access[self.tuple_dict[patter_no]]

    def get_complete_indices(self):
        """
        Function to determine complete cases based on results table. 
        Leverages pre-computed information and is quicker than dropna method.
        
        Returns
        -------
        array : indices list of complete cases
        """
        complete_pattern_no = self.get_pattern().apply(self._check_complete_row, axis=1).max()
        self.complete_idx = complete_pattern_no
        if complete_pattern_no >= 0:
            return self.get_pattern_indices(complete_pattern_no)
        else:
            raise ValueError("All instances seem to have missing values.")

    def get_continuous(self):
        """
        Returns copy of continuous variable names. 
        :return: list
        """
        if self.continuous_variables:
            return list(self.continuous_variables)
        else:
            raise ValueError("Variables aren't analzed yet.")

    def get_discrete(self):
        """
        Returns copy of discrete variable names. 
        :return: list
        """
        if self.discrete_variables:
            return list(self.discrete_variables)
        else:
            raise ValueError("Variables aren't analzed yet.")

    def get_pattern(self, data=None, unique_instances=10):
        """
        Returns NaN-patterns based on primary computation or
        initiates new computation of NaN-patterns.
        
        Parameters
        ----------
        data: pd.DataFrame
        
        Returns
        -------
        pd.DataFrame with NaN-pattern overview
        """
        # If pattern is already computed, return stored result
        if self.pattern_store_temp:
            return self.pattern_store_temp["result"]
        # compute new pattern analysis
        elif not data.empty:
            return self._compute_pattern(data, unique_instances)['table']
        else:
            raise ValueError("No pattern stored and missing data to compute pattern.")

    def get_single_nan_idx(self):
        """
        Returns all pattern indices of single nans
        :return: 
        """
        tmp = self.get_pattern().drop('Count', axis=1)
        tmp[tmp == 'NaN'] = 0
        tmp = 10 - tmp.sum(axis=1)
        tmp = tmp[tmp == 1]
        return tmp.index

    def get_pattern_indices(self, pattern_no):
        """
        Returns data points for a specific pattern_no for further
        investigation.

        Parameters
        ----------
        pattern_no: index int value that indicates pattern

        Returns
        -------
        self.data: data points that have a certain pattern
        """
        if not self.pattern_index_store_temp:
            raise ValueError("Pattern needs to be computed first.")
        if pattern_no not in self.pattern_index_store_temp:
            raise ValueError("Pattern index not in store.")

        return self.pattern_index_store_temp[pattern_no]

    def remove_pattern(self, pattern_no):
        """ Removes a certain pattern. Deletes dictionary entry in the pattern index store
        as well as drops the entry in the results table.
        
        Parameters
        ----------
        pattern_no: index int value that indicates pattern
        
        Returns
        -------
        None
        """
        for col in self.get_column_name(pattern_no):
            # search for index of column in missing summary list
            decrease_pointer = 0
            for pointer in range(len(self.column_names)):
                if self.column_names[pointer] == col:
                    # decrease missing values by count of pattern values
                    decrease_pointer = pointer
                    pattern_table = self.pattern_store_temp["result"]
                    decrease_value = pattern_table[pattern_table.index == pattern_no]["Count"]
                    self.missing_per_column[decrease_pointer] -= int(decrease_value)
        del(self.pattern_index_store_temp[pattern_no])
        self.pattern_store_temp["result"].drop(pattern_no, axis=0, inplace=True)

    def row_nan_pattern(self, row):
        """
        Function to evaluate row on its NaN value patterns.
        Works with is_nan function to determine whether a value is empty or not.

        Parameters
        ----------
        row: any row of a data set

        Returns
        -------
        tuple with pattern indicator
        """
        tmp_label = []
        tmp_nan_col_idc = []
        tmp_counter = 0
        tmp_col_lists = []
        for idx, value in enumerate(row):
            # For each value, check if NaN
            if self.nan_checker.is_nan(value):
                self.missing_per_column[idx] += 1
                # Add NaN-indicator to label
                tmp_label.append('NaN')
                # Store column indicators
                tmp_nan_col_idc.append(tmp_counter)
                tmp_col_lists.append(self.column_names[tmp_counter])
            else:
                # Add complete-indicator to label
                tmp_label.append(1)
            tmp_counter += 1
        try:
            self.store_tuple_columns[tuple(tmp_label)] = tmp_nan_col_idc
            self._store_tuple(tuple(tmp_label), row.name, tmp_col_lists)
        # in case of list
        except AttributeError:
            return tuple(tmp_label)


class Impyter:
    """
    Value imputation class.
    
    Parameters
    ----------
    data = pandas DataFrame
    
    Examples
    ----------
    Importing DataFrame from numpy ndarray:
    
    >>> imputer = Impyter(np.random.randint(low=0, high=10, size=(4,4)))
    >>> imputer
       0  1  2  3
    0  1  5  1  1
    1  1  9  9  4
    2  5  7  2  1
    3  9  7  5  3
    
    Testing list for NaN values
    
    >>> nan_array = ["Test", None, '', 23, [None, "42"]]
    >>> imputer = Impyter()
    >>> print imputer.is_nan(nan_array)
    
    [False, True, True, False, [True, False]]
    
    """

    def __init__(self, data=None):
        self.data = None
        self.load_data(data)
        # initialize machine learning estimator
        self.clf = {}
        self.pattern_log = Pattern()
        self.model_log = {}
        self.result = None

    def __str__(self):
        """
        String representation of Impyter class.
        :return: stored DataFrame
        """
        if self.data is not None:
            return str(self.data)

    @staticmethod
    def _data_check(data):
        """
        Checks if data is pandas DataFrame. Otherwise the data will be transformed.
        """
        # perform instance check on data if available in constructor
        if isinstance(data, pd.DataFrame):
            return data
        # if data is not a DataFrame, try turning it into one
        else:
            try:
                return_data = pd.DataFrame(data)
                return return_data
            except ValueError as e:
                print "Value Error: {}".format(e)
                return pd.DataFrame()

    def load_data(self, data):
        """
        Function to load data into Impyter class.
        to reload and erase not needed information.
        
        :param data: preferably pandas DataFrame 
        """
        # check if data is set in constructor otherwise load empty set.
        if data is None:
            data = pd.DataFrame()
        else:
            data = self._data_check(data)

        self.data = data.copy()
        self.pattern_log = Pattern()
        #return data

    def pattern(self):
        """
        Returns missing value patterns of data set.
        """
        if self.data.empty:
            raise ValueError("Error: Load data first.")
        else:
            return self.pattern_log.get_pattern(self.data)

    def get_pattern_column_name(self, pattern_no):
        tmp = self.pattern()
        return tmp[tmp.index == pattern_no]

    def get_pattern(self, pattern_no):
        """
        Returns data points for a specific pattern_no for further
        investigation.

        Parameters
        ----------
        pattern_no: index int value that indicates pattern

        Returns
        -------
        self.data: data points that have a certain pattern
        """
        return self.data[self.data.index.isin(
            self.pattern_log.get_pattern_indices(pattern_no))]

    def get_data(self):
        return self.data

    def get_result(self):
        if self.result is not None:
            return self.result.copy()
        else:
            raise ValueError("Need to impute values first.")

    def get_missing_summary(self, importance_filter=False):
        """
        Shows simple overview of missing values.
        :param importance_filter: Show only features with at least one missing value.
        :return: pd.DataFrame
        """
        return self.pattern_log._get_missing_value_percentage(self.data, importance_filter)

    def get_model(self, model_no):
        return self.model_log[model_no]

    def get_complete_old(self):
        """
        Old but easy to read method to get complete indices. 
        """
        return self.data.dropna().index

    def drop_pattern(self, pattern_no, inplace=False):
        temp_patterns = self.pattern_log.get_pattern_indices(pattern_no)

        if inplace:
            # Drop self.data with overwrite function
            self.data = self.data[~self.data.index.isin(temp_patterns)]
            # Delete indices in pattern_log
            self.pattern_log.remove_pattern(pattern_no)
            return self.data

        return self.data[~self.data.index.isin(temp_patterns)].copy()

    def print_pattern(self, data=None):
        """
        Counts individual NaN patterns and returns them in a dictionary.
        :return: dict
        """
        if data is None:
            data = self.data

        return self.pattern_log.print_pattern(data)

    def load_model(self, model):
        """
        Load a stored machine learning model to perform value imputation.
        :param model: pickle object or filename of model. 
        """
        try:
            self.clf = joblib.load(model)
        except IOError as e:
            print "File not found: {}".format(e)

    def one_hot_encode(self, data):
        """
        Uses pandas get_dummies method to return a one-hot-encoded
        DataFrame.
        :param data: 
        :return: pd.DataFrame - with one-hot-encoded categorical values
        """
        return_table = pd.DataFrame(index=data.index)

        for col, col_data in data.iteritems():
            # If data type is categorical, convert to dummy variables
            if col_data.dtype == object:
                col_data = pd.get_dummies(col_data, prefix=col + "_ohe")

            # Collect the revised columns
            return_table = return_table.join(col_data)
        return return_table

    def one_hot_decode(self, data):
        """
        Decodes one-hot-encoded features into single column again.
        Generally speaking, this function inverses the one-hot-encode function. 
        :return: pd.DataFrame - data set with collapsed information.
        """
        all_columns = data.columns
        ohe_selector = []
        for col in all_columns:
            if '_ohe_' in col:
                ohe_selector.append(col)
        encoded_data = data[ohe_selector].copy()

        ohe_columns = ohe_selector
        unique_cols = []
        ohe_column_transform_val = []
        for col in ohe_columns:
            if '_ohe_' in col:
                column_split = col.split('_ohe_')
                if column_split[0] not in unique_cols:
                    unique_cols.append(column_split[0])
                ohe_column_transform_val.append(column_split[1])

        list_of_lists = []
        for idx, row in encoded_data.iterrows():
            tmp_list = []
            for value in range(len(row)):
                if row[value] == 1:
                    tmp_list.append(ohe_column_transform_val[value])
            list_of_lists.append(tmp_list)
        return_table = pd.DataFrame(list_of_lists)
        return_table.columns = unique_cols
        data.drop(ohe_selector, inplace=True, axis=1)
        data = pd.concat([data, return_table], axis=1)
        return data


    def save_model(self, name=None):
        """
        Save a learned machine learning model to disk.
        :param name: Name of file.  
        """
        if name is None:
            name = str(date.today()) + "-impyte-mdl.pkl"
            print name
        joblib.dump(self.clf, name)

    def impute(self,
               data=None,
               cv=5,
               verbose=True,
               classifier='rf',
               multi_nans=False,
               one_hot_encode=True,
               auto_scale=True,
               recursive=False):
        """
        data: data to be imputed
        cv: Amount of cross-validation runs.
        verbose: Boolean value, whether prediction results should be printed out.
        classifier: 'rf: Random Forest', 
                    'svr: Support Vector Regression', 
                    'sgd: Stochastic Gradient Descent'
                    'knn: KNearest Neighbor Regressor'
                    'bayes: Bayesian Ridge Regressor',
                    'dt: Decision Tree Regressor',
                    'gbr: Gradient Boosting Regressor',
                    'mlp: Multi-layer Perceptron Regressor (neural network)'
        multi_nans: Boolean indicator if data points with multiple NaN values should be imputed as well
        one_hoe_encode: Boolean - if set to True one-hot-encoding of categorical variables happens
        auto_scale: Boolean - if set to True continuous variables are automatically scaled 
                    and transformed back after imputation.
        recursive: Boolean - if set to True predicted values are being used to further train and predict multi-nans
        """
        if data is None:
            data = self.data
        if not isinstance(data, pd.DataFrame):
            raise ValueError('Input data has wrong format. pd.DataFrame expected.')

        # Decide which classifier to use and initialize
        if classifier is not None:
            if classifier == 'rf':
                self.clf["Regression"] = RandomForestRegressor()
                self.clf["Classification"] = RandomForestClassifier()
            elif classifier == 'bayes':
                self.clf["Regression"] = BayesianRidge()
                self.clf["Classification"] = GaussianNB()
            elif classifier == 'dt':
                self.clf["Regression"] = DecisionTreeRegressor()
                self.clf["Classification"] = DecisionTreeClassifier()
            elif classifier == 'gbr':
                self.clf["Regression"] = GradientBoostingRegressor()
                self.clf["Classification"] = GradientBoostingClassifier()
            elif classifier == 'knn':
                self.clf["Regression"] = KNeighborsRegressor()
                self.clf["Classification"] = KNeighborsClassifier()
            elif classifier == 'mlp':
                self.clf["Regression"] = MLPRegressor()
                self.clf["Classification"] = MLPClassifier()
            elif classifier == 'sgd':
                self.clf["Regression"] = SGDRegressor()
                self.clf["Classification"] = SGDClassifier()
            elif classifier == 'svr':
                self.clf["Regression"] = SVR()
                self.clf["Classification"] = SVC()
            else:
                raise ValueError('Classifier unknown')
        # for debugging purposes
        result_data = self.data.copy()

        # Logic
        # Split into categorical and none categorical variables
        # TODO: Error handling: If data has no pattern yet, simply compute it
        variable_store_cont = self.pattern_log.get_continuous()
        variable_store_disc = self.pattern_log.get_discrete()

        # Get complete cases
        complete_cases = self.data[self.data.index.isin(self.pattern_log.get_complete_indices())]
        complete_idx = self.pattern_log.get_complete_id()

        # impute single nan patterns
        for pattern in self.pattern_log.get_single_nan_idx():
            # filter out complete cases
            if complete_idx != pattern:
                # regressor flag
                regressor = False

                X_train = complete_cases.drop(self.pattern_log.get_column_name(pattern), axis=1)
                if one_hot_encode:
                    X_train = self.one_hot_encode(X_train)

                col_name = self.pattern_log.get_column_name(pattern)[0]
                y_train = complete_cases[col_name]

                # Get data of pattern for prediction
                X_test = self.get_pattern(pattern).drop(col_name, axis=1)

                # Pre-processing of data
                if one_hot_encode:
                    X_test = self.one_hot_encode(X_test)
                if auto_scale:
                    # Scaling for ml pre-processing X_train
                    X_scaler = StandardScaler()
                    y_scaler = StandardScaler()
                    X_train = X_scaler.fit_transform(X_train)
                    X_test = X_scaler.fit_transform(X_test)

                # Select appropriate estimator
                if col_name in self.pattern_log.get_continuous():
                    regressor = True
                    # use regressor
                    scoring = "r2"
                    if auto_scale:
                        y_train = y_scaler.fit_transform(y_train.values.reshape(-1, 1))  # scale continuous
                        y_train = y_train.ravel() # turn 1d array back into matching format
                    model = self.clf["Regression"]
                else:
                    # use classifier
                    scoring = "f1_macro"
                    model = self.clf["Classification"]

                # This is where the imputation happens
                print "Label: {} \t Fitting {}".format(col_name, model.__class__.__name__)
                model.fit(X_train, y_train)
                scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
                #scores = abs(scores)
                print "CV-Scores: {}".format(scores)
                to_append = model.predict(X_test)
                if regressor and auto_scale:
                    to_append = y_scaler.inverse_transform(to_append)  # unscale continuous
                self.model_log[pattern] = ImpyterModel(
                    estimator_name=model.__class__.__name__,
                    model=model,
                    pattern_no=pattern,
                    feature_name=col_name,
                    accuracy=scores)
                indices = self.pattern_log.get_pattern_indices(pattern)
                for pointer, idx in enumerate(indices):
                    #print idx
                    #print pointer
                    #print to_append[pointer]
                    result_data.at[idx, col_name] = to_append[pointer]
                #print indices[:2]
                #print to_append[:2]

        self.result = result_data

        return result_data


class ImpyterModel:
    def __init__(self, estimator_name, model=None, pattern_no=None, feature_name=None, accuracy=None):
        self.model = model
        self.pattern_no = pattern_no
        self.feature_name = feature_name
        self.accuracy = accuracy
        self.estimator_name = estimator_name

    def set_model(self, model):
        self.model = model

    def set_pattern(self, pattern):
        self.pattern = pattern

    def set_feature_name(self, feature_name):
        self.feature_name = feature_name

    def set_accuracy(self, accuracy):
        self.accuracy = accuracy

    def get_model(self):
        return self.model

    def get_pattern_no(self):
        return self.pattern_no

    def get_feature_name(self):
        return self.feature_name

    def get_accuracy(self):
        return self.accuracy
