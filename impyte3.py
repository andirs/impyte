"""
Module to impute missing values using machine learning algorithms.
author: Andreas Rubin-Schwarz
"""

import math
import numpy as np
import pandas as pd
import warnings

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
    
    Examples
    ----------
    Testing list for NaN values
    
    >>> nan_array = ["Test", None, '', 23, [None, "42"]]
    >>> nan_checker = impyte.NanChecker()
    >>> print nan_checker.is_nan(nan_array)
    
    [False, True, True, False, [True, False]]
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
                elif isinstance(item, str):
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
        elif isinstance(data, str):
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
        self.column_names = []
        self.complete_idx = None
        self.continuous_variables = []
        self.discrete_variables = []
        self.easy_access = {}
        self.missing_per_column = None
        self.nan_checker = NanChecker()
        self.pattern_col_names = {}
        self.pattern_index_store = {}
        self.pattern_store = {}
        self.store_tuple_columns = {}
        self.result_pattern = {}
        self.tuple_counter = 0
        self.tuple_counter_dict = {}
        self.tuple_dict = {}
        self.unique_instances = 10

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

    def _compute_pattern(self, data, nan_values="", unique_instances=10):
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
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input has to be DataFrame")
        if data.empty:
            raise ValueError("DataFrame can't be empty")

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

        # Iteration via apply - stores results in self.result_pattern
        data.apply(self.row_nan_pattern, axis=1)

        # Beautification of result
        result_pattern = pd.DataFrame.from_dict(self.result_pattern, orient='index')
        final_result = []
        index_list = []
        for tuple_val in result_pattern.index:
            # Get index label from tuple_counter_dict
            index_list.append(self.tuple_counter_dict[tuple_val])
            # Store pattern as list per column
            final_result.append(list(tuple_val))
        final_result = pd.DataFrame(final_result)
        final_result.columns = data.columns
        final_result["Count"] = result_pattern.values
        final_result.index = index_list
        final_result.sort_values("Count", ascending=False, inplace=True)
        old_indices = self.pattern_index_store
        new_indices = {}
        new_tuple_dict = {}
        # Rearrange values for better ordering (from 0 to n)
        tuple_list = [tuple(x) for x in final_result.drop('Count', axis=1).values]
        for old, new in zip(final_result.index, range(len(final_result))):
            new_indices[new] = old_indices[old]
            new_tuple_dict[new] = tuple_list[new]

        self.tuple_dict = new_tuple_dict
        self.pattern_index_store = new_indices
        final_result.reset_index(inplace=True, drop=True)

        # Store result in object
        self.pattern_store["result"] = final_result

        return_dict = {"table": final_result,
                       "indices": self.pattern_index_store}

        variable_store = self._get_discrete_and_continuous(data, unique_instances)
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
            print(cell_idx, cell_value)
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

    @staticmethod
    def _get_unique_vals(data):
        unique_vals = []
        for col in data.columns:
            unique_vals.append(len(data[col].unique()))

        return unique_vals

    def get_missing_value_percentage(self, data, importance_filter=False):
        return_table = pd.DataFrame(self.missing_per_column)
        return_table.index = self.column_names
        return_table.columns = ["Missing"]
        return_table["Unique"] = self._get_unique_vals(data)
        return_table["Complete"] = len(data) - return_table["Missing"]
        return_table["Percentage"] = (return_table["Missing"] / len(data))
        return_table["Percentage"] = pd.Series(
            ["{0:.2f} %".format(val * 100) for val in return_table["Percentage"]], index=return_table.index)
        return_table.sort_values("Missing", inplace=True)
        if importance_filter:
            return_table = return_table[return_table["Missing"] > 0]
        return return_table[["Complete", "Missing", "Percentage", "Unique"]]

    def _store_tuple(self, tup, row_idx, tmp_col_names):
        if tup in self.result_pattern:
            self.result_pattern[tup] += 1
            # Get corresponding label number from dict
            tuple_label = self.tuple_counter_dict[tup]
            self.pattern_index_store[tuple_label].append(row_idx)
        # else: tuple hasn't been seen yet
        else:
            # Enter only if variable exists
            if tmp_col_names:
                self.easy_access[tup] = tmp_col_names
            self.result_pattern[tup] = 1
            self.tuple_counter_dict[tup] = self.tuple_counter
            # Add first row id to pattern_index_store
            self.pattern_index_store[self.tuple_counter] = [row_idx]
            self.tuple_counter += 1

    def get_complete_id(self):
        """
        Returns all ids that are complete.
        :return: list - indices of complete cases
        """
        if not self.complete_idx:
            self.get_complete_indices()
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
        return list(self.continuous_variables)

    def get_discrete(self):
        """
        Returns copy of discrete variable names. 
        :return: list
        """
        return list(self.discrete_variables)

    def get_pattern(self, data=None, unique_instances=10):
        """
        Returns NaN-patterns based on primary computation or
        initiates new computation of NaN-patterns.
        
        Parameters
        ----------
        data: pd.DataFrame
        unique_instances: int - determines how many unique values are needed to count as continuous variable
        
        Returns
        -------
        pd.DataFrame with NaN-pattern overview
        """
        # If pattern is already computed, return stored result
        if self.pattern_store:
            return self.pattern_store["result"]
        # compute new pattern analysis
        elif not data.empty:
            return self._compute_pattern(data, unique_instances)['table']
        else:
            raise ValueError("No pattern stored and missing data to compute pattern.")

    def get_single_nan_pattern_nos(self):
        """
        Returns all pattern indices of single nans
        :return: 
        """
        return self.get_multi_nan_pattern_nos(multi=False)

    def get_multi_nan_pattern_nos(self, multi=True):
        """
        Returns all pattern indices of multi-nans or single-nans
        :return: 
        """
        # TODO: More beautiful way of refactoring with get_single_nan_pattern_nos
        tmp = self.get_pattern().drop('Count', axis=1)
        tmp.replace('NaN', 0, inplace=True)
        tmp = len(tmp.columns) - tmp.sum(axis=1)
        if multi:
            tmp = tmp[tmp > 1]
        else:
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
        if not self.pattern_index_store:
            raise ValueError("Pattern needs to be computed first.")
        if pattern_no not in self.pattern_index_store:
            raise ValueError("Pattern index not in store.")

        return self.pattern_index_store[pattern_no]

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
            for pointer in range(len(self.column_names)):
                if self.column_names[pointer] == col:
                    # decrease missing values by count of pattern values
                    pattern_table = self.pattern_store["result"]
                    decrease_value = pattern_table[pattern_table.index == pattern_no]["Count"]
                    self.missing_per_column[pointer] -= int(decrease_value)
        del(self.pattern_index_store[pattern_no])
        self.pattern_store["result"].drop(pattern_no, axis=0, inplace=True)

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
    
    """

    def __init__(self, data=None):
        self.data = None  # stores original data
        self.result = None  # stores result data set - in beginning a copy of data
        self.load_data(data)  # loads or initializes data set
        self.clf = {}  # stores classifier - deprecated
        self.pattern_log = Pattern()  # stores Pattern() object for data set
        self.model_log = {}  # stores all models once impute has been run
        self.error_string = ""
        self.pattern_predictor_dict = {}

    def __str__(self):
        """
        String representation of Impyter class.
        :return: stored DataFrame
        """
        if self.data is not None:
            return str(self.data)

    @staticmethod
    def _set_display_options(length, cols=True):
        if cols:
            pd.set_option("display.max_columns", length)
        else:
            pd.set_option("display.max_rows", length)

    @staticmethod
    def _get_display_options(cols=True):
        if cols:
            return pd.get_option("display.max_columns")
        else:
            return pd.get_option("display.max_rows")

    @staticmethod
    def _data_check(data):
        """
        Checks if data is pandas DataFrame. Otherwise the data will be transformed.
        """
        # perform instance check on data if available in constructor
        if not isinstance(data, pd.DataFrame):
            # if data is not a DataFrame, try turning it into one
            try:
                return_data = pd.DataFrame(data)
                return return_data
            except ValueError as e:
                print("Value Error: {}".format(e))
                return pd.DataFrame()
        return data

    def drop_imputation(self, threshold, verbose=True):
        models = dict(self.model_log)
        for pattern_no in models:
            model = self.get_model(pattern_no)
            for idx in range(len(model.get_model())):
                if 'r2' in model.get_scoring()[idx]:
                    cur_threshold = threshold[1]
                else:
                    cur_threshold = threshold[0]
                # average score in case it's a multi-nan pattern
                avg_score = np.mean(model.get_score()[idx])
                if avg_score < cur_threshold:
                    if verbose:
                        print("Dropping pattern {} ({} < {} {})".format(
                            pattern_no, avg_score, cur_threshold, model.get_scoring()[0]))

                    self.drop_pattern(pattern_no, inplace=True)
                    del (self.model_log[pattern_no])
                    break  # only one value below threshold is enough to discard pattern

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
        self.result = self.data.copy()
        self.pattern_log = Pattern()

    def pattern(self):
        """
        Returns missing value patterns of data set.
        """
        if self.data.empty:
            raise ValueError("Error: Load data first.")
        else:
            return_table = self.pattern_log.get_pattern(self.data)
            if len(return_table.columns) > Impyter._get_display_options():
                Impyter._set_display_options(len(return_table.columns))
            if len(return_table) > Impyter._get_display_options(False):  # check if too many rows to display
                Impyter._set_display_options(len(return_table), False)
        return return_table

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

    def get_summary(self, importance_filter=True):
        """
        Shows simple overview of missing values.
        :param importance_filter: Show only features with at least one missing value.
        :return: pd.DataFrame
        """
        if not self.pattern_log.pattern_store:
            self.pattern()

        result_table = self.pattern_log.get_missing_value_percentage(self.data, importance_filter)
        return result_table

    def get_model(self, pattern_no):
        if pattern_no in self.model_log:
            return self.model_log[pattern_no]
        else:
            raise ValueError("There is no model for pattern {}".format(pattern_no))

    def drop_pattern(self, pattern_no, inplace=False):
        temp_patterns = self.pattern_log.get_pattern_indices(pattern_no)

        if inplace:
            # Drop self.data with overwrite function
            self.result = self.result[~self.result.index.isin(temp_patterns)]
            # Delete indices in pattern_log
            self.pattern_log.remove_pattern(pattern_no)
            return self.result

        return self.data[~self.data.index.isin(temp_patterns)].copy()

    def map_model_to_pattern(mdl):
        pred_variables = mdl.get_predictor_variables()
        feature = mdl.get_feature_name()
        pattern_no = None

        def compare_features(list1, list2):
            return Counter(list1) == Counter(list2)

        for i in self.pattern_log.store_tuple_columns:
            if compare_features(feature, self.pattern_log.store_tuple_columns[i]):
                pattern_no = self.pattern_log.tuple_counter_dict[i]
                break
        if pattern_no and compare_features(pred_variables, self.pattern_predictor_dict[pattern_no]):
            return pattern_no
        else:
            return None

    def load_model(self, pattern_no, model):
        """
        Load a stored machine learning model to perform value imputation.
        :param model: pickle object or filename of model. 
        """
        try:
            mdl = joblib.load(model)
            mdl_pattern_no = self.map_model_to_pattern(mdl)
            if pattern_no and mdl_pattern_no:
                self.model_log[self.map_model_to_pattern(mdl)] = mdl
            elif pattern_no and not mdl_pattern_no:
                raise ValueError("Model and pattern seem to be inconsistent")
            else:
                self.model_log = joblib.load(model)
        except IOError as e:
            print("File not found: {}".format(e))

    def one_hot_encode(self, data, verbose=False):
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
                if verbose > 3:
                    print("Getting dummies for {} (Unique: {})".format(col, len(data[col].unique())))
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

    def save_model(self, pattern_no=None, name=None):
        """
        Save a learned machine learning model to disk.
        :param name: Name of file.  
        """
        name_str = ""
        if pattern_no is None:
            model = self.model_log
        else:
            model = self.get_model(pattern_no)
            name_str = "pattern_{}_".format(pattern_no)
        if name is None:
            name = name_str + str(date.today()) + "-impyte-mdl.pkl"
            print(name)
        joblib.dump(model, name)

    def ensemble(self, estimator_list=["rf", "dt"]):
        """
        Exhaustive search for best estimator to predict a certain feature. 
        Work in progress and beta method. Needs further work on summaries and plotting.
        :return: 
        """

        imp = Impyter()
        imp.load_data(self.data.copy())
        # ["rf", "svm", "sgd", "knn", "bayes", "dt", "gb", "mlp"]
        if not estimator_list:
            estimator_list = ["rf", "svm", "sgd", "knn", "bayes", "dt", "gb", "mlp"]
        for estimator in estimator_list:
            _ = self.impute(estimator=estimator)

    @staticmethod
    def _print_header(threshold):
        # print threshold:
        print("{:<30}{:<30}{:<30}".format("Scoring Threshold", "Classification", "Regression"))
        print("=" * 90)
        print("{:<30}{:<30}{:<30}".format("", str(threshold[0]), str(threshold[1])))
        print("")
        print("{:<30}{:<30}{:<30}".format(
            "Pattern: Label",
            "Score",
            "Estimator"))
        print("=" * 90)

    @staticmethod
    def _print_results_line(scores, scoring, pattern, col_name, tmp_error_string, model, error_count):
        score_temp = "{:.3f} ({})".format(np.mean(scores), scoring)
        col_temp = "{}: {}".format(pattern, col_name)
        if tmp_error_string:
            col_temp += " (* {})".format(error_count)

        return "{:<30}{:<30}{:<30} ".format(
            col_temp,
            score_temp,
            model.__class__.__name__)

    def _impute(self, pattern, col_name, X_train, X_test, y_train, one_hot_encode,
                auto_scale, threshold, result_data, cv, verbose_string, verbose):
        global error_count
        # regressor flag
        tmp_error_string = ""
        regressor = False
        if pattern in self.model_log:
            store_models = self.get_model(pattern).get_model()
            store_scores = self.get_model(pattern).get_scores()
            store_scoring = self.get_model(pattern).get_scoring()
            store_estimator_names = self.get_model(pattern).get_estimator_name()
            feature_names = self.get_model(pattern).get_feature_name()
        else:
            store_models, store_scores, store_scoring, store_estimator_names = [], [], [], []
            feature_names = []
        predictor_variables = X_train.columns
        self.pattern_predictor_dict[pattern] = predictor_variables

        # Pre-processing of data
        if one_hot_encode:
            if verbose > 3:
                print("One-hot encoding for {}...".format(col_name))
            test = X_train.append(X_test)
            test = self.one_hot_encode(test, verbose)
            X_train = test[test.index.isin(X_train.index)]
            X_test = test[test.index.isin(X_test.index)]

        # Scaling for ml pre-processing X_train
        if auto_scale:
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
                y_train = y_train.ravel()  # turn 1d array back into matching format
            model = self.clf["Regression"]
            tmp_threshold_cutoff = threshold[1]  # for regression
        else:
            # use classifier
            scoring = "f1_macro"
            model = self.clf["Classification"]
            tmp_threshold_cutoff = threshold[0]  # for classification

        # This is where the imputation happens
        if verbose > 3:
            print("Starting imputation {}...".format(col_name))

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            model.fit(X_train, y_train)
            try:
                scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
            except (ValueError, Warning) as e:
                error_count += 1
                tmp_error_string = "* (" + str(error_count) + ") " + col_name + ": " + str(e)
                self.error_string += tmp_error_string + "\n"
                scores = [0.] * cv

        store_models.append(model)
        store_scores.append(scores)
        store_scoring.append(scoring)
        feature_names.append(col_name)

        # prepare statement line for verbose printout
        if verbose:
            verbose_string = self._print_results_line(
                np.mean(scores), scoring, pattern, col_name, tmp_error_string, model, error_count)

        to_append = model.predict(X_test)
        if regressor and auto_scale:
            to_append = y_scaler.inverse_transform(to_append)  # unscale continuous

        store_estimator_names.append(model.__class__.__name__)

        self.model_log[pattern] = ImpyterModel(
            estimator_name=store_estimator_names,
            model=store_models,
            pattern_no=pattern,
            feature_name=feature_names,
            scores=store_scores,
            scoring=store_scoring,
            predictor_variables=predictor_variables)
        indices = self.pattern_log.get_pattern_indices(pattern)
        if not tmp_threshold_cutoff or tmp_threshold_cutoff <= np.mean(scores):
            verbose_string += " imputed..."
            for pointer, idx in enumerate(indices):
                result_data.at[idx, col_name] = to_append[pointer]
        else:
            verbose_string += " not imputed..."
        if verbose:
            print(verbose_string)

    def impute(self,
               data=None,
               cv=5,
               verbose=True,
               estimator='rf',
               multi_nans=False,
               one_hot_encode=True,
               auto_scale=True,
               threshold=[None, None]):
        """
        data: data to be imputed
        cv: Amount of cross-validation runs.
        verbose: Boolean value, whether prediction results should be printed out.
        estimator:  'rf: Random Forest', 
                    'svm: Support Vector Machine', 
                    'sgd: Stochastic Gradient Descent'
                    'knn: KNearest Neighbor'
                    'bayes: (Naive) Bayes',
                    'dt: Decision Tree',
                    'gb: Gradient Boosting',
                    'mlp: Multi-layer Perceptron (neural network)'
        multi_nans: Boolean indicator if data points with multiple NaN values should be imputed as well
        one_hoe_encode: Boolean - if set to True one-hot-encoding of categorical variables happens
        auto_scale: Boolean - if set to True continuous variables are automatically scaled 
                    and transformed back after imputation.
        threshold: list - classification and regression threshold cut-offs. At this point f1 score and R2.
        """
        if data is None:
            data = self.result
        if not isinstance(data, pd.DataFrame):
            raise ValueError('Input data has wrong format. pd.DataFrame expected.')

        # show warning if less than 50 data points available
        if not data.empty and len(data) < 50:
            with warnings.catch_warnings():
                warnings.simplefilter('always')
                warnings.warn("There might be too few data points for imputation. (Threshold >= 50)\n", UserWarning)

        # If data has no pattern yet, simply compute it
        if not self.pattern_log.pattern_store:
            print("Computing NaN-patterns first ...\n")
            self.pattern()

        # reset error string
        self.error_string = ""

        # print output header
        self._print_header(threshold)

        # Decide which classifier to use and initialize
        if estimator is not None:
            if estimator == 'rf':
                self.clf["Regression"] = RandomForestRegressor()
                self.clf["Classification"] = RandomForestClassifier()
            elif estimator == 'bayes':
                self.clf["Regression"] = BayesianRidge()
                self.clf["Classification"] = GaussianNB()
            elif estimator == 'dt':
                self.clf["Regression"] = DecisionTreeRegressor()
                self.clf["Classification"] = DecisionTreeClassifier()
            elif estimator == 'gb':
                self.clf["Regression"] = GradientBoostingRegressor()
                self.clf["Classification"] = GradientBoostingClassifier()
            elif estimator == 'knn':
                self.clf["Regression"] = KNeighborsRegressor()
                self.clf["Classification"] = KNeighborsClassifier()
            elif estimator == 'mlp':
                self.clf["Regression"] = MLPRegressor()
                self.clf["Classification"] = MLPClassifier()
            elif estimator == 'sgd':
                self.clf["Regression"] = SGDRegressor()
                self.clf["Classification"] = SGDClassifier()
            elif estimator == 'svm':
                self.clf["Regression"] = SVR()
                self.clf["Classification"] = SVC()
            else:
                raise ValueError('Classifier unknown')
        result_data = self.result

        # Get complete cases
        complete_idx = self.pattern_log.get_complete_id()
        complete_cases = self.get_pattern(complete_idx)

        # error string
        global error_count
        error_count = 0

        # impute single nan patterns
        for pattern in self.pattern_log.get_single_nan_pattern_nos():
            tmp_error_string = ""
            # filter out complete cases
            if complete_idx != pattern:
                col_name = self.pattern_log.get_column_name(pattern)[0]
                # Get data of pattern for prediction
                X_train = complete_cases.drop(col_name, axis=1)
                X_test = self.get_pattern(pattern).drop(col_name, axis=1)
                y_train = complete_cases[col_name]
                self._impute(
                    pattern, col_name, X_train, X_test, y_train, one_hot_encode,
                    auto_scale, threshold, result_data, cv, tmp_error_string, verbose)

        # Multi-Nan
        multi_nan_patterns = self.pattern_log.get_multi_nan_pattern_nos()
        if multi_nans and not multi_nan_patterns.empty:
            print("")
            print("Multi nans")
            print("=" * 90)
            for pattern_no in multi_nan_patterns:
                tmp_error_string = ""

                multi_nan_columns = self.pattern_log.get_column_name(pattern_no)
                for col_name in multi_nan_columns:
                    # Get data of pattern for prediction
                    X_train = complete_cases.drop(multi_nan_columns, axis=1)
                    X_test = self.get_pattern(pattern_no).drop(multi_nan_columns, axis=1)
                    y_train = complete_cases[col_name]
                    self._impute(
                        pattern_no, col_name, X_train, X_test, y_train, one_hot_encode,
                        auto_scale, threshold, result_data, cv, tmp_error_string, verbose)

        # print error categories
        if verbose and self.error_string:
            print("\n")
            print(self.error_string)

        self.result = result_data
        return result_data


class ImpyterModel:
    """
    Stores computed Impyter machine learning models.
    """
    def __init__(self, estimator_name,
                 model=None,
                 pattern_no=None,
                 feature_name=None,
                 scores=None,
                 scoring=None,
                 predictor_variables=None):
        self.model = model
        self.pattern_no = pattern_no
        self.feature_name = feature_name
        self.scores = scores
        self.estimator_name = estimator_name
        self.scoring = scoring
        self.predictor_variables = predictor_variables

    def set_model(self, model):
        """
        Setter method to update machine learning model.
        :param model: Machine Learning model
        """
        self.model = model

    def set_pattern(self, pattern_no):
        """
        Setter method, updating pattern_no
        :param pattern_no: int  
        """
        self.pattern_no = pattern_no

    def set_feature_name(self, feature_name):
        """
        Setter method, updates feature names
        :param feature_name: 
        """
        self.feature_name = feature_name

    def set_scores(self, scores):
        self.scores = scores

    def set_predictor_variables(self, predictor_variables):
        """
        Setter method, updates predictor names
        :param predictor_names: 
        """
        self.predictor_variables = predictor_variables

    def get_model(self):
        return self.model

    def get_pattern_no(self):
        return self.pattern_no

    def get_feature_name(self):
        return self.feature_name

    def get_predictor_variables(self):
        return self.predictor_variables

    def get_score(self):
        ret_list = []
        for score_list in self.scores:
            ret_list.append(np.mean(score_list))
        return ret_list

    def get_scores(self):
        return self.scores

    def get_scoring(self):
        return self.scoring

    def get_estimator_name(self):
        return self.estimator_name
