"""
Module to impute missing values using machine learning algorithms.
author: Andreas Rubin-Schwarz
"""

import math
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.externals import joblib
from datetime import date


class NanChecker:
    """
    Class that checks data set, lists or single
    values for NaN occurrences. 
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

    def pattern(self, data, nan_values="", verbose=False):
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
                # Add first row id to patern_index_store
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
        final_result['Count'] = result_pattern.values
        final_result.index = index_list

        final_result = final_result.sort_values('Count', ascending=False)
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
        self.pattern_index_store = pattern_index_store

        return_dict = {}
        return_dict['table'] = final_result
        return_dict['indices'] = pattern_index_store

        return return_dict

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
        for idx, value in enumerate(row):
            # For each value, check if NaN
            if self.nan_checker.is_nan(value):
                # Add NaN-indicator to label
                tmp_label.append('NaN')
            else:
                # Add complete-indicator to label
                tmp_label.append(1)

        return tuple(tmp_label)

    def print_pattern(self, data):
        """
        Counts individual NaN patterns and returns them in a dictionary.
        :return: dict
        """

        return Counter(data.apply(self.row_nan_pattern, axis=1))


class Imputer:
    """
    Value imputation class.
    
    Parameters
    ----------
    data = pandas DataFrame
    clf = machine learning estimator 
    
    Examples
    ----------
    Importing DataFrame from numpy ndarray:
    
    >>> imputer = Imputer(np.random.randint(low=0, high=10, size=(4,4)))
    >>> imputer
       0  1  2  3
    0  1  5  1  1
    1  1  9  9  4
    2  5  7  2  1
    3  9  7  5  3
    
    Testing list for NaN values
    
    >>> nan_array = ["Test", None, '', 23, [None, "42"]]
    >>> imputer = Imputer()
    >>> print imputer.is_nan(nan_array)
    
    [False, True, True, False, [True, False]]
    
    """

    def __init__(self, data=None):
        # check if data is set in constructor otherwise load empty set.
        if data is None:
            self.data = pd.DataFrame()
        else:
            self.data = self._data_check(data)
        # initialize machine learning estimator
        self.clf = {}
        self.pattern = Pattern()

    def __str__(self):
        """
        String representation of Imputer class.
        :return: stored DataFrame
        """
        if self.data is not None:
            return str(self.data)

    @staticmethod
    def _data_check(data):
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
        Function to load data into Imputer class.
        
        :param data: preferably pandas DataFrame 
        :return: pandas DataFrame
        """
        self.data = self._data_check(data)

    def print_pattern(self, data=None):
        """
        Counts individual NaN patterns and returns them in a dictionary.
        :return: dict
        """
        if data is None:
            data = self.data

        return self.pattern_logger.print_pattern(data)

    def load_model(self, model):
        """
        Load a stored machine learning model to perform value imputation.
        :param model: pickle object or filename of model. 
        """
        try:
            self.clf = joblib.load(model)
        except IOError as e:
            print "File not found: {}".format(e)

    def save_model(self, name=None):
        """
        Save a learned machine learning model to disk.
        :param name: Name of file.  
        """
        if name is None:
            name = str(date.today()) + "-impyte-mdl.pkl"
            print name
        joblib.dump(self.clf, name)
