import sys
sys.path.append('..')  # to enable impyte import
import unittest
import impyte
import numpy as np
import pandas as pd
from impyte import NanChecker
from impyte import Pattern
import pandas.util.testing as pdt
import numpy.testing as npt


def assert_frames_equal(actual, expected, use_close=False):
    """
    Compare DataFrame items by index and column and
    raise AssertionError if any item is not equal.

    Ordering is unimportant, items are compared only by label.
    NaN and infinite values are supported.

    Parameters
    ----------
    actual : pandas.DataFrame
    expected : pandas.DataFrame
    use_close : bool, optional
        If True, use numpy.testing.assert_allclose instead of
        numpy.testing.assert_equal.

    """
    if use_close:
        comp = npt.assert_allclose
    else:
        comp = npt.assert_equal

    assert (isinstance(actual, pd.DataFrame) and
            isinstance(expected, pd.DataFrame)), \
        'Inputs must both be pandas DataFrames.'

    for i, exp_row in expected.iterrows():
        assert i in actual.index, 'Expected row {!r} not found.'.format(i)

        act_row = actual.loc[i]

        for j, exp_item in exp_row.iteritems():
            assert j in act_row.index, \
                'Expected column {!r} not found.'.format(j)

            act_item = act_row[j]

            try:
                comp(act_item, exp_item)
            except AssertionError as e:
                raise AssertionError(
                    e.message + '\n\nColumn: {!r}\nRow: {!r}'.format(j, i))


class TestNanChecker(unittest.TestCase):

    def setUp(self):
        pass

    def test_is_nan_empty(self):
        self.assertEqual(NanChecker.is_nan([]), [])

    def test_is_nan_empty_string(self):
        self.assertEqual(NanChecker.is_nan(""), True)

    def test_is_nan_non_empty_string(self):
        self.assertEqual(NanChecker.is_nan("A"), False)

    def test_is_nan_simple_list(self):
        self.assertEqual(NanChecker.is_nan(["", 'None', 'NaN']), [True, False, False])

    def test_is_nan_recursive(self):
        self.assertEqual(
            NanChecker.is_nan(["", None, 'NaN', ["List Value 1", '', None]]), [True, True, False, [False, True, True]])

    # Values can be declared as nan-values
    def test_is_nan_simple_list_nan_vals(self):
        self.assertEqual(
            NanChecker.is_nan(['NaN', 'Empty', 'None', 'N/A'], nan_vals=['NaN', 'N/A']), [True, False, False, True])

    # Values can be declared as nan-values
    # mixed values are possible as well
    def test_is_nan_simple_list_nan_vals_mix(self):
        self.assertEqual(
            NanChecker.is_nan(['NaN', 1, 'None', 'N/A'], nan_vals=['NaN', 'N/A']), [True, False, False, True])


class TestPattern(unittest.TestCase):

    def setUp(self):
        pass

    def test__compute_pattern(self):
        data = [[1, 1, 1, np.nan], [1, 2, 3, np.nan], [3, np.nan, 4, 5, 6]]
        data = pd.DataFrame(data)
        pattern_log = Pattern()
        result_dict = {0: {0: 1, 1: 1},
                       1: {0: 1, 1: 'NaN'},
                       2: {0: 1, 1: 1},
                       3: {0: 'NaN', 1: 1},
                       4: {0: 'NaN', 1: 1},
                       'Count': {0: 2, 1: 1}}
        result_table = pd.DataFrame(result_dict)
        result_indices = {0: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], 1: [2, 2, 2, 2, 2, 2]}
        real_table = pattern_log._compute_pattern(data)["table"]
        assert_frames_equal(real_table, result_table)

    def test__compute_pattern_empty_df(self):
        data = pd.DataFrame()
        pattern_log = Pattern()
        self.assertRaises(ValueError, pattern_log._compute_pattern, data)

    def test__compute_pattern_no_df(self):
        data = []
        pattern_log = Pattern()
        self.assertRaises(ValueError, pattern_log._compute_pattern, data)

if __name__ == "__main__":
    unittest.main()