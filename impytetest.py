import unittest
import impyte


class TestNanChecker(unittest.TestCase):

    def setUp(self):
        pass

    def test_is_nan_simple_list(self):
        self.assertEqual(impyte.NanChecker.is_nan(["", 'None', 'NaN']), [True, False, False])

    def test_is_nan_recursive(self):
        self.assertEqual(
            impyte.NanChecker.is_nan(["", None, 'NaN', ["List Value 1", '', None]]), [True, True, False, [False, True, True]])

    # Values can be declared as nan-values
    def test_is_nan_simple_list_nan_vals(self):
        self.assertEqual(
            impyte.NanChecker.is_nan(['NaN', 'Empty', 'None', 'N/A'], nan_vals=['NaN', 'N/A']), [True, False, False, True])

    # Values can be declared as nan-values
    # mixed values are possible as well
    def test_is_nan_simple_list_nan_vals_mix(self):
        self.assertEqual(
            impyte.NanChecker.is_nan(['NaN', 1, 'None', 'N/A'], nan_vals=['NaN', 'N/A']), [True, False, False, True])

if __name__ == "__main__":
    unittest.main()