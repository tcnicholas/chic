import unittest


class Test1(unittest.TestCase):
    """Test 1."""
    def test_sum(self):
        return 3+3


class Test2(unittest.TestCase):
    """Test 2."""
    def test_sum_triple(self):
        return 3+3+3

    def test_mult(self):
        return 3*3


if __name__=='__main__':
    unittest.main()
