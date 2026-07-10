import unittest


class TestHelloWorld(unittest.TestCase):
    def test1(self):
        str = "Hello, World!"
        self.assertEqual(str, "Hello, World!")


if __name__ == "__main__":
    unittest.main()
