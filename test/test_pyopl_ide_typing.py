import unittest

# Import should not fail regardless of Pillow availability
from pyopl import pyopl_ide_bootstrap
from pyopl.pyopl_ide_bootstrap import OPLIDE


class TestPyOPLIDETyping(unittest.TestCase):
    def test_pillow_optional_imports_exist(self):
        # Module should define these attributes
        self.assertTrue(hasattr(pyopl_ide_bootstrap, "PILImage"))
        self.assertTrue(hasattr(pyopl_ide_bootstrap, "PILImageTk"))

    def test_index_from_pos_mapping(self):
        s = "hello\nworld"
        # pos 0 => line 1, col 0
        self.assertEqual(OPLIDE._index_from_pos(None, s, 0), "1.0")
        # after 'hello' (5), at newline index 5 => still line 1, col 5
        self.assertEqual(OPLIDE._index_from_pos(None, s, 5), "1.5")
        # index 6 is start of second line 'w' => line 2, col 0
        self.assertEqual(OPLIDE._index_from_pos(None, s, 6), "2.0")
        # end of string len=11 => line 2, col len('world')=5
        self.assertEqual(OPLIDE._index_from_pos(None, s, len(s)), "2.5")


if __name__ == "__main__":
    unittest.main()
