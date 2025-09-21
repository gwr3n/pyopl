import unittest

from pyopl.pyopl_core import GurobiCodeGenerator, OPLLexer, OPLParser


class TestNotEqualRewriteGurobi(unittest.TestCase):
    def gen_code(self, opl_src: str):
        lexer = OPLLexer()
        parser = OPLParser()
        ast = parser.parse(lexer.tokenize(opl_src))
        return GurobiCodeGenerator(ast).generate_code()

    def test_boolean_neq_rewritten_to_xor(self):
        opl = """
        dvar boolean a; dvar boolean b;
        minimize 0;
        subject to { a != b; }
        """
        code = self.gen_code(opl)
        # Expect no '!=' in emitted code, and presence of a + b == 1 constraint
        self.assertNotIn("!=", code)
        self.assertIn("addConstr(a + b == 1", code)

    def test_integer_neq_uses_bigM_delta(self):
        opl = """
        dvar int x; dvar int y;
        minimize 0;
        subject to { x != y; }
        """
        code = self.gen_code(opl)
        # Should introduce a binary delta var and two big-M constraints
        self.assertIn("neq_flag_c0", code)
        self.assertIn("addVar(vtype=GRB.BINARY", code)
        # Check pattern of disjunctive separation
        self.assertIn("x - y + 1000000.0 * neq_flag_c0 >= 1", code)
        self.assertIn("y - x + 1000000.0 * (1 - neq_flag_c0) >= 1", code)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
