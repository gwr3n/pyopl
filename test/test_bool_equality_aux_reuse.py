import unittest

from pyopl.pyopl_core import OPLLexer, OPLParser
from pyopl.scipy_codegen_csc import SciPyCSCCodeGenerator


class TestBoolEqualityAuxReuse(unittest.TestCase):
    def gen(self, src: str):
        lexer = OPLLexer()
        parser = OPLParser()
        ast = parser.parse(lexer.tokenize(src))
        gen = SciPyCSCCodeGenerator(ast)
        gen._build_variables()
        gen._build_objective()
        gen._build_constraints()
        return gen

    def test_reuse_comparison_truth_in_multiple_equalities(self):
        opl = """
        dvar boolean b1; dvar boolean b2; dvar int x; dvar int y;
        minimize 0;
        subject to {
            b1 == (x != y);
            b2 == (x != y);
        }
        """
        gen = self.gen(opl)
        # Count comparison truth vars for x != y (should be 1 reused)
        cmp_flags = [v for v in gen.var_indices if v.startswith("cmp_flag_")]
        # Expect exactly one comparison flag for the repeated expression
        self.assertEqual(len(cmp_flags), len(set(cmp_flags)))
        # Ensure both equalities tie their respective boolean vars to SAME cmp_flag (no second flag created)
        # We check A_eq rows tying b1 or b2 to any cmp_flag; each should reference the same cmp_flag index.
        flag_idx = gen.var_indices[cmp_flags[0]] if cmp_flags else None
        b1_idx = gen.var_indices["b1"]
        b2_idx = gen.var_indices["b2"]
        ties = []
        for row in gen.A_eq:
            if abs(row[b1_idx]) == 1 and flag_idx is not None and abs(row[flag_idx]) == 1:
                ties.append(("b1", row[flag_idx]))
            if abs(row[b2_idx]) == 1 and flag_idx is not None and abs(row[flag_idx]) == 1:
                ties.append(("b2", row[flag_idx]))
        self.assertEqual(
            len([t for t in ties if t[0] == "b1"]),
            1,
            f"Expected one tie equality for b1; ties={ties}",
        )
        self.assertEqual(
            len([t for t in ties if t[0] == "b2"]),
            1,
            f"Expected one tie equality for b2; ties={ties}",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
