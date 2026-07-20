import unittest

from pyopl.pyopl_core import OPLLexer, OPLParser
from pyopl.scipy_codegen_csc import SciPyCSCCodeGenerator
from pyopl.semantic_error import SemanticError


class TestNotEqualRewriteSciPy(unittest.TestCase):
    def gen(self, src: str):
        lexer = OPLLexer()
        parser = OPLParser()
        ast = parser.parse(lexer.tokenize(src))
        gen = SciPyCSCCodeGenerator(ast)
        gen._build_variables()
        gen._build_objective()
        gen._build_constraints()
        return gen

    def test_boolean_neq_rewritten_to_xor(self):
        opl = """
        dvar boolean a; dvar boolean b; minimize 0; subject to { a != b; }
        """
        gen = self.gen(opl)
        # Expect an equality row enforcing a + b == 1
        # Build expected row coefficients: a and b each appear once in some equality row with RHS=1
        a_idx = gen.var_indices["a"]
        b_idx = gen.var_indices["b"]
        found = False
        for row, rhs in zip(gen.A_eq, gen.b_eq):
            if abs(rhs - 1.0) < 1e-9 and abs(row[a_idx] - 1.0) < 1e-9 and abs(row[b_idx] - 1.0) < 1e-9:
                found = True
                break
        self.assertTrue(found, f"Did not find XOR row a + b == 1; A_eq={gen.A_eq}, b_eq={gen.b_eq}")

    def test_integer_neq_uses_bigM_delta(self):
        opl = """
        dvar int x; dvar int y; minimize 0; subject to {
            x >= -10; x <= 10; y >= -10; y <= 10; x != y;
        }
        """
        gen = self.gen(opl)
        # Identify newly introduced binary variable (not x or y)
        aux_vars = [v for v in gen.var_indices if v not in ("x", "y")]
        self.assertEqual(
            len(aux_vars),
            1,
            f"Expected exactly one auxiliary binary var, found {aux_vars}",
        )
        delta = aux_vars[0]
        delta_idx = gen.var_indices[delta]
        x_idx = gen.var_indices["x"]
        y_idx = gen.var_indices["y"]
        # Determine M heuristically: look at absolute coefficient on delta (largest)
        M_candidates = [abs(row[delta_idx]) for row in gen.A_ub]
        M = max(M_candidates) if M_candidates else 1_000_000.0
        found_forms = 0
        for row, rhs in zip(gen.A_ub, gen.b_ub):
            # Form 1: x - y - M*delta <= -1
            if (
                abs(row[x_idx] - 1.0) < 1e-9
                and abs(row[y_idx] + 1.0) < 1e-9
                and abs(row[delta_idx] + M) < 1e-6
                and abs(rhs + 1.0) < 1e-6
            ):
                found_forms += 1
            # Form 2: -x + y + M*delta <= M - 1 (generic unified encoding variant)
            if (
                abs(row[x_idx] + 1.0) < 1e-9
                and abs(row[y_idx] - 1.0) < 1e-9
                and abs(row[delta_idx] - M) < 1e-6
                and abs(rhs - (M - 1.0)) < 1e-3
            ):
                found_forms += 1
            # Alternative legacy: x - y + M*delta <= M - 1
            if (
                abs(row[x_idx] - 1.0) < 1e-9
                and abs(row[y_idx] + 1.0) < 1e-9
                and abs(row[delta_idx] - M) < 1e-6
                and abs(rhs - (M - 1.0)) < 1e-3
            ):
                found_forms += 1
        self.assertGreaterEqual(
            found_forms,
            2,
            f"Did not detect two distinct big-M inequalities for x != y; rows={gen.A_ub}, b_ub={gen.b_ub}, M={M}, delta={delta}",
        )
        self.assertTrue(
            self._rows_allow_assignment(gen, {"x": 1, "y": 2}),
            "x < y must be feasible for x != y",
        )
        self.assertTrue(
            self._rows_allow_assignment(gen, {"x": 2, "y": 1}),
            "x > y must be feasible for x != y",
        )
        self.assertFalse(
            self._rows_allow_assignment(gen, {"x": 1, "y": 1}),
            "x == y must be infeasible for x != y",
        )

    def test_float_neq_is_rejected_without_an_explicit_tolerance_policy(self):
        opl = """
        dvar float x; dvar float y; minimize 0; subject to {
            x >= 0; x <= 1; y >= 0; y <= 1; x != y;
        }
        """

        with self.assertRaisesRegex(SemanticError, "integer|tolerance|not-equal"):
            self.gen(opl)

    def test_unbounded_integer_neq_is_rejected_without_finite_big_m_bounds(self):
        opl = """
        dvar int x; dvar int y; minimize 0; subject to { x != y; }
        """

        with self.assertRaisesRegex(SemanticError, "finite.*bounds|big-M"):
            self.gen(opl)

    def test_expand_and_treats_not_equal_as_a_disjunction(self):
        opl = """
        dvar int x; dvar int y; minimize 0; subject to {
            x >= -10; x <= 10; y >= -10; y <= 10;
        }
        """
        gen = self.gen(opl)
        comparison = {
            "type": "constraint",
            "op": "!=",
            "left": {"type": "name", "value": "x", "sem_type": "int"},
            "right": {"type": "name", "value": "y", "sem_type": "int"},
        }

        gen._expand_and([comparison])

        self.assertTrue(
            self._rows_allow_assignment(gen, {"x": 1, "y": 2}),
            "The helper must preserve the x < y branch of x != y",
        )
        self.assertTrue(
            self._rows_allow_assignment(gen, {"x": 2, "y": 1}),
            "The helper must preserve the x > y branch of x != y",
        )
        self.assertFalse(
            self._rows_allow_assignment(gen, {"x": 1, "y": 1}),
            "The helper must reject the equality branch of x != y",
        )

    def _rows_allow_assignment(self, gen, values):
        auxiliary_names = [name for name in gen.var_indices if name not in values]
        for mask in range(1 << len(auxiliary_names)):
            assignment = dict(values)
            for bit, name in enumerate(auxiliary_names):
                assignment[name] = (mask >> bit) & 1
            if all(
                sum(row[gen.var_indices[name]] * value for name, value in assignment.items()) <= rhs + 1e-9
                for row, rhs in zip(gen.A_ub, gen.b_ub)
            ) and all(
                abs(sum(row[gen.var_indices[name]] * value for name, value in assignment.items()) - rhs) <= 1e-9
                for row, rhs in zip(gen.A_eq, gen.b_eq)
            ):
                return True
        return False

    def test_strict_constraints_are_normalized(self):
        opl = """
        dvar float x; minimize 0; subject to { x > 1; x < 5; }
        """
        gen = self.gen(opl)
        x_idx = gen.var_indices["x"]
        found_gt = any(abs(row[x_idx] + 1.0) < 1e-9 and abs(rhs + 1.000001) < 1e-9 for row, rhs in zip(gen.A_ub, gen.b_ub))
        found_lt = any(abs(row[x_idx] - 1.0) < 1e-9 and abs(rhs - 4.999999) < 1e-9 for row, rhs in zip(gen.A_ub, gen.b_ub))
        self.assertTrue(found_gt, f"Did not find x >= 1 + BOOL_EPS row; A_ub={gen.A_ub}, b_ub={gen.b_ub}")
        self.assertTrue(found_lt, f"Did not find x <= 5 - BOOL_EPS row; A_ub={gen.A_ub}, b_ub={gen.b_ub}")

    def test_strict_implication_consequent_is_normalized(self):
        opl = """
        dvar boolean b; dvar float x; minimize 0; subject to { b == 1 => x < 5; }
        """
        gen = self.gen(opl)
        x_idx = gen.var_indices["x"]
        flag_idx = gen.var_indices["b"]
        found = any(
            abs(row[x_idx] - 1.0) < 1e-9 and row[flag_idx] > 0 and abs(rhs - (row[flag_idx] + 4.999999)) < 1e-6
            for row, rhs in zip(gen.A_ub, gen.b_ub)
        )
        self.assertTrue(found, f"Did not find strict consequent gated row; A_ub={gen.A_ub}, b_ub={gen.b_ub}")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
