import unittest

from pyopl.scipy_codegen_csc import SciPyCSCCodeGenerator


class TestImplicationReified(unittest.TestCase):
    def _decl_bool(self, name):
        return {"type": "dvar", "name": name, "var_type": "boolean"}

    def _atom(self, var, val):
        return {
            "type": "constraint",
            "left": {"type": "name", "value": var},
            "op": "==",
            "right": {"type": "number", "value": val},
        }

    def test_implication_var_to_var1(self):
        # a==1 => b==1  encoded as implication_constraint
        ast = {
            "declarations": [self._decl_bool("a"), self._decl_bool("b")],
            "constraints": [
                {
                    "type": "implication_constraint",
                    "antecedent": self._atom("a", 1),
                    "consequent": self._atom("b", 1),
                }
            ],
            "objective": {
                "type": "minimize",
                "expression": {"type": "number", "value": 0},
            },
        }
        gen = SciPyCSCCodeGenerator(ast)
        gen._build_variables()
        gen._build_objective()
        gen._build_constraints()
        # Expect inequality: -b + a <= 0  (row coefficients a:1, b:-1)
        found = False
        for row, rhs in zip(gen.A_ub, gen.b_ub):
            coeffs = {gen.var_names[i]: row[i] for i in range(len(row)) if abs(row[i]) > 1e-12}
            if "a" in coeffs and "b" in coeffs and coeffs["a"] == 1 and coeffs["b"] == -1 and abs(rhs) < 1e-12:
                found = True
        self.assertTrue(
            found,
            f"Expected implication inequality a->b; A_ub={gen.A_ub}, b_ub={gen.b_ub}",
        )

    def test_reified_sum_ge_k(self):
        # y == (a + b + c >= 2)
        ast = {
            "declarations": [
                self._decl_bool("a"),
                self._decl_bool("b"),
                self._decl_bool("c"),
                self._decl_bool("y"),
            ],
            "constraints": [
                {
                    "type": "constraint",
                    "left": {"type": "name", "value": "y"},
                    "op": "==",
                    "right": {
                        "type": "constraint",
                        "op": ">=",
                        "left": {
                            "type": "binop",
                            "op": "+",
                            "left": {
                                "type": "binop",
                                "op": "+",
                                "left": {"type": "name", "value": "a"},
                                "right": {"type": "name", "value": "b"},
                            },
                            "right": {"type": "name", "value": "c"},
                        },
                        "right": {"type": "number", "value": 2},
                    },
                }
            ],
            "objective": {
                "type": "minimize",
                "expression": {"type": "number", "value": 0},
            },
        }
        gen = SciPyCSCCodeGenerator(ast)
        gen._build_variables()
        gen._build_objective()
        gen._build_constraints()
        # Expect two inequalities implementing reification
        self.assertGreaterEqual(
            len(gen.A_ub),
            2,
            f"Expected >=2 inequalities for reification; A_ub={gen.A_ub}",
        )

    def test_reified_trivial_cases(self):
        # y == (a + b >= 0)  -> y == 1
        ast1 = {
            "declarations": [
                self._decl_bool("a"),
                self._decl_bool("b"),
                self._decl_bool("y"),
            ],
            "constraints": [
                {
                    "type": "constraint",
                    "left": {"type": "name", "value": "y"},
                    "op": "==",
                    "right": {
                        "type": "constraint",
                        "op": ">=",
                        "left": {
                            "type": "binop",
                            "op": "+",
                            "left": {"type": "name", "value": "a"},
                            "right": {"type": "name", "value": "b"},
                        },
                        "right": {"type": "number", "value": 0},
                    },
                }
            ],
            "objective": {
                "type": "minimize",
                "expression": {"type": "number", "value": 0},
            },
        }
        gen1 = SciPyCSCCodeGenerator(ast1)
        gen1._build_variables()
        gen1._build_objective()
        gen1._build_constraints()
        # Should have equality y == 1
        y_idx = gen1.var_indices["y"]
        found_y1 = any(abs(row[y_idx] - 1.0) < 1e-12 and abs(rhs - 1.0) < 1e-12 for row, rhs in zip(gen1.A_eq, gen1.b_eq))
        self.assertTrue(
            found_y1,
            f"Expected y==1 equality for trivial k<=0; A_eq={gen1.A_eq}, b_eq={gen1.b_eq}",
        )
        # y == (a + b >= 3) -> y == 0 (impossible threshold)
        ast2 = {
            "declarations": [
                self._decl_bool("a"),
                self._decl_bool("b"),
                self._decl_bool("y"),
            ],
            "constraints": [
                {
                    "type": "constraint",
                    "left": {"type": "name", "value": "y"},
                    "op": "==",
                    "right": {
                        "type": "constraint",
                        "op": ">=",
                        "left": {
                            "type": "binop",
                            "op": "+",
                            "left": {"type": "name", "value": "a"},
                            "right": {"type": "name", "value": "b"},
                        },
                        "right": {"type": "number", "value": 3},
                    },
                }
            ],
            "objective": {
                "type": "minimize",
                "expression": {"type": "number", "value": 0},
            },
        }
        gen2 = SciPyCSCCodeGenerator(ast2)
        gen2._build_variables()
        gen2._build_objective()
        gen2._build_constraints()
        y_idx2 = gen2.var_indices["y"]
        found_y0 = any(abs(row[y_idx2] - 1.0) < 1e-12 and abs(rhs - 0.0) < 1e-12 for row, rhs in zip(gen2.A_eq, gen2.b_eq))
        self.assertTrue(
            found_y0,
            f"Expected y==0 equality for impossible k>|S|; A_eq={gen2.A_eq}, b_eq={gen2.b_eq}",
        )


if __name__ == "__main__":
    unittest.main()
