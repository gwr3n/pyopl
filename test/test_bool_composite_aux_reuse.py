import os
import unittest

from pyopl.pyopl_core import OPLLexer, OPLParser
from pyopl.scipy_codegen_csc import SciPyCSCCodeGenerator


class TestBoolCompositeAuxReuse(unittest.TestCase):
    def _dump_state(self, label, gen):
        if not os.environ.get("PYOPL_DEBUG_REUSE") and "FAIL" not in label:
            return
        print(f"\n[DEBUG] {label} var_names: {gen.var_names}")
        print(f"[DEBUG] {label} var_indices: {gen.var_indices}")
        print(f"[DEBUG] {label} _bool_subtree_cache: {getattr(gen, '_bool_subtree_cache', None)}")
        print(f"[DEBUG] {label} A_eq ({len(gen.A_eq)} rows):")
        for i, row in enumerate(gen.A_eq):
            print(f"  [DEBUG] A_eq[{i}]: {row}")
        print(f"[DEBUG] {label} b_eq: {getattr(gen, 'b_eq', None)}")
        print(f"[DEBUG] {label} A_ub ({len(gen.A_ub)} rows):")
        for i, row in enumerate(gen.A_ub):
            print(f"  [DEBUG] A_ub[{i}]: {row}")
        print(f"[DEBUG] {label} b_ub: {getattr(gen, 'b_ub', None)}")
        constraints = getattr(gen, "_debug_ast", {}).get("constraints", [])
        for idx, c in enumerate(constraints):
            print(f"  [DEBUG] AST constraint[{idx}]: {c}")

    def gen(self, src: str):
        lexer = OPLLexer()
        parser = OPLParser()
        ast = parser.parse(lexer.tokenize(src))
        gen = SciPyCSCCodeGenerator(ast)
        gen._build_variables()
        gen._build_objective()
        gen._build_constraints()
        gen._debug_ast = ast  # attach for debug dumps
        return gen

    def test_and_composite_reuse(self):
        opl = """
        dvar boolean b1; dvar boolean b2; dvar boolean x; dvar boolean y;
        minimize 0;
        subject to {
            b1 == ((x == 1) && (y == 0));
            b2 == ((x == 1) && (y == 0));
        }
        """
        gen = self.gen(opl)

        self._dump_state("AND_REUSE", gen)

        # Collect auxiliary boolean variables introduced (_baux*)
        baux_vars = [v for v in gen.var_indices if v.startswith("_baux")]
        # Expect exactly two auxiliaries: one for (y==0) negation and one for the AND node.
        self.assertLessEqual(len(baux_vars), 2, f"Unexpected extra auxiliaries created: {baux_vars}")
        self.assertEqual(
            len(set(baux_vars)),
            len(baux_vars),
            "Duplicate auxiliary names detected (should be impossible)",
        )
        # Ensure that both equalities tie to the same AND composite auxiliary.
        # Identify AND composite aux: it participates in two equality rows with b1 and b2 (coefficient pattern b - z == 0)
        b1_idx = gen.var_indices["b1"]
        b2_idx = gen.var_indices["b2"]
        candidate = None
        for row in gen.A_eq:
            if abs(row[b1_idx]) == 1:
                # find aux with coefficient -1
                for v in baux_vars:
                    vidx = gen.var_indices[v]
                    if abs(row[vidx]) == 1:
                        candidate = v
                        break
        if candidate is None:
            self._dump_state("AND_REUSE_FAIL", gen)
        self.assertIsNotNone(candidate, "Failed to locate AND composite auxiliary tied to b1")
        z_idx = gen.var_indices[candidate]
        # Count ties for b1 and b2
        ties_b1 = sum(1 for row in gen.A_eq if abs(row[b1_idx]) == 1 and abs(row[z_idx]) == 1)
        ties_b2 = sum(1 for row in gen.A_eq if abs(row[b2_idx]) == 1 and abs(row[z_idx]) == 1)
        self.assertEqual(ties_b1, 1, f"Expected one equality tying b1 to composite aux {candidate}")
        self.assertEqual(ties_b2, 1, f"Expected one equality tying b2 to composite aux {candidate}")

    def test_or_composite_reuse(self):
        opl = """
        dvar boolean b1; dvar boolean b2; dvar boolean x; dvar boolean y;
        minimize 0;
        subject to {
            b1 == ((x == 1) || (y == 0));
            b2 == ((x == 1) || (y == 0));
        }
        """
        gen = self.gen(opl)

        self._dump_state("OR_REUSE", gen)

        baux_vars = [v for v in gen.var_indices if v.startswith("_baux")]
        # Expect at most two auxiliaries: one for (y==0) negation and one for OR node.
        self.assertLessEqual(len(baux_vars), 2, f"Unexpected extra auxiliaries created: {baux_vars}")
        # Reuse: ensure only one OR composite aux by checking equality ties
        b1_idx = gen.var_indices["b1"]
        b2_idx = gen.var_indices["b2"]
        # locate candidate OR aux (appears in two equality rows with b1/b2)
        candidate = None
        for v in baux_vars:
            vidx = gen.var_indices[v]
            ties = sum(1 for row in gen.A_eq if abs(row[vidx]) == 1 and (abs(row[b1_idx]) == 1 or abs(row[b2_idx]) == 1))
            if ties >= 2:  # appears with both b1 and b2
                candidate = v
                break
        if candidate is None:
            self._dump_state("OR_REUSE_FAIL", gen)
        self.assertIsNotNone(
            candidate,
            "Failed to locate OR composite auxiliary reused for both equalities",
        )

    def test_nested_parentheses_reuse(self):
        """Regression: excessive parentheses around identical boolean expressions must not break reuse."""
        opl = """
        dvar boolean b1; dvar boolean b2; dvar boolean b3; dvar boolean b4; dvar boolean x; dvar boolean y;
        minimize 0;
        subject to {
            b1 == (((((x == 1))))) && ((((y == 0))));
            b2 == ((x == 1) && (y == 0));
            b3 == (((x == 1)) || ((((y == 0)))));
            b4 == ((x == 1) || (y == 0));
        }
        """
        gen = self.gen(opl)

        self._dump_state("NESTED_REUSE", gen)

        # helper to find shared aux for a pair of booleans
        def shared_aux(b_left, b_right):
            l_idx = gen.var_indices[b_left]
            r_idx = gen.var_indices[b_right]
            baux_vars = [v for v in gen.var_indices if v.startswith("_baux")]
            for v in baux_vars:
                vidx = gen.var_indices[v]
                ties_left = any(abs(row[l_idx]) == 1 and abs(row[vidx]) == 1 for row in gen.A_eq)
                ties_right = any(abs(row[r_idx]) == 1 and abs(row[vidx]) == 1 for row in gen.A_eq)
                if ties_left and ties_right:
                    return v
            return None

        and_aux = shared_aux("b1", "b2")
        if and_aux is None:
            self._dump_state("NESTED_AND_REUSE_FAIL", gen)
        self.assertIsNotNone(
            and_aux,
            "Nested parentheses AND reuse failed (expected shared auxiliary for b1,b2)",
        )
        or_aux = shared_aux("b3", "b4")
        if or_aux is None:
            self._dump_state("NESTED_OR_REUSE_FAIL", gen)
        self.assertIsNotNone(
            or_aux,
            "Nested parentheses OR reuse failed (expected shared auxiliary for b3,b4)",
        )

    def test_mixed_nesting_reuse(self):
        """Regression: duplicated AND inside an OR ( (A&&B) || (A&&B) ) should not create extra AND auxiliaries; OR aux reused."""
        opl = """
        dvar boolean b1; dvar boolean b2; dvar boolean b3; dvar boolean x; dvar boolean y;
        minimize 0;
        subject to {
            b1 == (((x == 1) && (y == 0)) || ((x == 1) && (y == 0)));
            b2 == ((x == 1) && (y == 0));
            b3 == ((((x == 1) && (y == 0))) || ((((x == 1) && (y == 0)))));
        }
        """
        gen = self.gen(opl)

        self._dump_state("MIXED_REUSE", gen)

        baux_vars = [v for v in gen.var_indices if v.startswith("_baux")]
        # Expected auxiliaries: at most one for (y==0) negation, one for AND, one for OR => <=3 total
        self.assertLessEqual(len(baux_vars), 3, f"Unexpected extra auxiliaries created: {baux_vars}")
        self.assertEqual(len(set(baux_vars)), len(baux_vars), "Duplicate auxiliary names detected")
        b1_idx = gen.var_indices["b1"]
        b3_idx = gen.var_indices["b3"]
        # Find OR aux reused between b1 and b3 equality rows
        or_candidate = None
        for v in baux_vars:
            vidx = gen.var_indices[v]
            ties = [row for row in gen.A_eq if abs(row[vidx]) == 1 and (abs(row[b1_idx]) == 1 or abs(row[b3_idx]) == 1)]
            # need ties to both b1 and b3 (two distinct rows likely)
            b1_tie = any(abs(row[b1_idx]) == 1 for row in ties)
            b3_tie = any(abs(row[b3_idx]) == 1 for row in ties)
            if b1_tie and b3_tie:
                or_candidate = v
                break
        if or_candidate is None:
            self._dump_state("MIXED_OR_REUSE_FAIL", gen)
        self.assertIsNotNone(
            or_candidate,
            "Failed to find shared OR auxiliary for duplicated AND disjunction",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
