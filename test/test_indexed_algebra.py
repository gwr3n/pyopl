import unittest

import sympy as sp

from pyopl._indexed_algebra import DecisionAtom, IndexTerm, QuantifiedExpression, render_indexed_ir


class IndexedAlgebraRenderingTests(unittest.TestCase):
    def test_render_nested_quantifier_preserves_binder_identities(self):
        atom = DecisionAtom("x", (IndexTerm("binder", 0), IndexTerm("binder", 1)))
        expression = QuantifiedExpression(
            ("I",),
            True,
            QuantifiedExpression(("I",), True, (sp.S.Zero, ((atom, sp.S.One),))),
        )

        self.assertEqual(render_indexed_ir(expression), "sum($0 in I) sum($1 in I) x[$0, $1]")

    def test_render_arithmetic_index_preserves_complete_tree(self):
        index = IndexTerm("arithmetic", ("+", IndexTerm("binder", 1), IndexTerm("number", "1")))
        atom = DecisionAtom("x", (IndexTerm("binder", 0), index))

        self.assertEqual(render_indexed_ir(atom), "x[$0, ($1 + 1)]")


if __name__ == "__main__":
    unittest.main()
