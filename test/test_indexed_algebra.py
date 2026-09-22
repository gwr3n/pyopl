import unittest

import sympy as sp

from pyopl._indexed_algebra import (
    DecisionAtom,
    DomainTerm,
    IndexTerm,
    QuantifiedExpression,
    domain_dependency_graph,
    free_binders,
    render_indexed_ir,
)


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

    def test_free_binders_traverses_nested_index_tree(self):
        index = IndexTerm("arithmetic", ("+", IndexTerm("binder", 1), IndexTerm("number", "1")))
        atom = DecisionAtom("x", (IndexTerm("binder", 0), index))

        self.assertEqual(free_binders(atom), frozenset({0, 1}))

    def test_render_parameter_selected_index(self):
        selected = IndexTerm("application", ("next", (IndexTerm("binder", 0),)))

        self.assertEqual(render_indexed_ir(selected), "next[$0]")

    def test_domain_dependency_graph_tracks_prior_binders(self):
        domains = (
            DomainTerm("named", "I"),
            DomainTerm("range", (IndexTerm("number", "1"), IndexTerm("binder", 0))),
        )

        self.assertEqual(domain_dependency_graph(domains, first_binder_id=0), {0: frozenset(), 1: frozenset({0})})

    def test_quantifier_free_binders_excludes_local_scope(self):
        outer = IndexTerm("binder", 4)
        local = IndexTerm("binder", 5)
        expression = QuantifiedExpression(
            (DomainTerm("range", (IndexTerm("number", "1"), outer)),),
            ("compare", ">=", ("binder", 5), ("number", "1")),
            (sp.S.Zero, ((DecisionAtom("x", (local, outer)), sp.S.One),)),
        )

        self.assertEqual(free_binders(expression, first_binder_id=5), frozenset({4}))


if __name__ == "__main__":
    unittest.main()
