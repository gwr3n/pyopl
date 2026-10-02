import json
import unittest

from pyopl.equivalence import (
    IndexedAffineExpression,
    IndexedAffineFamily,
    IndexedAffineTerm,
    IndexedObservablePair,
    ParameterContract,
    ParameterFact,
    SymbolicAffineExpression,
    SymbolicDomain,
    SymbolicPredicate,
)
from pyopl.equivalence._indexed_projection import indexed_observable_pair_from_record, indexed_observable_pair_to_record


class IndexedProjectionFoundationTests(unittest.TestCase):

    def test_contract_and_observable_family_are_typed_data(self):
        period = SymbolicDomain("t", SymbolicAffineExpression.constant_value(1), SymbolicAffineExpression.parameter("N"))
        family = IndexedAffineFamily(
            ("t",),
            (period,),
            IndexedAffineExpression((IndexedAffineTerm("order", ("t",)),)),
        )
        pair = IndexedObservablePair("order[{t}]", family, family)
        contract = ParameterContract(
            (
                ParameterFact(
                    SymbolicPredicate(
                        SymbolicAffineExpression.parameter("N"), ">=", SymbolicAffineExpression.constant_value(1)
                    ),
                    "declared",
                ),
            ),
        )

        self.assertEqual(pair.left.expression.terms[0].declaration, "order")
        self.assertEqual(contract.render(), ("declared: 1*N >= 1",))

    def test_indexed_family_alpha_normalization_ignores_binder_names(self):
        left = IndexedAffineFamily(
            ("period",),
            (SymbolicDomain("period", SymbolicAffineExpression.constant_value(1), SymbolicAffineExpression.parameter("N")),),
            IndexedAffineExpression((IndexedAffineTerm("x", ("period + 1",)),)),
        )
        right = IndexedAffineFamily(
            ("t",),
            (SymbolicDomain("t", SymbolicAffineExpression.constant_value(1), SymbolicAffineExpression.parameter("N")),),
            IndexedAffineExpression((IndexedAffineTerm("x", ("t + 1",)),)),
        )

        self.assertEqual(left.alpha_normalized(), right.alpha_normalized())

    def test_indexed_family_guard_is_alpha_normalized_and_serialized(self):
        left = IndexedAffineFamily(
            ("i", "j"),
            (
                SymbolicDomain("i", SymbolicAffineExpression.constant_value(1), SymbolicAffineExpression.parameter("N")),
                SymbolicDomain("j", SymbolicAffineExpression.constant_value(1), SymbolicAffineExpression.parameter("N")),
            ),
            IndexedAffineExpression((IndexedAffineTerm("x", ("i", "j")),)),
            SymbolicPredicate(
                SymbolicAffineExpression.parameter("i"),
                "!=",
                SymbolicAffineExpression.parameter("j"),
            ),
        )
        right = IndexedAffineFamily(
            ("tail", "head"),
            (
                SymbolicDomain("tail", SymbolicAffineExpression.constant_value(1), SymbolicAffineExpression.parameter("N")),
                SymbolicDomain("head", SymbolicAffineExpression.constant_value(1), SymbolicAffineExpression.parameter("N")),
            ),
            IndexedAffineExpression((IndexedAffineTerm("x", ("tail", "head")),)),
            SymbolicPredicate(
                SymbolicAffineExpression.parameter("tail"),
                "!=",
                SymbolicAffineExpression.parameter("head"),
            ),
        )
        pair = IndexedObservablePair("x[{i}][{j}]", left, right)
        self.assertEqual(left.alpha_normalized(), right.alpha_normalized())
        record = indexed_observable_pair_to_record(pair)
        restored = indexed_observable_pair_from_record(json.loads(json.dumps(record)))
        self.assertEqual(restored, pair)

    def test_indexed_family_preserves_and_normalizes_nested_sum_binders(self):
        left = IndexedAffineFamily(
            ("t",),
            (SymbolicDomain("t", SymbolicAffineExpression.constant_value(1), SymbolicAffineExpression.parameter("N")),),
            IndexedAffineExpression(
                (IndexedAffineTerm("x", ("t", "j")),),
                sum_binders=("j",),
                sum_domains=(
                    SymbolicDomain(
                        "j",
                        SymbolicAffineExpression((("t", 1),), 1),
                        SymbolicAffineExpression((("N", 1),), 1),
                    ),
                ),
            ),
        )
        right = IndexedAffineFamily(
            ("period",),
            (SymbolicDomain("period", SymbolicAffineExpression.constant_value(1), SymbolicAffineExpression.parameter("N")),),
            IndexedAffineExpression(
                (IndexedAffineTerm("x", ("period", "k")),),
                sum_binders=("k",),
                sum_domains=(
                    SymbolicDomain(
                        "k",
                        SymbolicAffineExpression((("period", 1),), 1),
                        SymbolicAffineExpression((("N", 1),), 1),
                    ),
                ),
            ),
        )

        normalized = left.alpha_normalized()
        self.assertEqual(normalized, right.alpha_normalized())
        self.assertEqual(normalized.expression.sum_binders, ("s0",))

    def test_non_affine_indexed_terms_are_rejected_by_shape(self):
        with self.assertRaises(ValueError):
            IndexedAffineTerm("x", ("i * j",), 1)

    def test_indexed_family_rejects_unbound_decision_indices(self):
        with self.assertRaisesRegex(ValueError, "unbound index"):
            IndexedAffineFamily(
                ("i",),
                (SymbolicDomain("i", SymbolicAffineExpression.constant_value(1), SymbolicAffineExpression.constant_value(2)),),
                IndexedAffineExpression((IndexedAffineTerm("x", ("j",)),)),
            )

    def test_indexed_domains_must_name_their_binders(self):
        with self.assertRaisesRegex(ValueError, "corresponding binders"):
            IndexedAffineExpression(
                (IndexedAffineTerm("x", ("i",)),),
                sum_binders=("i",),
                sum_domains=(
                    SymbolicDomain(
                        "j", SymbolicAffineExpression.constant_value(1), SymbolicAffineExpression.constant_value(2)
                    ),
                ),
            )
