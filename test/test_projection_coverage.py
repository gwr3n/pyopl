import json
import unittest
from fractions import Fraction

from pyopl.equivalence import (
    AffineObservable,
    IndexDomain,
    IndexedAffineComponent,
    IndexedAffineExpression,
    IndexedAffineFamily,
    IndexedAffineTerm,
    IndexedLinearPredicate,
    IndexedObservablePair,
    IndexedParameterFact,
    IndexedScopedAffineExpression,
    ObservablePair,
    ObservableTemplate,
    ParameterContract,
    ParameterFact,
    SymbolicAffineExpression,
    SymbolicDomain,
    SymbolicPredicate,
    ground_indexed_observable_mapping,
    indexed_observable_pair_from_record,
    parameter_contract_from_record,
    parameter_contract_to_record,
    parse_indexed_observable_mapping,
    parse_observable_mapping,
    serialize_indexed_observable_mapping,
    serialize_observable_mapping,
)
from pyopl.equivalence._projected_value import expand_observable_templates, legacy_observables, render_observable_mapping


def source_pair(expression, domain="1..N"):
    family = {"expression": expression, "domains": [{"name": "i", "range": domain}]}
    return {"kind": "indexed", "name": "value[{i}]", "left": family, "right": family}


class ProjectedValueTests(unittest.TestCase):
    def test_affine_observables_canonicalize_and_render(self):
        expression = AffineObservable((("z", 2), ("a", 3), ("z", -2), ("tiny", 1e-13)), -0.0)
        self.assertEqual(expression.coefficients, (("a", 3.0),))
        pairs = (ObservablePair("value", expression, AffineObservable((), 0)),)
        self.assertEqual(render_observable_mapping(pairs), ("value: 3*a <-> 0",))
        self.assertEqual(parse_observable_mapping(serialize_observable_mapping(pairs)), pairs)
        self.assertEqual(tuple(pair.name for pair in legacy_observables({"z": "b", "a": "x"})), ("a", "z"))

    def test_affine_observable_rejects_invalid_values(self):
        cases = [
            ((("", 1),), 0, "column names"),
            ((("x", "bad"),), 0, "coefficients"),
            ((("x", float("inf")),), 0, "coefficients"),
            ((), "bad", "constants"),
            ((), float("nan"), "constants"),
        ]
        for coefficients, constant, message in cases:
            with self.subTest(message=message, coefficients=coefficients, constant=constant):
                with self.assertRaisesRegex(ValueError, message):
                    AffineObservable(coefficients, constant)
        with self.assertRaisesRegex(ValueError, "names must be nonempty"):
            ObservablePair("", AffineObservable(()), AffineObservable(()))

    def test_finite_templates_expand_in_order_and_filter(self):
        def make(indices):
            return AffineObservable(((f"x_{indices['i']}", 1),))

        template = ObservableTemplate("x[{i}]", (IndexDomain("i", (2, 1, 3)),), make, make, lambda e: e["i"] != 1)
        constant = ObservableTemplate("constant", (), lambda _: AffineObservable((), 2), lambda _: AffineObservable(()))
        self.assertEqual(
            tuple(pair.name for pair in expand_observable_templates((template, constant))), ("x[2]", "x[3]", "constant")
        )
        for name, values in (("", (1,)), ("i", (1, 1))):
            with self.subTest(name=name, values=values), self.assertRaisesRegex(ValueError, "unique"):
                IndexDomain(name, values)

    def test_grounded_mapping_parser_rejects_malformed_records(self):
        invalid = [
            ("{", "invalid observable mapping JSON"),
            ("{}", "must contain a list"),
            (json.dumps([1]), "requires a string name"),
            (json.dumps([{"name": "x"}, {"name": "x"}]), "requires left and right"),
            (json.dumps([{"name": "x", "left": {}, "right": {}}, {"name": "x"}]), "names must be unique"),
            (json.dumps([{"name": "x", "left": [], "right": {}}]), "expressions must be JSON objects"),
            (json.dumps([{"name": "x", "left": {"coefficients": []}, "right": {}}]), "coefficients must be"),
            (json.dumps([{"name": "x", "left": {"coefficients": {}, "constant": "1"}, "right": {}}]), "constants must be"),
        ]
        for text, message in invalid:
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, message):
                parse_observable_mapping(text)
        pair = ObservablePair("x", AffineObservable(()), AffineObservable(()))
        with self.assertRaisesRegex(ValueError, "names must be unique"):
            serialize_observable_mapping((pair, pair))
        with self.assertRaisesRegex(ValueError, "unsupported entry type"):
            serialize_observable_mapping((object(),))

    def test_uniform_envelope_validates_schema_and_unique_names(self):
        pair = parse_observable_mapping(json.dumps([source_pair("x[i]")]))[0]
        encoded = serialize_indexed_observable_mapping((pair,), ParameterContract())
        payload = json.loads(encoded)
        self.assertEqual(parse_indexed_observable_mapping(encoded), ((pair,), ParameterContract()))
        invalid = [
            ("{", "invalid indexed observable mapping JSON"),
            ("[]", "must contain an object"),
            (json.dumps({**payload, "schema": "other"}), "unsupported uniform indexed mapping schema"),
            (json.dumps({**payload, "version": 2}), "schema version"),
            (json.dumps({**payload, "observables": []}), "nonempty observables"),
            (json.dumps({**payload, "observables": payload["observables"] * 2}), "names must be unique"),
            (json.dumps({**payload, "parameter_contract": []}), "parameter_contract object"),
        ]
        for text, message in invalid:
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, message):
                parse_indexed_observable_mapping(text)
        for mapping, message in (((), "at least one"), ((pair, pair), "unique"), ((object(),), "at least one")):
            with self.subTest(mapping=mapping), self.assertRaisesRegex(ValueError, message):
                serialize_indexed_observable_mapping(mapping, ParameterContract())

    def test_source_expression_scopes_and_constants_ground_correctly(self):
        expression = "2 * x[i+1] - x[i-1] / 2 + sum(j in 1..i) (x[j] + 3)"
        mapping = parse_observable_mapping(json.dumps([source_pair(expression, "1..2")]))
        model = "dvar boolean x[0..3]; minimize 0; subject to {}"
        grounded = ground_indexed_observable_mapping(mapping, model, model)
        self.assertEqual(
            grounded[0].left,
            AffineObservable((("x_2", 2), ("x_0", -0.5), ("x_1", 1)), 3),
        )
        self.assertEqual(
            grounded[1].left,
            AffineObservable((("x_3", 2), ("x_1", -0.5), ("x_1", 1), ("x_2", 1)), 6),
        )
        self.assertEqual(parse_observable_mapping(serialize_observable_mapping(mapping)), mapping)

    def test_grounding_rejects_mismatched_domains_guards_and_nonintegral_bounds(self):
        mapping = parse_observable_mapping(json.dumps([source_pair("x[i]")]))
        model = "int N = ...; dvar boolean x[1..N]; minimize 0; subject to {}"
        with self.assertRaisesRegex(ValueError, "mismatched grounded domains"):
            ground_indexed_observable_mapping(mapping, model, model, "N = 1;", "N = 2;")
        with self.assertRaisesRegex(ValueError, "indexed observable pairs"):
            ground_indexed_observable_mapping((object(),), model, model, "N = 1;", "N = 1;")
        one = SymbolicAffineExpression.constant_value(1)
        two = SymbolicAffineExpression.constant_value(2)
        domain = SymbolicDomain("i", one, two)
        expression = IndexedAffineExpression((IndexedAffineTerm("x", ("i",)),))
        left = IndexedAffineFamily(("i",), (domain,), expression)
        right = IndexedAffineFamily(("i",), (domain,), expression, SymbolicPredicate(one, "==", two))
        with self.assertRaisesRegex(ValueError, "mismatched grounded guards"):
            ground_indexed_observable_mapping(
                (IndexedObservablePair("x[{i}]", left, right),), model, model, "N = 2;", "N = 2;"
            )
        fraction_domain = SymbolicDomain("i", SymbolicAffineExpression.constant_value("1/2"), two)
        fraction_family = IndexedAffineFamily(("i",), (fraction_domain,), expression)
        with self.assertRaisesRegex(ValueError, "not integral"):
            ground_indexed_observable_mapping(
                (IndexedObservablePair("x[{i}]", fraction_family, fraction_family),), model, model, "N = 2;", "N = 2;"
            )

    def test_source_mapping_rejects_invalid_expressions_and_domains(self):
        cases = [
            ("x[i]", "1", "range separator"),
            ("x[i]", "1..N..2", "range separator"),
            ("x[i*2]", "1..N", "indices must be"),
            ("x[i] * x[i]", "1..N", "numeric coefficients"),
            ("sum(i in 1..N) x[i]", "1..N", "shadows"),
        ]
        for expression, domain, message in cases:
            with self.subTest(expression=expression, domain=domain), self.assertRaisesRegex(ValueError, message):
                parse_observable_mapping(json.dumps([source_pair(expression, domain)]))

    def test_grounding_handles_offsets_empty_sums_and_legacy_sum_constants(self):
        one = SymbolicAffineExpression.constant_value(1)
        two = SymbolicAffineExpression.constant_value(2)
        outer = SymbolicDomain("i", one, two)
        empty = SymbolicDomain("j", two, one)
        expression = IndexedAffineExpression(
            (IndexedAffineTerm("x", ("-i - 1",)),),
            7,
            ("j",),
            (empty,),
        )
        family = IndexedAffineFamily(("i",), (outer,), expression)
        model = "dvar boolean x[1..2]; minimize 0; subject to {}"
        result = ground_indexed_observable_mapping((IndexedObservablePair("x[{i}]", family, family),), model, model)
        self.assertEqual([pair.left for pair in result], [AffineObservable((), 7), AffineObservable((), 7)])
        expression = IndexedAffineExpression((IndexedAffineTerm("x", ("-i - 1",)),))
        family = IndexedAffineFamily(("i",), (outer,), expression)
        result = ground_indexed_observable_mapping((IndexedObservablePair("x[{i}]", family, family),), model, model)
        self.assertEqual(result[0].left, AffineObservable((("x_-2", 1),)))

    def test_grounding_rejects_unknown_symbolic_parameter(self):
        mapping = parse_observable_mapping(json.dumps([source_pair("x[i]", "1..M")]))
        model = "dvar boolean x[1..2]; minimize 0; subject to {}"
        with self.assertRaisesRegex(ValueError, "cannot evaluate symbolic index value 'M'"):
            ground_indexed_observable_mapping(mapping, model, model)


class IndexedProjectionCoverageTests(unittest.TestCase):
    def test_symbolic_arithmetic_substitution_is_exact(self):
        expression = SymbolicAffineExpression((("x", "1/2"), ("x", "1/2"), ("y", -2)), 3)
        self.assertEqual(expression.substitute({"x": SymbolicAffineExpression((("y", 2),), 1)}).render(), "4")
        self.assertEqual(expression.subtract(expression).render(), "0")
        self.assertEqual(expression.scale("1/2").constant, Fraction(3, 2))
        self.assertEqual(
            SymbolicDomain("i", expression, expression, True).render(), "i in 1*x + -2*y + 3..1*x + -2*y + 3 (nonempty)"
        )
        for value in ("not-a-number", "1/0"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite rational"):
                SymbolicAffineExpression.constant_value(value)
        with self.assertRaisesRegex(ValueError, "parameter names"):
            SymbolicAffineExpression((("", 1),))

    def test_indexed_expression_arithmetic_and_alpha_normalization(self):
        domain = SymbolicDomain("j", SymbolicAffineExpression.parameter("i"), SymbolicAffineExpression.constant_value(3))
        expression = IndexedAffineExpression((IndexedAffineTerm("x", ("i", "j + 1"), 2),), 1, ("j",), (domain,))
        self.assertEqual(expression.subtract(expression).terms, ())
        self.assertEqual(expression.scale("1/2").terms[0].coefficient, Fraction(1))
        normalized = expression.alpha_normalized()
        self.assertEqual(normalized.sum_binders, ("s0",))
        self.assertEqual(normalized.terms[0].indices, ("i", "s0 + 1"))
        with self.assertRaisesRegex(ValueError, "matching sum schemas"):
            expression.add(IndexedAffineExpression())

    def test_indexed_contract_roundtrip_and_alpha_normalization(self):
        one = SymbolicAffineExpression.constant_value(1)
        domain = SymbolicDomain("i", one, SymbolicAffineExpression.parameter("N"))
        inner = SymbolicDomain("j", SymbolicAffineExpression.parameter("i"), SymbolicAffineExpression.parameter("N"))
        left = IndexedAffineExpression((IndexedAffineTerm("x", ("i", "j - 1")),), 1, ("j",), (inner,))
        row = IndexedLinearPredicate(left, "<=", IndexedAffineExpression((IndexedAffineTerm("y", ("i",)),)), ("i",), (domain,))
        normalized = row.alpha_normalized()
        self.assertEqual(normalized.left.terms[0].indices, ("r0", "s0 - 1"))
        self.assertEqual(normalized.left.sum_domains[0].lower, SymbolicAffineExpression.parameter("r0"))
        contract = ParameterContract(
            (ParameterFact(SymbolicPredicate(SymbolicAffineExpression.parameter("N"), ">=", one), "assumed"),),
            (IndexedParameterFact(row, "derived"),),
        )
        self.assertEqual(parameter_contract_from_record(parameter_contract_to_record(contract)), contract)
        self.assertEqual(len(contract.render()), 2)
        self.assertIn("r0", contract.render()[1])

    def test_scoped_components_are_alpha_normalized_and_serialize(self):
        one = SymbolicAffineExpression.constant_value(1)
        two = SymbolicAffineExpression.constant_value(2)
        domain = SymbolicDomain("j", one, two)
        expression = IndexedScopedAffineExpression(
            (
                IndexedAffineComponent((IndexedAffineTerm("x", ("j",)),), 2, ("j",), (domain,)),
                IndexedAffineComponent(constant=1),
            )
        )
        normalized = expression.alpha_normalized()
        self.assertTrue(any(component.sum_binders == ("s0",) for component in normalized.components))
        family = IndexedAffineFamily((), (), expression)
        pair = IndexedObservablePair("total", family, family)
        self.assertEqual(parse_observable_mapping(serialize_observable_mapping((pair,))), (pair,))
        self.assertEqual(family.alpha_normalized().expression.alpha_normalized(), normalized)
        model = "dvar boolean x[1..2]; minimize 0; subject to {}"
        self.assertEqual(
            ground_indexed_observable_mapping((pair,), model, model)[0].left, AffineObservable((("x_1", 1), ("x_2", 1)), 5)
        )

    def test_contract_and_family_validation(self):
        one = SymbolicAffineExpression.constant_value(1)
        domain = SymbolicDomain("i", one, one)
        expression = IndexedAffineExpression()
        invalid = [
            (lambda: SymbolicPredicate(one, "=", one), "unsupported symbolic relation"),
            (lambda: SymbolicDomain("", one, one), "nonempty name"),
            (lambda: IndexedAffineTerm(""), "declaration name"),
            (lambda: IndexedAffineComponent(sum_binders=("i",)), "unique binder"),
            (lambda: IndexedAffineComponent(sum_binders=("",), sum_domains=(domain,)), "nonempty"),
            (lambda: IndexedAffineComponent(sum_binders=("j",), sum_domains=(domain,)), "name their binders"),
            (lambda: IndexedScopedAffineExpression(()), "at least one component"),
            (lambda: IndexedAffineExpression(sum_binders=("i",)), "unique binder"),
            (lambda: IndexedAffineFamily(("i",), (), expression), "unique binder"),
            (lambda: IndexedAffineFamily(("",), (domain,), expression), "nonempty"),
            (
                lambda: IndexedAffineFamily(
                    ("i",), (domain,), IndexedAffineExpression(sum_binders=("i",), sum_domains=(domain,))
                ),
                "shadow",
            ),
            (lambda: IndexedAffineFamily(("i",), (domain,), expression, "bad"), "symbolic predicate"),
            (
                lambda: IndexedObservablePair(
                    "", IndexedAffineFamily((), (), expression), IndexedAffineFamily((), (), expression)
                ),
                "nonempty",
            ),
            (lambda: ParameterFact(SymbolicPredicate(one, "==", one), "bad"), "provenance"),
            (lambda: IndexedLinearPredicate(expression, "==", expression, ("i",), ()), "unique binder"),
            (lambda: IndexedLinearPredicate(expression, "==", expression, ("j",), (domain,)), "corresponding binders"),
            (lambda: IndexedParameterFact(IndexedLinearPredicate(expression, "==", expression), "assumed"), "provenance"),
        ]
        for construct, message in invalid:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                construct()
        for record, message in (
            ([], "JSON objects"),
            ({"facts": {}}, "JSON arrays"),
            ({"facts": [{}]}, "predicate and provenance"),
        ):
            with self.subTest(record=record), self.assertRaisesRegex(ValueError, message):
                parameter_contract_from_record(record)
        for record, message in (
            ({}, "kind="),
            ({"kind": "indexed"}, "string name"),
            ({"kind": "indexed", "name": "x"}, "missing left"),
        ):
            with self.subTest(record=record), self.assertRaisesRegex(ValueError, message):
                indexed_observable_pair_from_record(record)
