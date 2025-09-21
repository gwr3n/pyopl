import unittest

from pyopl.pyopl_core import OPLLexer, OPLParser


class TestLogicPrecedence(unittest.TestCase):
    def parse(self, src: str):
        lexer = OPLLexer()
        parser = OPLParser()
        return parser.parse(lexer.tokenize(src))

    def test_and_vs_or_precedence_basic(self):
        # With equality higher precedence than || and &&, expression a == b || c && b parses as (a == b) || (c && b)
        opl = "dvar boolean a; dvar boolean b; dvar boolean c; minimize 0; subject to { a == b || c && b; }"
        ast = self.parse(opl)
        constr = ast["constraints"][0]
        # Boolean expression coerced to == true: constraint.left holds the actual OR tree
        root = constr["left"]
        if root.get("type") == "parenthesized_expression":
            root = root["expression"]
        self.assertEqual(root["type"], "or")
        # left child should be equality binop, right child an 'and'
        self.assertEqual(root["left"]["type"], "binop")
        self.assertEqual(root["left"]["op"], "==")
        self.assertEqual(root["right"]["type"], "and")

    def test_chained_and_or_associativity(self):
        # a == a && b && c || d  parses as ((a == a && b && c) || d) because equality binds before && and && left-associative before ||
        opl = (
            "dvar boolean a; dvar boolean b; dvar boolean c; dvar boolean d; minimize 0; subject to { a == a && b && c || d; }"
        )
        ast = self.parse(opl)
        constr = ast["constraints"][0]
        root = constr["left"]
        if root.get("type") == "parenthesized_expression":
            root = root["expression"]
        self.assertEqual(root["type"], "or")
        left_branch = root["left"]
        # left branch should be an 'and' chain whose left-most element is a binop equality
        while left_branch.get("type") == "parenthesized_expression":
            left_branch = left_branch["expression"]
        self.assertEqual(left_branch["type"], "and")
        deepest = left_branch["left"]
        while deepest.get("type") == "and":
            deepest = deepest["left"]
        self.assertEqual(deepest["type"], "binop")
        self.assertEqual(deepest["op"], "==")

    def test_parentheses_override(self):
        opl = "dvar boolean a; dvar boolean b; dvar boolean c; minimize 0; subject to { a == (b || c) && b; }"
        ast = self.parse(opl)
        constr = ast["constraints"][0]
        root = constr["left"]
        if root.get("type") == "parenthesized_expression":
            root = root["expression"]
        # Top-level should be 'and'
        self.assertEqual(root["type"], "and")
        # Left side of 'and' should be equality binop: a == (b || c)
        left = root["left"]
        if left.get("type") == "parenthesized_expression":
            left = left["expression"]
        self.assertEqual(left["type"], "binop")
        self.assertEqual(left["op"], "==")
        # Its right child should be a parenthesized or-expression
        or_expr = left["right"]
        if or_expr.get("type") == "parenthesized_expression":
            or_expr = or_expr["expression"]
        self.assertEqual(or_expr["type"], "or")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
