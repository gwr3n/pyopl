import re


def _with_repair_hint(message: str) -> str:
    if "Hint:" in message:
        return message

    patterns: list[tuple[str, str]] = [
        (
            r"^Syntax error at or near token",
            "Hint: rewrite the construct using simpler supported PyOPL syntax. In particular, avoid filtered declarations, advanced inline indexing, and OPL-only shorthand that is not part of this implementation.",
        ),
        (
            r"^Syntax error at end of file",
            "Hint: the model or data is likely missing a closing delimiter such as ';', '}', ']', or ')'. Check the most recent declaration or constraint block.",
        ),
        (
            r"^Illegal character in \.dat file",
            "Hint: use plain PyOPL .dat syntax, not escaped JSON text or unsupported OPL-specific formatting.",
        ),
        (
            r"^Unsupported index expression type",
            "Hint: precompute the needed lookup table in the model/data and index it directly with iterator variables instead of indexing one indexed expression with another.",
        ),
        (
            r"(Undeclared symbol|Unknown name|Unknown symbol|not found in environment|Parameter '.*' not found|AST parameter '.*' with indices .* not found)",
            "Hint: declare the symbol explicitly in the model or provide matching data with the expected name and index structure.",
        ),
        (
            r"(Range bounds must be integer-valued|Index range bounds must be integer-valued|Range bound must be integer-valued)",
            "Hint: use integer-valued literals, parameters, or expressions for all range bounds.",
        ),
        (
            r"(Range bounds must be non-negative literals|Negative literal indices are not allowed)",
            "Hint: rewrite the index or range using non-negative bounds supported by this compiler.",
        ),
        (
            r"(Type mismatch in arithmetic|Type mismatch in comparison)",
            "Hint: make both sides type-compatible before combining them. Do not mix strings, booleans, and numeric expressions in the same arithmetic or comparison context.",
        ),
        (
            r"^Unsupported function",
            "Hint: use only the functions supported by this implementation, or precompute the value in data or an auxiliary parameter.",
        ),
        (
            r"(Non-ground boolean|Condition does not evaluate to boolean)",
            "Hint: boolean comparisons cannot be used as arithmetic values. Rewrite them as explicit constraints or binary-variable linkages.",
        ),
        (
            r"Division by zero",
            "Hint: guard the denominator or rewrite the expression so the divisor is guaranteed nonzero at compile time.",
        ),
        (
            r"^Unsupported ",
            "Hint: rewrite this construct using the supported PyOPL subset implemented by this compiler.",
        ),
    ]

    for pattern, hint in patterns:
        if re.search(pattern, message):
            return f"{message} {hint}"
    return message


class SemanticError(Exception):
    """Custom exception for semantic errors."""

    def __init__(self, message, lineno=None):
        enriched = _with_repair_hint(str(message))
        self.message = enriched
        self.lineno = lineno
        super().__init__(f"Semantic Error (Line {lineno}): {enriched}" if lineno else f"Semantic Error: {enriched}")
