class SemanticError(Exception):
    """Custom exception for semantic errors."""

    def __init__(self, message, lineno=None):
        self.message = message
        self.lineno = lineno
        super().__init__(f"Semantic Error (Line {lineno}): {message}" if lineno else f"Semantic Error: {message}")
