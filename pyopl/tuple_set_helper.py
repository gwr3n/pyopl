class TupleSetHelper:
    @staticmethod
    def _to_tuple_recursive(val):
        # Recursively convert nested tuple literals (dicts with 'elements') to native Python tuples
        if isinstance(val, dict) and "elements" in val:
            return tuple(TupleSetHelper._to_tuple_recursive(e) for e in val["elements"])
        elif isinstance(val, (list, tuple)):
            return tuple(TupleSetHelper._to_tuple_recursive(e) for e in val)
        else:
            return val

    @staticmethod
    def get_tuple_set(set_name, ast, data_dict):
        # Always prefer data_dict if present, and ensure all elements are tuples (recursively)
        if set_name in data_dict:
            tuple_set = data_dict[set_name]
            if isinstance(tuple_set, dict):
                if "elements" in tuple_set:
                    return [TupleSetHelper._to_tuple_recursive(t) for t in tuple_set["elements"]]
                if "value" in tuple_set:
                    return [TupleSetHelper._to_tuple_recursive(t) for t in tuple_set["value"]]
            elif isinstance(tuple_set, list):
                return [TupleSetHelper._to_tuple_recursive(t) for t in tuple_set]
        # Fallback: try to find in AST declarations
        for decl in ast.get("declarations", []):
            if decl.get("name") == set_name:
                if decl.get("type") in ("set", "set_of_tuples"):
                    if "elements" in decl:
                        return [TupleSetHelper._to_tuple_recursive(t) for t in decl["elements"]]
                    if "value" in decl:
                        return [TupleSetHelper._to_tuple_recursive(t) for t in decl["value"]]
        return []
