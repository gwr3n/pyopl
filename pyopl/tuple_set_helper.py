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
        if set_name in data_dict:
            return TupleSetHelper._from_data_dict(data_dict[set_name])
        return TupleSetHelper._from_ast_declarations(set_name, ast)

    @staticmethod
    def _from_data_dict(tuple_set):
        if isinstance(tuple_set, dict):
            elements = tuple_set.get("elements", tuple_set.get("value"))
            if elements is not None:
                return [TupleSetHelper._to_tuple_recursive(item) for item in elements]
        elif isinstance(tuple_set, (list, tuple)):
            return [TupleSetHelper._to_tuple_recursive(item) for item in tuple_set]
        else:
            return tuple_set
        return []

    @staticmethod
    def _from_ast_declarations(set_name, ast):
        for decl in ast.get("declarations", []):
            if decl.get("name") == set_name:
                if decl.get("type") in ("set", "set_of_tuples", "set_of_tuples_external"):
                    elements = decl.get("elements", decl.get("value"))
                    if elements is not None:
                        return [TupleSetHelper._to_tuple_recursive(item) for item in elements]
        return []
