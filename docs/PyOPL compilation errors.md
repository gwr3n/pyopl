# PyOPL compilation errors

This table lists the distinct compile-time `SemanticError` message templates currently emitted by PyOPL.

- Scope: lexing, parsing, semantic validation, `.dat` parsing, AST rewrites, computed-parameter evaluation, and solver-specific code generation.
- Deduplication: identical message templates raised from multiple call sites are merged into one row.
- Excluded from the table: pass-through rethrows that simply re-emit `e.message`, plus runtime solve/execute errors outside compilation.

| Phase | Source | Error template | Typical trigger |
| --- | --- | --- | --- |
| Lexing | `pyopl_core.py:290` | `Illegal character '{t.value[0]}'` | An unsupported character appears in a model file. |
| Lexing | `pyopl_core.py:3111` | `Illegal character in .dat file: '{t.value[0]}'` | An unsupported character appears in a `.dat` file. |
| Symbol table | `pyopl_core.py:78` | `Cannot exit global scope.` | Internal scope stack underflow while leaving parser scope. |
| Symbol table | `pyopl_core.py:93,3516,3525` | `Identifier '{name}' is reserved and cannot be used in the model (conflicts with Python keywords or built-ins). Please rename it.` | A model symbol or merged data key uses a reserved Python identifier such as a keyword or `len`. |
| Symbol table | `pyopl_core.py:100` | `Symbol '{name}' already declared in this scope.` | A symbol is redeclared in the same scope. |
| Symbol table | `pyopl_core.py:123` | `Undeclared symbol '{name}'.` | A referenced symbol was never declared. |
| Parsing | `pyopl_core.py:1414` | `Syntax error at or near token {tok_type}, value '{tok_val}'.` | The parser encounters an unexpected token in a model file. |
| Parsing | `pyopl_core.py:1419` | `Syntax error at end of file (EOF).` | The model ends before a valid parse can complete. |
| Parsing | `pyopl_core.py:323` | `Set '{p.NAME1}' is declared as a set of tuples '{{p.NAME0}}', but scalar elements were provided. Use tuple literals like <...>.` | A typed tuple set is initialized with scalar values instead of tuple literals. |
| Parsing | `pyopl_core.py:800` | `Invalid identifier token (expected NAME).` | A grammar position that requires an identifier received some other token. |
| Declarations | `pyopl_core.py:1455,1466` | `String decision variables are not supported. Use 'string' only for tuple fields or typed scalar sets.` | `dvar string ...` is declared. |
| Declarations | `pyopl_core.py:1501,1656,2926` | `Symbol '{dim_name}' used as dimension must be a 'range' or 'set', but found '{symbol_info['type']}'.` | An indexed declaration uses a symbol as a dimension that is not a declared range or set. |
| Declarations | `pyopl_core.py:1506` | `Undeclared symbol '{name}' used as dimension.` | An indexed declaration references a dimension symbol that does not exist. |
| Declarations | `pyopl_core.py:1511,1663,2933` | `Single number index '{dim_spec['value']}' not allowed in declaration dimensions. Use 'range' like [1..N] or a named 'set'/'range'.` | A declaration uses a literal index where a dimension domain is required. |
| Declarations | `pyopl_core.py:1516,1668,2938` | `Unsupported dimension type in declaration: {dim_spec['type']}` | A declaration contains an index-domain AST node that the compiler does not support. |
| Sets | `pyopl_core.py:1030` | `Scalar sets in model files must be typed. Use '{int}', '{float}', '{boolean}', or '{string}': e.g., {int} S = {1,2};` | An untyped scalar set literal is declared in a model file. |
| Sets | `pyopl_core.py:1059,1066` | `Expected integer literal in {int} set, got '{v}'.` | A `{int}` set contains a non-integer literal. |
| Sets | `pyopl_core.py:1075,1082` | `Expected numeric literal in {float} set, got '{v}'.` | A `{float}` set contains a non-numeric literal. |
| Expressions | `pyopl_core.py:999` | `Field access '{field}' applied to non-tuple expression.` | `x.y` is used on a value that is not tuple-typed. |
| Expressions | `pyopl_core.py:1007,3816` | `Unknown field '{field}' for tuple type '{base_sem_type}'.` | A tuple field name does not exist on the tuple type. |
| Expressions | `pyopl_core.py:1020` | `Unsupported function '{func}'. Only sqrt(...) is supported.` | An unsupported unary function call is parsed in the older function-call path. |
| Expressions | `pyopl_core.py:1193` | `Cannot apply unary minus to a boolean expression.` | Unary `-` is applied to a boolean expression. |
| Expressions | `pyopl_core.py:1237` | `Chained comparisons (e.g., a <= b <= c) are not supported. Split into two constraints: a <= b; b <= c;` | A relational or equality expression chains comparisons. |
| Expressions | `pyopl_core.py:1309` | `Logical '&&' requires boolean operands.` | `&&` is applied to non-boolean operands. |
| Expressions | `pyopl_core.py:1325` | `Logical '||' requires boolean operands.` | `||` is applied to non-boolean operands. |
| Objective labels | `pyopl_core.py:1776` | `Objective labels must be unindexed. Use 'minimize z: expr;' not 'minimize z[...]: expr;'.` | An indexed label is used with `minimize ... :`. |
| Objective labels | `pyopl_core.py:1783` | `Objective labels must be unindexed. Use 'maximize z: expr;' not 'maximize z[...]: expr;'.` | An indexed label is used with `maximize ... :`. |
| Objective labels | `pyopl_core.py:1790` | `Objective labels must be unindexed. Use 'minimize z = expr;' not 'minimize z[...] = expr;'.` | An indexed label is used with `minimize ... =`. |
| Objective labels | `pyopl_core.py:1797` | `Objective labels must be unindexed. Use 'maximize z = expr;' not 'maximize z[...] = expr;'.` | An indexed label is used with `maximize ... =`. |
| Constraint labels | `pyopl_core.py:1811,1826` | `Constraint labels may not prefix a forall. To label constraints produced by a forall, put the label inside the forall, e.g.:\n  forall(i in I) ct: expr;\nor\n  forall(i in I) { ct: expr; }` | A top-level label is placed before a `forall` instead of inside it. |
| Constraint labels | `pyopl_core.py:1837` | `Indexed constraint labels are not allowed. Use an unindexed label, e.g., 'ct: ...;'. To label constraints generated by a forall, put the plain label inside the forall: forall(i in I) ct: ...;` | A top-level constraint label uses indices such as `ct[i]:`. |
| Constraint labels | `pyopl_core.py:1852` | `Indexed constraint labels are not allowed inside forall. Use a plain, unindexed label inside the forall: forall(i in I) ct: ...;` | A labeled constraint inside `forall` uses indices. |
| Constraints | `pyopl_core.py:1897` | `Implication sides must be constraints or boolean expressions.` | Either side of `=>` is neither a comparison nor a boolean-valued expression. |
| Constraints | `pyopl_core.py:1936` | `'{op}' operator only supported for int/float/boolean types, got '{left_type}' and '{right_type}'.` | A comparison operator is applied to unsupported semantic types. |
| Constraints | `pyopl_core.py:1954` | `Standalone arithmetic expression not allowed as constraint; use comparison (e.g., expr <= value).` | A constraint is just an arithmetic expression rather than a boolean/comparison. |
| Constraints | `pyopl_core.py:1989` | `Labeled constraints must be comparison or boolean expression.` | A labeled constraint body is not boolean-valued. |
| Constraints | `pyopl_core.py:2023` | `forall labeled constraint must be comparison or boolean expression.` | A labeled constraint inside `forall` is not boolean-valued. |
| Typing | `pyopl_core.py:2067` | `Type mismatch in comparison: {left_type} vs {right_type}` | Incompatible semantic types are compared. |
| Ranges | `pyopl_core.py:1543,2152,2187` | `Range bounds must be non-negative literals.` | A range bound is a negative numeric literal. |
| Ranges | `pyopl_core.py:1554` | `Range start ({s_val}) cannot be greater than end ({e_val}).` | A declared range has descending bounds. |
| Ranges | `pyopl_core.py:2149` | `Range bounds must be integer-valued.` | A general range bound expression is not integer-typed. |
| Ranges | `pyopl_core.py:2173` | `Index range bounds must be integer-valued.` | An index-domain range uses non-integer bounds. |
| Ranges | `pyopl_core.py:2176` | `Index range bounds must be non-negative literals.` | An index-domain range uses negative literal bounds. |
| Ranges | `pyopl_core.py:2184` | `Range bound must be integer-valued, got type '{expr['sem_type']}'.` | A range expression reduces to a non-integer semantic type. |
| Indexing | `pyopl_core.py:2196,2204` | `Negative literal indices are not allowed.` | An index expression contains a literal negative index. |
| Indexing | `pyopl_core.py:2232` | `Unsupported index expression type: {expr['type']}.` | The parser encounters an index-expression AST form it does not accept. |
| `dexpr` | `pyopl_core.py:521` | `Symbol '{rng['name']}' used in dexpr index is not a declared range or set.` | A `dexpr` iterator ranges over an undeclared or invalid symbol. |
| `dexpr` | `pyopl_core.py:818` | `Expected indexed dexpr, but '{name}' is declared with indices. Missing dimensions.` | A reference to an indexed `dexpr` omits its indices. |
| Variable use | `pyopl_core.py:824` | `Expected scalar variable, but '{name}' is an indexed variable. Missing dimensions.` | An indexed variable is referenced without required indices. |
| Arithmetic | `pyopl_core.py:2475` | `Cannot use tuple variable '{left_expr.get('value', '?')}' of type '{left_type}' in arithmetic; use a field access like '{left_expr.get('value', '?')}.field'.` | A tuple-typed value appears directly in arithmetic on the left-hand side. |
| Arithmetic | `pyopl_core.py:2483` | `Cannot use tuple variable '{right_expr.get('value', '?')}' of type '{right_type}' in arithmetic; use a field access like '{right_expr.get('value', '?')}.field'.` | A tuple-typed value appears directly in arithmetic on the right-hand side. |
| Arithmetic | `pyopl_core.py:2490` | `Type mismatch in arithmetic: {left_type} vs {right_type}` | Incompatible semantic types are used in arithmetic. |
| Functions | `pyopl_core.py:2517` | `sqrt(...) takes exactly one argument.` | `sqrt` is called with the wrong arity. |
| Functions | `pyopl_core.py:2521,4046` | `{func}(...) requires at least one argument.` | `minl` or `maxl` is called with zero arguments. |
| Functions | `pyopl_core.py:2526` | `{func}(...) expects numeric arguments.` | `sqrt`, `minl`, or `maxl` receives a non-numeric argument. |
| Functions | `pyopl_core.py:2529` | `Unsupported function '{func}'. Only sqrt, maxl, minl are supported.` | A function call uses a name outside the supported function set. |
| Aggregates | `pyopl_core.py:2536,2551` | `min aggregate expects numeric expression.` | `min(...)` ranges over a non-numeric body expression. |
| Aggregates | `pyopl_core.py:2566,2580` | `max aggregate expects numeric expression.` | `max(...)` ranges over a non-numeric body expression. |
| Indexed references | `pyopl_core.py:2628` | `Expected indexed variable, but '{p.NAME}' is a scalar variable.` | A scalar variable is used with indices. |
| Indexed references | `pyopl_core.py:2636` | `Incorrect number of dimensions for '{p.NAME}'. Declared {len(declared_dims)}, but used {len(used_dims)}.` | A variable is indexed with the wrong arity. |
| Indexed references | `pyopl_core.py:2614` | `Incorrect number of dimensions for dexpr '{p.NAME}'. Declared {len(decl_dims)}, but used {len(used_dims)}.` | A `dexpr` is indexed with the wrong arity. |
| Indexed references | `pyopl_core.py:2670` | `Index {used_index_spec['value']} for dimension {i + 1} of '{p.NAME}' is out of declared range [{s_val}..{e_val}].` | A literal index falls outside a declared integer range dimension. |
| Indexed references | `pyopl_core.py:2683` | `Index expression for dimension {i + 1} of '{p.NAME}' must be integer-valued, got type '{index_sem_type}'.` | An integer/range dimension is indexed by a non-integer expression. |
| Indexed references | `pyopl_core.py:2694` | `Unsupported index type for integer/range dimension: {used_index_spec['type']}` | An unsupported index AST form is used against an integer/range dimension. |
| Indexed references | `pyopl_core.py:2719` | `Index expression for tuple set dimension {i + 1} of '{p.NAME}' must be of tuple type '{tuple_type}', got type '{idx_sem_type}'.` | A tuple-set dimension is indexed by the wrong tuple type. |
| Indexed references | `pyopl_core.py:2729` | `Index expression for set dimension {i + 1} of '{p.NAME}' must be {base_type}-valued, got type '{idx_sem_type}'.` | A typed scalar set dimension is indexed by the wrong scalar type. |
| Indexed references | `pyopl_core.py:2737` | `Index expression for set dimension {i + 1} of '{p.NAME}' must be string- or integer-valued, got type '{idx_sem_type}'.` | A generic set dimension is indexed by an unsupported semantic type. |
| Indexed references | `pyopl_core.py:2743` | `Dimension {i + 1} of '{p.NAME}' is not indexable. Declared as type: {dim_type}.` | A declaration dimension does not support indexing semantics. |
| Iterator headers | `pyopl_core.py:2354` | `Symbol '{rng['name']}' used in 'in' clause is not declared.` | A `sum` or `forall` iterator domain references an undeclared symbol. |
| Iterator headers | `pyopl_core.py:2366` | `Symbol '{rng['name']}' used in 'in' clause is not a declared range or set.` | A `sum` or `forall` iterator domain references a symbol that is not a range or set. |
| `.dat` parsing | `pyopl_core.py:3119` | `Typed scalar set prefixes (e.g. '{string} S = {...};') are not allowed in .dat files. Declare the typed set in the model (.mod) and use 'S = {...};' in the data file.` | A `.dat` file uses a typed scalar set declaration instead of assigning to a model-declared set. |
| `.dat` parsing | `pyopl_core.py:3128` | `Typed set-of-tuples prefix ('{TupleType} S = {...};') is not allowed in .dat files. Declare '{TupleType} S;' in the model and use 'S = { <...>, ... };' in the data file.` | A `.dat` file uses a typed tuple-set declaration. |
| `.dat` parsing | `pyopl_core.py:3293` | `Range bounds in .dat file must be integers, got {type(start_val).__name__} and {type(end_val).__name__}.` | A `.dat` range assignment uses non-integer endpoints. |
| `.dat` parsing | `pyopl_core.py:3298` | `Range bounds in .dat file must be non-negative, got {start_val}..{end_val}.` | A `.dat` range assignment uses negative endpoints. |
| `.dat` parsing | `pyopl_core.py:3303` | `Range start ({start_val}) cannot be greater than end ({end_val}).` | A `.dat` range assignment uses descending bounds. |
| `.dat` parsing | `pyopl_core.py:3409` | `Syntax error in .dat file at or near token {p.type}, value '{p.value}'.` | The data parser encounters an unexpected token. |
| `.dat` parsing | `pyopl_core.py:3415` | `Syntax error in .dat file at end of file (EOF).` | The data file ends before a valid parse can complete. |
| Reserved names after merge | `pyopl_core.py:3516` | `Identifier '{bad}' is reserved and cannot be used as a model symbol. Please rename it in the .mod file.` | A final model declaration name conflicts with a reserved Python identifier. |
| Reserved names after merge | `pyopl_core.py:3525` | `Identifier '{bad}' is reserved and cannot appear as a data key (would shadow Python keywords or built-ins). Please rename it in the .dat or model data.` | A data key conflicts with a reserved Python identifier. |
| Set comprehensions | `pyopl_core.py:3544,3705` | `Unknown name in range bound: {expr.get('value')}` | A bound expression in a comprehension or computed parameter references an unknown symbol. |
| Set comprehensions | `pyopl_core.py:3558,3718` | `Unsupported bound expr: {expr}` | A range bound expression uses an unsupported AST form. |
| Set comprehensions | `pyopl_core.py:3578` | `Named range '{rng_name}' not found for set comprehension.` | A set comprehension references a named range missing from merged data/model declarations. |
| Set comprehensions | `pyopl_core.py:3609` | `Unsupported iterator range type '{rng['type']}' in set comprehension for '{decl.get('name')}'.` | A comprehension iterator uses an unsupported range type. |
| Computed parameters | `pyopl_core.py:3739` | `Named range '{rng_name}' not found for computed parameter.` | A computed parameter iterator references a missing named range. |
| Computed parameters | `pyopl_core.py:3781` | `Unknown name '{nm}' in computed parameter expression.` | A computed parameter expression references an unknown symbol. |
| Computed parameters | `pyopl_core.py:3811` | `Cannot resolve tuple type for field access in computed parameter.` | Tuple field access in a computed parameter cannot be typed. |
| Computed parameters | `pyopl_core.py:3820` | `Field access failed on base value: {e}` | Evaluating a tuple field access in a computed parameter fails at runtime. |
| Computed parameters | `pyopl_core.py:3828` | `Parameter '{base}' not found for indexed access.` | A computed parameter indexes into a missing parameter. |
| Computed parameters | `pyopl_core.py:3838` | `List parameter '{base}' requires integer indices, got {type(idx_val).__name__}: {idx_val}` | A list-valued parameter is indexed with a non-integer key. |
| Computed parameters | `pyopl_core.py:3845` | `Index out of bounds for '{base}' at {idx_val}: {e}` | A computed parameter uses an out-of-range list index. |
| Computed parameters | `pyopl_core.py:3851` | `Key '{idx_val}' not found in parameter '{base}': {e}` | A dict-valued parameter is indexed by a missing key. |
| Computed parameters | `pyopl_core.py:3853` | `Cannot index into value of type {type(cur).__name__} for '{base}'.` | A computed parameter attempts to index a non-indexable value. |
| Computed parameters | `pyopl_core.py:3883` | `Unsupported range in sum aggregate: {rng['type']}` | A computed-parameter `sum(...)` iterator uses an unsupported range type. |
| Computed parameters | `pyopl_core.py:3944` | `Unsupported range in aggregate: {rng['type']}` | A computed-parameter `min(...)` or `max(...)` iterator uses an unsupported range type. |
| Computed parameters | `pyopl_core.py:3991` | `Aggregate domain is empty in computed parameter expression.` | A computed-parameter aggregate ranges over an empty domain. |
| Computed parameters | `pyopl_core.py:4028` | `Unsupported operator in computed parameter expression: {op}` | A computed-parameter expression uses an unsupported binary operator. |
| Computed parameters | `pyopl_core.py:4038` | `Unsupported function '{fname}' in computed parameter expression.` | A computed-parameter expression calls an unsupported function. |
| Computed parameters | `pyopl_core.py:4044` | `{t} in parameter must be numeric and ground.` | A computed-parameter aggregate argument is not ground numeric. |
| Computed parameters | `pyopl_core.py:4048` | `Unsupported node in computed parameter expression: {t}` | A computed-parameter expression contains an unsupported AST node type. |
| Computed parameters | `pyopl_core.py:4060` | `Field access in computed parameter indices not supported.` | A computed-parameter index expression includes tuple field access. |
| Computed parameters | `pyopl_core.py:4071` | `Unsupported index binop: {op}` | A computed-parameter index expression uses an unsupported operator. |
| Computed parameters | `pyopl_core.py:4078` | `Unsupported index expr: {t}` | A computed-parameter index expression uses an unsupported AST node type. |
| Computed parameters | `pyopl_core.py:4134` | `Unsupported iterator range type '{rng['type']}' for computed parameter '{name}'.` | A computed parameter iterator uses an unsupported domain kind. |
| Parameter shape validation | `pyopl_core.py:4250` | `Parameter '{name}' expected nested list matching declared domains, got {type(node).__name__}` | Inline or merged parameter data has too little nesting for its declared dimensions. |
| Parameter shape validation | `pyopl_core.py:4256` | `Parameter '{name}' data length shorter than domain at dimension {depth + 1}` | Parameter data does not supply enough entries for a declared domain. |
| Parameter shape validation | `pyopl_core.py:4289` | `Parameter '{param_name}' is 1-D over '{d.get('name', '')}' but data value for key {repr(k)} is an array; expected a scalar (e.g., 2.0). Remove extra brackets like [2.0] -> 2.0.` | A keyed 1-D parameter stores arrays where scalars are required. |
| Parameter shape validation | `pyopl_core.py:4313` | `Range bound refers to unknown name '{expr['value']}'` | Shape validation encounters a named range bound that does not exist. |
| Parameter shape validation | `pyopl_core.py:4341` | `Parameter '{param_name}' expected a {len(dims)}D array, got scalar at dimension {dim + 1}.` | A multidimensional parameter is given a scalar too early in its nesting. |
| Parameter shape validation | `pyopl_core.py:4345` | `Parameter '{param_name}' data length {len(param_data)} does not match declared dimension '{d.get('name')}' of length {expected_len} at dimension {dim + 1}.` | A parameter array length does not match its declared dimension length. |
| Typed set validation | `pyopl_core.py:4404` | `Set '{name}' must be assigned a list of values, got {type(values).__name__}.` | A typed set is assigned a non-list value after data merge. |
| Typed set validation | `pyopl_core.py:4408` | `All elements of set '{name}' must be integers.` | A `{int}` set contains a non-integer after merge. |
| Typed set validation | `pyopl_core.py:4411` | `All elements of set '{name}' must be numeric (int/float).` | A `{float}` set contains a non-numeric value after merge. |
| Typed set validation | `pyopl_core.py:4417` | `All elements of set '{name}' must be booleans (true/false).` | A `{boolean}` set contains a non-boolean value after merge. |
| Typed set validation | `pyopl_core.py:4420` | `All elements of set '{name}' must be strings.` | A `{string}` set contains a non-string value after merge. |
| Named range validation | `pyopl_core.py:4484` | `Range '{name}' was supplied in the data file, but ranges used for indexing must be declared with explicit bounds in the model file. Declare it in the model (e.g., 'range {name} = 1..N;') and remove it from the .dat.` | A range used for indexing is supplied only through data instead of being declared in the model. |
| Named range validation | `pyopl_core.py:4490` | `Range '{name}' is used as an index but not declared in the model.` | A named range appears in dimensions or iterators without a model declaration. |
| If-constraint rewrite | `pyopl_core.py:4909,4957` | `Malformed if-constraint: missing condition.` | An internal `if`-constraint AST node is missing its condition. |
| If-constraint rewrite | `pyopl_core.py:4912` | `Condition of if-constraint inside forall must not reference decision variables.` | A `forall`-scoped `if` condition depends on decision variables. |
| If-constraint rewrite | `pyopl_core.py:4961` | `Condition of if-constraint must be ground (must not reference decision variables).` | A top-level `if` condition depends on decision variables. |
| If-constraint rewrite | `pyopl_core.py:4964` | `Condition of if-constraint at top level cannot reference iterators.` | A top-level `if` condition still contains iterator references. |
| Ground-condition evaluation | `pyopl_core.py:5176` | `Condition does not evaluate to boolean: {expr}` | A compile-time condition simplifies to a non-boolean value. |
| Ground-condition evaluation | `pyopl_core.py:5193,5197` | `Unknown symbol in ground expression: {name}` | A compile-time ground expression references an unknown symbol. |
| Ground-condition evaluation | `pyopl_core.py:5209` | `Index error in ground expression {base}[...]: {e}` | Evaluating a ground indexed expression fails due to bad indexing. |
| Ground-condition evaluation | `pyopl_core.py:5254` | `Error evaluating ground binop '{op}': {e}` | Compile-time evaluation of a binary operator raises an error. |
| Ground-condition evaluation | `pyopl_core.py:5255` | `Unsupported operator in ground expression: {op}` | A compile-time condition uses an unsupported operator. |
| Ground-condition evaluation | `pyopl_core.py:5257` | `Unsupported expression in ground condition: {t}` | A compile-time condition uses an unsupported AST node type. |
| Max/min lowering | `pyopl_core.py:5067,5332` | `Non-convex objective: supported only minimize max(...) or maximize min(...).` | `maxl` or `minl` appears in an unsupported objective form. |
| Max/min lowering | `pyopl_core.py:5278` | `maxl/minl require at least one argument.` | `maxl` or `minl` receives zero arguments during convex lowering. |
| Max/min lowering | `pyopl_core.py:5379` | `Non-convex: equality with maxl/minl is not supported.` | A constraint uses `==` with `maxl` or `minl`. |
| Max/min lowering | `pyopl_core.py:5382` | `Non-convex or unsupported placement of maxl/minl in constraint. Allowed only in: maxl(...) <= rhs, lhs >= maxl(...), minl(...) >= rhs, lhs <= minl(...).` | `maxl` or `minl` appears in an unsupported inequality orientation. |
| Max/min lowering | `pyopl_core.py:5390` | `Non-convex: maxl/minl not supported inside implication constraints.` | `maxl` or `minl` appears inside an implication constraint. |
| Max/min lowering | `pyopl_core.py:5104` | `Unsupported non-convex aggregate placement (==, >, <, or reversed forms).` | A non-convex aggregate is placed where the lowering only supports convex forms. |
| Gurobi codegen | `gurobi_codegen.py:354` | `Parameter '{param_name}' expected a {len(dims)}D array, got scalar at dimension {dim + 1}.` | A multidimensional parameter is provided as a scalar while generating Gurobi data access code. |
| Gurobi codegen | `gurobi_codegen.py:368` | `Parameter '{param_name}' data length {len(param_data)} does not match declared dimension '{d.get('name')}' of length {expected_len} at dimension {dim + 1}.` | Parameter data length does not match a declared dimension during Gurobi preprocessing. |
| Gurobi codegen | `gurobi_codegen.py:505` | `Parameter '{name}' declared as 1-D over '{dims[0].get('name', '')}' expects scalar values per key, but data provides an array for key {repr(bad_key)}. Use scalar values (e.g., 2.0), not [2.0].` | A keyed 1-D parameter provides arrays rather than scalars. |
| Gurobi codegen | `gurobi_codegen.py:603` | `Parameter '{name}' dimensionality mismatch: data depth {len(idx_pos)} vs declared {len(keys_per_dim)}.` | Nested parameter data depth does not match declared dimensionality. |
| Gurobi codegen | `gurobi_codegen.py:608` | `Parameter '{name}' index {pi} out of bounds for dimension {dim_i + 1}.` | Parameter data is indexed outside its declared extent. |
| Gurobi codegen | `gurobi_codegen.py:977,1047` | `Named range '{rng_dim.get('name')}' has no bounds.` | Code generation needs a named range that exists but has no resolved bounds. |
| Gurobi codegen | `gurobi_codegen.py:984` | `Parameter '{name}' row for key {k} has length {len(row)}; expected {expected_len}.` | A keyed row in parameter data has the wrong width. |
| Gurobi codegen | `gurobi_codegen.py:1136` | `Parameter '{name}' has {len(value)} items but declared set '{range_name}' has {len(set_elems)} elements.` | Parameter data length does not match its declared set domain size. |
| Gurobi codegen | `gurobi_codegen.py:1178` | `Parameter '{name}' has {len(value)} items but declared range '{range_name}' expects {expected_len}.` | Parameter data length does not match its declared numeric range size. |
| Gurobi codegen | `gurobi_codegen.py:2073` | `Range '{name}' declared as external. Ranges must be defined in the model with explicit bounds (e.g., range Items = 1..N;)` | A range is declared with `= ...` instead of explicit bounds. |
| Gurobi codegen | `gurobi_codegen.py:2083` | `Set '{name}' declared in .mod but not found in .dat file.` | An external set declaration has no matching data. |
| Gurobi codegen | `gurobi_codegen.py:2115` | `Parameter '{name}' length {len(vals)} does not match index set '{set_name}' length {len(tuple_keys)}.` | A parameter indexed by a set of tuples has the wrong number of values. |
| Gurobi codegen | `gurobi_codegen.py:2130` | `Parameter '{name}' declared in .mod but not found in .dat file. Add '= ...;' to explicitly declare it as external if intended.` | A parameter is missing from data even though it was not declared external. |
| Gurobi codegen | `gurobi_codegen.py:2144` | `Parameter '{name}' declared with '= ...' in .mod but not found in .dat file.` | An explicitly external parameter has no data supplied. |
| Gurobi codegen | `gurobi_codegen.py:3416` | `Range '{name}' not found in declarations.` | Code generation needs a range declaration that was never created. |
| SciPy codegen | `scipy_codegen_csc.py:85` | `minl() requires at least one argument` | `minl` is evaluated with zero arguments in SciPy compilation. |
| SciPy codegen | `scipy_codegen_csc.py:106` | `maxl() requires at least one argument` | `maxl` is evaluated with zero arguments in SciPy compilation. |
| SciPy codegen | `scipy_codegen_csc.py:171` | `ExpressionEvaluator.eval: handler for type '{t}' returned non-tuple result: {result}` | A SciPy expression handler returns an unexpected value shape. |
| SciPy codegen | `scipy_codegen_csc.py:228` | `_eval_field_access: base expression '{base}' could not be evaluated (unknown type or error)` | Field access evaluation fails in the SciPy expression evaluator. |
| SciPy codegen | `scipy_codegen_csc.py:600` | `_handle_binop_add: left or right is None: left={l_result}, right={r_result}` | Addition evaluation receives a missing operand result. |
| SciPy codegen | `scipy_codegen_csc.py:602` | `_handle_binop_add: left or right did not return a tuple: left={l_result}, right={r_result}` | Addition evaluation receives an operand result in an unexpected format. |
| SciPy codegen | `scipy_codegen_csc.py:618` | `_handle_binop_sub: left or right is None: left={l_result}, right={r_result}` | Subtraction evaluation receives a missing operand result. |
| SciPy codegen | `scipy_codegen_csc.py:620` | `_handle_binop_sub: left or right did not return a tuple: left={l_result}, right={r_result}` | Subtraction evaluation receives an operand result in an unexpected format. |
| SciPy codegen | `scipy_codegen_csc.py:637` | `_handle_binop_mul: left or right is None: left={l_result}, right={r_result}` | Multiplication evaluation receives a missing operand result. |
| SciPy codegen | `scipy_codegen_csc.py:639` | `_handle_binop_mul: left or right did not return a tuple: left={l_result}, right={r_result}` | Multiplication evaluation receives an operand result in an unexpected format. |
| SciPy codegen | `scipy_codegen_csc.py:669` | `_handle_binop_div: left or right is None: left={l_result}, right={r_result}` | Division evaluation receives a missing operand result. |
| SciPy codegen | `scipy_codegen_csc.py:671` | `_handle_binop_div: left or right did not return a tuple: left={l_result}, right={r_result}` | Division evaluation receives an operand result in an unexpected format. |
| SciPy codegen | `scipy_codegen_csc.py:690` | `Division by zero` | Constant evaluation encounters division by zero. |
| SciPy codegen | `scipy_codegen_csc.py:1376,1733,1984` | `Parameter or variable '{name}' with indices {indices} not found in environment.` | SciPy expression evaluation cannot resolve an indexed parameter or variable. |
| SciPy codegen | `scipy_codegen_csc.py:2103` | `AST parameter '{name}' with indices {indices} not found.` | An indexed AST parameter reference cannot be resolved. |
| SciPy codegen | `scipy_codegen_csc.py:3846` | `Unsupported atomic boolean literal` | Boolean linearization encounters an unsupported literal form. |
| SciPy codegen | `scipy_codegen_csc.py:4706` | `Unsupported or non-resolvable boolean expression node type: {t} ({repr(node)})` | Boolean linearization encounters an unsupported or unresolved AST node type. |

## Pass-through rethrows

These sites raise `SemanticError(e.message, ...)` rather than introducing a new message template, so they are not duplicated in the table above:

- `pyopl_core.py:1661`
- `pyopl_core.py:2606`
- `pyopl_core.py:2931`
