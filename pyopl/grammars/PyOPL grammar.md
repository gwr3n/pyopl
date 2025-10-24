# PyOPL Grammar (Aligned with implementation)

This document specifies the grammar implemented in `pyopl/pyopl_core.py` and consumed by both code generators (`gurobi_codegen.py`, `scipy_codegen*.py`). It includes logical operators, implication (`=>`), conditional expressions, field access on tuples, typed scalar sets, tuple arrays, decision-expressions (dexpr), min/max aggregates, sqrt/minl/maxl functions, modulo, and richer .dat file constructs.

Reference: `pyopl/pyopl_core.py`

## Overview

The grammar supports:

- Declarations:
  - decision variables (`dvar`) – scalar and indexed
  - parameters (`param` optional) – scalar (external or inline), indexed (external, inline array, or computed by expression with iterators)
  - ranges
  - typed scalar sets (`{string}`, `{int}`, `{float}`, `{boolean}`)
  - tuple types
  - sets of tuples (typed by tuple type)
  - tuple arrays (`TupleType Arr[Set]`)
  - untyped set-of-tuples assignment in model (tuple literals only)
  - decision expressions (`dexpr`) – scalar and indexed, expanded in-place on use
- Objectives: `minimize` or `maximize` any numeric or boolean expression, with optional label (`minimize z: expr;` or `minimize z = expr;`)
- Constraints:
  - standard comparisons
  - labelled constraints
  - quantified constraints with `forall` (single or block)
  - implication constraints using `=>`
  - conditional constraints `if (...) { ... } [else { ... }]` (condition must be ground)
- Expressions:
  - arithmetic `+ - * / %`, unary `-`
  - logical `&& || !`
  - comparisons `== != <= >= < >`
  - conditional ternary `(? :)` with parenthesized condition
  - function calls: `sqrt(expr)`, `maxl(arg1, ..., argN)`, `minl(arg1, ..., argN)`
  - aggregates: `sum`, `min(... aggregate ...)`, `max(... aggregate ...)`
  - field access `a.b` on tuples (right-assoc)
  - names, indexed names (range, set, tuple indices), tuple literal indices
  - parenthesized expressions
- Data files (.dat):
  - scalars, sets, arrays (nested), ranges, sets of tuples, tuple arrays
  - key-value arrays with string/tuple labels mapping to scalars or arrays
  - untyped set-of-tuples assignments, and typed-set-of-tuples forms

Notes:
- Boolean values may appear in arithmetic and sums; booleans are treated numerically (false=0, true=1) where needed.
- Boolean objectives are allowed.
- `forall` is statement-level (constraints); `sum`/`min`/`max` aggregates are expressions.
- Field access (`a.b`) follows tuple typing metadata and supports chaining.
- In model files, scalar sets must be typed; untyped set assignments in models are only allowed for sets of tuples (tuple literals only).

## BNF Grammar

BNF is simplified for readability. Optional elements are in [brackets]. Alternatives use |. `*` means zero or more. `ε` denotes empty.

### Model Structure

```
<model> ::= <declarations_opt> <objective_section> <constraints_section>

<declarations_opt> ::= <declaration_list> | ε
<declaration_list> ::= <declaration_list> <declaration> | <declaration>

<objective_section> ::= 'minimize' <expression> ';'
                      | 'maximize' <expression> ';'
                      | 'minimize' <NAME> ':' <expression> ';'
                      | 'maximize' <NAME> ':' <expression> ';'
                      | 'minimize' <NAME> '=' <expression> ';'
                      | 'maximize' <NAME> '=' <expression> ';'

<constraints_section> ::= 'subject to' '{' <constraint_list_opt> '}'
<constraint_list_opt> ::= <constraint_list> | ε
<constraint_list> ::= <constraint_list> <constraint> | <constraint>
```

### Constraints

```
<constraint> ::= <expression> '=>' <expression> ';'              // implication
               | <expression> ';'                                // standard or boolean-equals-true
               | <NAME> ':' <expression> ';'                     // labelled
               | 'forall' <forall_index_header> <constraint>     // single (including labelled or implication)
               | 'forall' <forall_index_header> <constraint_block>
               | 'if' '(' <expression> ')' <constraint_block>    // condition must be ground (no dvars)
               | 'if' '(' <expression> ')' <constraint_block> 'else' <constraint_block>

<constraint_block> ::= '{' <constraint_list> '}'
```

Semantics:
- In `<expression> ';'`, if the expression is a comparison, it is used directly; if it is boolean-valued, it is equated to `true`.
- Implication sides accept constraints or boolean expressions; boolean expressions are normalized to equality with `true`.
- `if` condition must be ground (no decision variables); conditions inside `forall` may reference iterators and parameters only.

### Declarations

Types include tuple type names as identifiers.

```
// Decision variables (numeric/boolean only; string not allowed)
<declaration> ::= 'dvar' <dvar_type> <NAME> ';'
                | 'dvar' <dvar_type> <NAME> <indexed_dimensions> ';'

// Ranges
                | 'range' <NAME> '=' <range_expr> '..' <range_expr> ';'
                | 'range' <NAME> ';'

// Untyped set symbol (no assignment in declaration)
                | 'set' <NAME> ';'

// Typed scalar sets (model)
                | <typed_set_declaration>

// Set of tuples (typed by a tuple type name)
                | <set_of_tuples_declaration>

// Untyped set-of-tuples assignment in model (tuple literals only)
                | <untyped_tuple_set_assignment>

// Tuple type declarations
                | <tuple_type_declaration>

// Tuple arrays
                | <tuple_array_declaration>

// Parameters (param keyword optional; external or inline)
                | <param_declaration>

// Decision expressions (expanded on use)
                | <dexpr_declaration>

// Types
<type> ::= 'int' | 'float' | 'int+' | 'float+' | 'boolean' | 'string' | <NAME>  // <NAME> can be a tuple type
<dvar_type> ::= 'int' | 'float' | 'int+' | 'float+' | 'boolean'                  // string is not permitted for dvar
```

Typed scalar sets in models:

```
// Strings
<typed_set_declaration> ::= '{' 'string' '}' <NAME> '=' '{' <element_list_string> '}' ';'
                          | '{' 'string' '}' <NAME> ';'
                          | '{' 'string' '}' <NAME> '=' '...' ';'

// Integers
                          | '{' 'int' '}' <NAME> '=' '{' <element_list_int> '}' ';'
                          | '{' 'int' '}' <NAME> ';'
                          | '{' 'int' '}' <NAME> '=' '...' ';'

// Floats (ints permitted; coerced to float)
                          | '{' 'float' '}' <NAME> '=' '{' <element_list_float> '}' ';'
                          | '{' 'float' '}' <NAME> ';'
                          | '{' 'float' '}' <NAME> '=' '...' ';'

// Booleans
                          | '{' 'boolean' '}' <NAME> '=' '{' <element_list_boolean> '}' ';'
                          | '{' 'boolean' '}' <NAME> ';'
                          | '{' 'boolean' '}' <NAME> '=' '...' ';'
```

Set of tuples in models:

```
// RHS must be tuple literals only (guard rejects scalar elements)
<set_of_tuples_declaration> ::= '{' <NAME> '}' <NAME> '=' '{' <tuple_literal_list> '}' ';'
                              | '{' <NAME> '}' <NAME> ';'
                              | '{' <NAME> '}' <NAME> '=' '...' ';'

// Untyped set-of-tuples assignment allowed in model (tuple literals only)
<untyped_tuple_set_assignment> ::= <NAME> '=' '{' <tuple_literal_list> '}' ';'
```

Tuple types:

```
<tuple_type_declaration> ::= 'tuple' <NAME> '{' <tuple_field_list> '}' [';']
                           | 'tuple' <NAME> '{' '}' [';']
<tuple_field_list> ::= <tuple_field_list> <tuple_field> | <tuple_field>
<tuple_field> ::= <type> <NAME> ';'
```

Tuple arrays (model declarations):

```
<tuple_array_declaration> ::= <NAME> <NAME> '[' <NAME> ']' '=' '...' ';'   // TupleType Arr[Set] = ...;
                            | <NAME> <NAME> '[' <NAME> ']' ';'             // TupleType Arr[Set];
```

Decision expressions (expanded on use):

```
// Scalar dexpr:
<dexpr_declaration> ::= 'dexpr' <type> <NAME> '=' <expression> ';'

// Indexed dexpr with flat header:
                     | 'dexpr' <type> <NAME> <dexpr_index_header> '=' <expression> ';'

// Indexed dexpr with strict nested headers:
                     | 'dexpr' <type> <NAME> <dexpr_index_headers> '=' <expression> ';'

// dexpr index headers:
<dexpr_index_header>  ::= '[' <dexpr_index_list> ']'
<dexpr_index_headers> ::= '[' <dexpr_index_list> ']' ( '[' <dexpr_index_list> ']' )*
<dexpr_index_list>    ::= <dexpr_index_list> ',' <dexpr_index> | <dexpr_index>
<dexpr_index>         ::= <NAME> 'in' <IN_RANGE>
```

Parameters (param keyword optional; external or inline; arrays or expressions):

```
// External scalar or indexed (implicit or explicit with '...'):
<param_declaration> ::= [ 'param' ] <type> <NAME> [ <indexed_dimensions> ] [ <opt_assign_ellipsis> ] ';'

// Inline scalar constant:
                      | [ 'param' ] <type> <NAME> '=' <NUMBER> ';'

// Inline scalar general expression (evaluated at compile-time):
                      | [ 'param' ] <type> <NAME> '=' <expression> ';'

// Inline indexed from array literal (nested lists):
                      | [ 'param' ] <type> <NAME> <indexed_dimensions> '=' <array_value> ';'

// Computed indexed parameter with flat header:
                      | [ 'param' ] <type> <NAME> <dexpr_index_header> '=' <expression> ';'

// Computed indexed parameter with strict nested headers:
                      | [ 'param' ] <type> <NAME> <dexpr_index_headers> '=' <expression> ';'

<opt_assign_ellipsis> ::= '=' '...' | ε
```

### Indexed dimensions and ranges

```
<indexed_dimensions> ::= <indexed_dimensions> '[' <index_specifier> ']'
                       | '[' <index_specifier> ']'

<index_specifier> ::= <expression> '..' <expression>    // range index (int-valued bounds)
                    | <expression>                      // general index: number/name/arithmetic/paren/field/string-literal
                    | <tuple_literal>                   // tuple index into set-of-tuples

<range_expr> ::= <expression>                           // must be integer-valued

// Shared by sum/forall/dexpr:
<IN_RANGE> ::= <expression> '..' <expression> | <NAME>  // NAME may denote a named range or a named set
```

Index expressions accept:
- number literal, name (iterator or parameter), arithmetic `+ - * / %`, unary minus, parentheses
- field access index (e.g., `t.a` if int-typed), normalized internally
- string literal (as index into typed sets)
- tuple literals for tuple-indexed variables/parameters

### Expressions

Precedence from lowest to highest: `? :`, `||`, `&&`, comparisons (`== != <= >= < >`), `+ -`, `* / %`, unary `!/-`, field access `.` (tightest, right-associative).

```
<expression> ::= <conditional>

<conditional> ::= <logic_or>
                | '(' <expression> ')' '?' <expression> ':' <expression>   // condition must be parenthesized

<logic_or> ::= <logic_or> '||' <logic_and> | <logic_and>
<logic_and> ::= <logic_and> '&&' <equality> | <equality>

<equality> ::= <equality> '==' <relational>
             | <equality> '!=' <relational>
             | <relational>

<relational> ::= <relational> '<' <additive>
               | <relational> '>' <additive>
               | <relational> '<=' <additive>
               | <relational> '>=' <additive>
               | <additive>

<additive> ::= <additive> '+' <multiplicative>
             | <additive> '-' <multiplicative>
             | <multiplicative>

<multiplicative> ::= <multiplicative> '*' <unary>
                   | <multiplicative> '/' <unary>
                   | <multiplicative> '%' <unary>
                   | <unary>

<unary> ::= '!' <unary>
          | '-' <unary>                        // not allowed on booleans
          | <primary>

<primary> ::= <NUMBER>
            | <BOOLEAN_LITERAL>
            | <STRING_LITERAL>
            | <NAME>
            | <NAME> <indexed_dimensions>
            | <sum_expression>
            | <min_aggregate>
            | <max_aggregate>
            | <function_call>
            | '(' <expression> ')'
            | <primary> '.' <NAME>             // field access (chained; right-assoc)
```

Functions and aggregates:

```
// Aggregates over indices (expression-level)
<sum_expression> ::= 'sum' <sum_index_header> <nonparen_expression>
                   | 'sum' <sum_index_header> <parenthesized_expression>

<min_aggregate> ::= 'min' <sum_index_header> <nonparen_expression>
                  | 'min' <sum_index_header> <parenthesized_expression>

<max_aggregate> ::= 'max' <sum_index_header> <nonparen_expression>
                  | 'max' <sum_index_header> <parenthesized_expression>

// Function calls
<function_call> ::= 'sqrt' '(' <expression> ')'
                  | 'maxl' '(' <arg_list> ')'
                  | 'minl' '(' <arg_list> ')'

<arg_list> ::= <expression> | <expression> ',' <arg_list>

<nonparen_expression> ::= <primary>
<parenthesized_expression> ::= '(' <expression> ')'
```

Sum/forall headers:

```
<sum_index_header> ::= '(' <sum_index_list> <opt_index_constraint> ')'
<forall_index_header> ::= '(' <sum_index_list> <opt_index_constraint> ')'

<sum_index_list> ::= <sum_index_list> ',' <sum_index> | <sum_index>
<sum_index> ::= <NAME> 'in' <IN_RANGE>

<opt_index_constraint> ::= ':' <expression> | ε
```

### Sets, Tuples, and Arrays (Model)

```
// Tuple literals (nested allowed; <> allowed)
<tuple_literal_list> ::= <tuple_literal_list> ',' <tuple_literal> | <tuple_literal>
<tuple_literal> ::= '<' <tuple_element_list> '>' | '<>'
<tuple_element_list> ::= <tuple_element_list> ',' <tuple_element> | <tuple_element>
<tuple_element> ::= <STRING_LITERAL> | <NUMBER> | <tuple_literal>

// Typed scalar set element lists (model)
<element_list_string> ::= <element_list_string> ',' <STRING_LITERAL> | <STRING_LITERAL>
<element_list_int> ::= <element_list_int> ',' <NUMBER> | <NUMBER>                    // integers only
<element_list_float> ::= <element_list_float> ',' <NUMBER> | <NUMBER>                // coerced to float
<element_list_boolean> ::= <element_list_boolean> ',' <BOOLEAN_LITERAL> | <BOOLEAN_LITERAL>

// Inline arrays for parameters (model) — nested arrays; entries may be number/string/boolean
<array_value> ::= '[' <row_list> ']'
<row_list> ::= <row_list> ',' <scalar_value>
             | <scalar_value>
             | <row_list> ',' <array_value>
             | <array_value>

<scalar_value> ::= <NUMBER> | <STRING_LITERAL> | <BOOLEAN_LITERAL>
```

Important modeling rule:
- In model files, only tuple literals are allowed on the RHS of untyped set assignments. Scalar sets in model files must be declared as typed sets using `{int}`, `{float}`, `{boolean}`, or `{string}`.

### Data File Grammar (.dat)

```
<data_file> ::= <data_declaration_list>
<data_declaration_list> ::= <data_declaration_list> <data_declaration> | <data_declaration>

<data_declaration> ::= 'param' <NAME> '=' <scalar_value> ';'
                     | 'set' <NAME> '=' <set_value> ';'
                     | 'param' <NAME> '=' <array_value> ';'
                     | <NAME> '=' <scalar_value> ';'
                     | <NAME> '=' <set_value> ';'
                     | <NAME> '=' <array_value> ';'
                     | <NAME> '=' <key_value_array> ';'
                     | <NAME> '=' 'param' <key_value_array> ';'
                     | <NAME> '=' 'set' <key_value_array> ';'
                     | <NAME> '=' <NUMBER> '..' <NUMBER> ';'
                     | <set_of_tuples_assignment>

// Set-of-tuples in .dat (untyped or typed)
<set_of_tuples_assignment> ::= <NAME> '=' '{' <tuple_literal_list> '}' ';'
                             | <NAME> '=' '[' <tuple_literal_list> ']' ';'
                             | '{' <NAME> '}' <NAME> '=' '{' <tuple_literal_list> '}' ';'

<key_value_array> ::= '[' <key_value_row_list> ']'
<key_value_row_list> ::= <key_value_row_list> ',' <key_value_row> | <key_value_row>
<key_value_row> ::= <STRING_LITERAL> <scalar_value>
                  | <tuple_literal> <scalar_value>
                  | <STRING_LITERAL> <array_value>     // label with array
                  | <tuple_literal> <array_value>      // tuple label with array

// Allow trailing comma via lexer/permissive parsing

<set_value> ::= '{' <element_list_scalar> '}'
<element_list_scalar> ::= <element_list_scalar> ',' <scalar_value> | <scalar_value>

// Arrays may be nested (same as model)
<array_value> ::= '[' <row_list> ']'
<row_list> ::= <row_list> ',' <scalar_value>
             |  <scalar_value>
             |  <row_list> ',' <array_value>
             |  <array_value>

<scalar_value> ::= <NUMBER> | <STRING_LITERAL> | <BOOLEAN_LITERAL>
```

### Operator Precedence and Associativity

From lowest to highest binding power:
1. Ternary `? :` (right-assoc; condition must be parenthesized)
2. Logical OR `||`
3. Logical AND `&&`
4. Comparisons `==`, `!=`, `<=`, `>=`, `<`, `>` (tokens are non-assoc in precedence; the grammar accepts chained comparisons and parses them left-nested; such chains evaluate as boolean expressions)
5. Add/Sub `+`, `-`
6. Mul/Div/Mod `*`, `/`, `%`
7. Unary NOT `!` and unary minus `-` (right-assoc)
8. Field access `.` (right-assoc; chains like `a.b.c`)

### Notes and Semantics

- Arithmetic and comparisons:
  - Mix of int, float, and boolean is allowed (booleans treated numerically where needed).
  - `%` (modulo) is supported in expressions for data and parameter evaluation (including inline/externally loaded parameters and .dat computations). It is not supported inside linear model parts (objectives or constraints). Using `%` in those contexts will be rejected or fail code generation.
- Boolean expressions:
  - `==`/`!=` are allowed on booleans and numbers/strings; boolean equality to `true` is used to normalize.
  - Logical `&&`, `||`, `!` are available; boolean objectives allowed.
- Aggregates:
  - `sum` supports multi-indices, optional index constraints, and can sum booleans (result type int).
  - Aggregates `min` and `max` are over index sets (expression-level aggregates).
  - Functions `maxl(a1,...,aN)` / `minl(a1,...,aN)` are function forms (lists of arguments); lowered to linear constraints where convex.
- Field access:
  - `a.b` is type-checked against declared tuple types and supports chaining. Tuple iterators carry their tuple type so `i.cost` works inside `sum/forall`.
- Indexing:
  - Range index form `[lo..hi]` requires integer-valued bounds.
  - General index expressions (names, number literals, string literals, arithmetic, parenthesized, or tuple field access with int type) are supported.
  - For set dimensions:
    - If the set is a set of tuples, the index must be of that tuple type (or a tuple literal).
    - If the set is a typed scalar set, the index must match its base type.
- Model typed scalar sets `{string}`, `{int}`, `{float}`, `{boolean}` are validated; floats coerce ints to floats.
- Untyped sets in model: only set-of-tuples assignments with tuple literals are allowed; scalar sets in model must be typed.
- Decision expressions (`dexpr`) are expanded on use (scalar and indexed forms).
- Conditional constraints: `if (cond) { ... } [else { ... }]` — cond must be ground; within `forall`, cond may reference iterators and parameters only.
- String is not a valid decision variable domain. Use string only for:
  - tuple fields,
  - typed scalar sets in models/data,
  - parameter values. Attempting `dvar string ...` is a semantic error.

### Notes on cardinality and reification patterns

The implementation supports common linear encodings involving booleanized comparisons:

- Sums of comparisons (cardinality):
  - Example: `sum(i in I) (x[i] >= 0) >= k`
  - Comparisons inside sums are treated as 0/1 and linearized via standard big-M bounds when necessary.

- Reified comparisons:
  - Example: `b == (x - y >= c)`
  - The boolean `b` is automatically linked to the comparison with big-M constraints based on available bounds.

Notes:
- These encodings are accepted wherever linearization is applicable. Quality of big-M depends on variable bounds; provide tight bounds in declarations to improve relaxations.
- Non-convex placements (equality to minl/maxl or unrestricted boolean arithmetic beyond supported patterns) are rejected.

## Example

Model:
```opl
tuple Edge { string u; string v; int w; }
{Edge} E = { <"A","B",5>, <"B","C",3> };

{string} Cities = { "SEA", "SFO" };
{int}    K = { 1, 2, 3 };
{float}  W = { 1, 2.5 };
{boolean} B = { true, false };

range T = 1..3;

param float c[E][T] = ...;              // external indexed by tuple-set × range
param int    N = 5;                     // scalar
param float alpha = (N - 2) % 3;        // inline expression
param float d[T] = [1, 0, 2];           // inline array
param float sqrt_d[t in T] = sqrt(d[t]); // computed indexed param

dexpr float d2[t in T] = d[t] * 2;      // decision expression, expanded on use

dvar boolean x[E];
dvar float+  y[T];

minimize max( t in T ) ( y[t] );        // aggregate max over T

subject to {
  cap: (sum(t in T) y[t]) >= 1;

  // Implication with boolean expressions on each side
  forall(e in E)
    (x[e] == 1) => (y[1] + y[2] >= 0.5);

  // Conditional with parenthesized condition
  z : ((true) ? y[1] : y[2]) <= 1;

  // If-constraint (ground condition)
  if (N >= 3) { y[1] <= 10; } else { y[1] <= 5; }

  // Boolean in arithmetic are allowed
  y[2] == (true);

  // Use of field access on tuple iterator and tuple array/indexed params
  forall(e in E) x[e] <= (e.w >= 4);

  // Sum body may be boolean; coerced to int
  sum(t in T : t >= 2) (y[t] >= 1) >= 1;

  // minl/maxl function forms (lowered where convex)
  maxl(y[1], y[2], y[3]) <= 4;
}
```

Data:
```opl
// c[e][t] for t in 1..3
c = [
  <"A","B",5> [1.1, 1.2, 1.3],
  <"B","C",3> [2.1, 2.2, 2.3],
];

```

## References

- Source: `pyopl/pyopl_core.py`
- Codegen: `pyopl/gurobi_codegen.py`, `pyopl/scipy_codegen*.py`