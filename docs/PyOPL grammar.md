# PyOPL Grammar (Aligned with implementation)

This document reflects the grammar currently implemented in `pyopl/pyopl_core.py`. It captures the latest features such as logical operators, implication (`=>`), conditional expressions, field access on tuples, typed scalar sets and tuple arrays, and richer .dat file constructs.

Reference: `pyopl/pyopl_core.py`

## Overview

The grammar supports:

- Declarations: decision variables (`dvar`), parameters (`param` optional), ranges, typed scalar sets, tuple types, sets of tuples, tuple arrays, and an untyped set-of-tuples assignment form
- Objectives: `minimize` or `maximize`
- Constraints: standard comparisons, labelled constraints, quantified constraints with `forall` (single or block), and implication constraints using `=>`
- Expressions: arithmetic, logical (`&&`, `||`, `!`), comparisons (`==`, `!=`, `<=`, `>=`, `<`, `>`), conditional (`? :` with parenthesized condition), field access (`.`), indexed names, and summation (`sum`)
- Data files (.dat): scalars, sets, arrays (nested), ranges, sets of tuples, and key-value arrays with string/tuple labels mapping to scalars or arrays

Notes:
- Boolean values may appear in arithmetic and sums; booleans are treated numerically (false=0, true=1) where needed.
- Boolean objectives are allowed.
- `forall` is statement-level (constraints); `sum` is an expression.
- Field access (`a.b`) follows tuple typing metadata and supports chaining.
- In model files, scalar sets must be typed; untyped set assignments in models are only allowed for sets of tuples (tuple literals only).

## BNF Grammar

BNF is simplified for readability. Optional elements are in `[brackets]`. Alternatives use `|`. `*` means zero or more. `ε` denotes empty.

### Model Structure

```
<model> ::= <declarations_opt> <objective_section> <constraints_section>

<declarations_opt> ::= <declaration_list> | ε
<declaration_list> ::= <declaration_list> <declaration> | <declaration>

<objective_section> ::= 'minimize' <expression> ';'
                      |  'maximize' <expression> ';'

<constraints_section> ::= 'subject to' '{' <constraint_list_opt> '}'
<constraint_list_opt> ::= <constraint_list> | ε
<constraint_list> ::= <constraint_list> <constraint> | <constraint>

<constraint> ::= <expression> '=>' <expression> ';'            // implication
               |  <expression> ';'                              // standard
               |  <NAME> ':' <expression> ';'                   // labelled
               |  'forall' <forall_index_header> <constraint>   // single (including labelled or implication)
               |  'forall' <forall_index_header> <constraint_block>

<constraint_block> ::= '{' <constraint_list> '}'
```

### Declarations

```
// Decision variables
<declaration> ::= 'dvar' <type> <NAME> ';'
                |  'dvar' <type> <NAME> <indexed_dimensions> ';'

// Ranges
                |  'range' <NAME> '=' <range_expr> '..' <range_expr> ';'
                |  'range' <NAME> ';'

// Untyped set symbol (no assignment in declaration)
                |  'set' <NAME> ';'

// Typed scalar sets (model)
                |  <typed_set_declaration>

// Set of tuples (typed by a tuple type name)
                |  <set_of_tuples_declaration>

// Untyped set-of-tuples assignment in model (tuple literals only; scalar elements not allowed)
                |  <untyped_tuple_set_assignment>

// Tuple type declarations
                |  <tuple_type_declaration>

// Parameters (param keyword optional)
                |  <param_declaration>

// Tuple arrays
                |  <tuple_array_declaration>

// Types include tuple type names as identifiers
<type> ::= 'int' | 'float' | 'int+' | 'float+' | 'boolean' | 'string' | <NAME>
```

Typed scalar sets in models (all variants are supported):
```
// Strings
<typed_set_declaration> ::= '{' 'string' '}' <NAME> '=' '{' <element_list_string> '}' ';'
                          |  '{' 'string' '}' <NAME> ';'
                          |  '{' 'string' '}' <NAME> '=' '...' ';'

// Integers
                          |  '{' 'int' '}' <NAME> '=' '{' <element_list_int> '}' ';'
                          |  '{' 'int' '}' <NAME> ';'
                          |  '{' 'int' '}' <NAME> '=' '...' ';'

// Floats (ints permitted; coerced to float)
                          |  '{' 'float' '}' <NAME> '=' '{' <element_list_float> '}' ';'
                          |  '{' 'float' '}' <NAME> ';'
                          |  '{' 'float' '}' <NAME> '=' '...' ';'

// Booleans
                          |  '{' 'boolean' '}' <NAME> '=' '{' <element_list_boolean> '}' ';'
                          |  '{' 'boolean' '}' <NAME> ';'
                          |  '{' 'boolean' '}' <NAME> '=' '...' ';'
```

Set of tuples in models:
```
// RHS must be tuple literals only (guard rejects scalar elements)
<set_of_tuples_declaration> ::= '{' <NAME> '}' <NAME> '=' '{' <tuple_literal_list> '}' ';'
                              |  '{' <NAME> '}' <NAME> ';'
                              |  '{' <NAME> '}' <NAME> '=' '...' ';'

// Untyped set-of-tuples assignment allowed in model (tuple literals only)
<untyped_tuple_set_assignment> ::= <NAME> '=' '{' <tuple_literal_list> '}' ';'
```

Tuple types:
```
<tuple_type_declaration> ::= 'tuple' <NAME> '{' <tuple_field_list> '}' [';']
                           |  'tuple' <NAME> '{' '}' [';']
<tuple_field_list> ::= <tuple_field_list> <tuple_field> | <tuple_field>
<tuple_field> ::= <type> <NAME> ';'
```

Parameters:
```
<param_declaration> ::= [ 'param' ] <type> <NAME> [ <indexed_dimensions> ] [ <opt_assign_ellipsis> ] ';'
                      |  [ 'param' ] <type> <NAME> '=' <NUMBER> ';'
                      |  [ 'param' ] <type> <NAME> <indexed_dimensions> '=' <array_value> ';'

<opt_assign_ellipsis> ::= '=' '...'
                        |  ε
```

Tuple arrays:
```
<tuple_array_declaration> ::= <NAME> <NAME> '[' <NAME> ']' '=' '...' ';'
                            |  <NAME> <NAME> '[' <NAME> ']' ';'
```

### Indexed dimensions and ranges

```
<indexed_dimensions> ::= <indexed_dimensions> '[' <index_specifier> ']'
                       |  '[' <index_specifier> ']'

<index_specifier> ::= <expression> '..' <expression>   // range index (int-valued bounds)
                    |  <expression>                    // number literal, name, arithmetic, unary, parens, or field access (if int-typed)
                    |  <field_access>                  // allowed as integer index (normalized internally)

<range_expr> ::= <expression>                          // must be integer-valued
```

### Expressions

Precedence from lowest to highest: `? :`, `||`, `&&`, comparisons (`==`, `!=`, `<=`, `>=`, `<`, `>`), `+ -`, `* /`, unary `!`, field access `.` (tightest, right-associative).

```
<expression> ::= <conditional>

<conditional> ::= <logic_or>
                |  '(' <expression> ')' '?' <expression> ':' <expression>  // condition must be parenthesized

<logic_or> ::= <logic_or> '||' <logic_and> | <logic_and>
<logic_and> ::= <logic_and> '&&' <equality> | <equality>

<equality> ::= <equality> '==' <relational>
             |  <equality> '!=' <relational>
             |  <relational>

<relational> ::= <relational> '<' <additive>
               |  <relational> '>' <additive>
               |  <relational> '<=' <additive>
               |  <relational> '>=' <additive>
               |  <additive>

<additive> ::= <additive> '+' <multiplicative>
             |  <additive> '-' <multiplicative>
             |  <multiplicative>

<multiplicative> ::= <multiplicative> '*' <unary>
                   |  <multiplicative> '/' <unary>
                   |  <unary>

<unary> ::= '!' <unary>
          |  '-' <unary>                  // not allowed on booleans
          |  <primary>

<primary> ::= <NUMBER>
            |  <BOOLEAN_LITERAL>
            |  <STRING_LITERAL>
            |  <NAME>
            |  <NAME> <indexed_dimensions>
            |  <sum_expression>
            |  '(' <expression> ')'
            |  <primary> '.' <NAME>       // field access (chained; right-associative)

<sum_expression> ::= 'sum' <sum_index_header> <nonparen_expression>
                   |  'sum' <sum_index_header> <parenthesized_expression>

<nonparen_expression> ::= <primary>
<parenthesized_expression> ::= '(' <expression> ')'
```

Sum/forall headers:
```
<sum_index_header> ::= '(' <sum_index_list> <opt_index_constraint> ')'
<forall_index_header> ::= '(' <sum_index_list> <opt_index_constraint> ')'

<sum_index_list> ::= <sum_index_list> ',' <sum_index> | <sum_index>
<sum_index> ::= <NAME> 'in' <IN_RANGE>

<IN_RANGE> ::= <expression> '..' <expression> | <NAME>   // NAME may denote a named range or a named set

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

// Inline arrays for parameters (model) — nested arrays, entries may be number/string/boolean
<array_value> ::= '[' <row_list> ']'
<row_list> ::= <row_list> ',' <scalar_value>
             |  <scalar_value>
             |  <row_list> ',' <array_value>
             |  <array_value>

<scalar_value> ::= <NUMBER> | <STRING_LITERAL> | <BOOLEAN_LITERAL>
```

Important modeling rule:
- In model files, only tuple literals are allowed on the RHS of untyped set assignments. Scalar sets in model files must be declared as typed sets using `{int}`, `{float}`, `{boolean}`, or `{string}`.

### Data File Grammar (.dat)

```
<data_file> ::= <data_declaration_list>
<data_declaration_list> ::= <data_declaration_list> <data_declaration> | <data_declaration>

<data_declaration> ::= 'param' <NAME> '=' <scalar_value> ';'
                     |  'set' <NAME> '=' <set_value> ';'
                     |  'param' <NAME> '=' <array_value> ';'
                     |  <NAME> '=' <scalar_value> ';'
                     |  <NAME> '=' <set_value> ';'
                     |  <NAME> '=' <array_value> ';'
                     |  <NAME> '=' <key_value_array> ';'
                     |  <NAME> '=' 'param' <key_value_array> ';'
                     |  <NAME> '=' 'set' <key_value_array> ';'
                     |  <NAME> '=' <NUMBER> '..' <NUMBER> ';'
                     |  <set_of_tuples_assignment>

<set_of_tuples_assignment> ::= <NAME> '=' '{' <tuple_literal_list> '}' ';'
                             |  <NAME> '=' '[' <tuple_literal_list> ']' ';'
                             |  '{' <NAME> '}' <NAME> '=' '{' <tuple_literal_list> '}' ';'

<key_value_array> ::= '[' <key_value_row_list> ']'
<key_value_row_list> ::= <key_value_row_list> ',' <key_value_row> | <key_value_row>
<key_value_row> ::= <STRING_LITERAL> <scalar_value>
                  |  <tuple_literal> <scalar_value>
                  |  <STRING_LITERAL> <array_value>     // label with array
                  |  <tuple_literal> <array_value>      // tuple label with array

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
6. Mul/Div `*`, `/`
7. Unary NOT `!` (right-assoc)
8. Field access `.` (right-assoc; chains like `a.b.c`)

### Notes and Semantics

- Implication constraints (`A => B`) accept either comparisons/constraints or general boolean expressions on each side; boolean expressions are normalized to equality with `true`.
- `forall` generates one or more constraints. Labels are allowed both on standalone constraints and inside `forall`, e.g., `forall(i in I) cap: x[i] <= 1;`.
- `sum` supports multi-indices, optional index constraints, and can sum booleans (result type is integer).
- Boolean literals and variables are allowed in arithmetic; result types follow standard numeric promotion (float if any float involved, otherwise int).
- Field access (`a.b`) is type-checked against declared tuple types and supports chaining.
- Indexing:
  - Range index form `[lo..hi]` requires integer-valued bounds.
  - General index expressions (names, number literals, arithmetic, parenthesized, or tuple field access with int type) are supported.
  - For set dimensions:
    - If the set is a set of tuples, the index must be of that tuple type (or a tuple literal).
    - If the set is a typed scalar set, the index must match its base type.
    - If the set is untyped (in data), string or integer indices are accepted.
- Model typed scalar sets `{string}`, `{int}`, `{float}`, `{boolean}` are validated for element types; floats coerce ints to floats.
- In model files, untyped scalar set assignments are not allowed; use typed set declarations.

## Examples

```opl
tuple Edge { string u; string v; }
{Edge} E = { <"A","B">, <"B","C"> };

// Untyped set-of-tuples assignment in model (allowed; tuple literals only)
ArcPairs = { <"A","B">, <"C","D"> };

{string} Cities = { "SEA", "SFO" };
{int}    K = { 1, 2, 3 };
{float}  W = { 1, 2.5 };       // ints allowed; coerced to float
{boolean} B = { true, false };

range T = 1..3;
param float c[E][T] = ...;     // external
param float a[T] = [1, 0, 2, 3, 0];

dvar boolean x[E];
dvar float+  y[T];

minimize sum(e in E, t in T : t >= 2) (c[e][t] * x[e]);

subject to {
  cap: (sum(t in T) y[t]) >= 1;

  // Implication with boolean expressions on each side
  forall(e in E)
    (x[e] == 1) => (y[1] + y[2] >= 0.5);

  // Conditional with parenthesized condition
  z : ((true) ? y[1] : y[2]) <= 1;

  // Chained comparisons parse left-nested and evaluate as boolean
  w: (y[1] < y[2] < y[3]);
}
```

## References

- Source: `pyopl/pyopl_core.py`