# PyOPL User Guide

## Table of Contents
- [Core Concepts](#core-concepts)
- [1. Declarations](#1-declarations)
  - [Decision Variables (`dvar`)](#decision-variables-dvar)
  - [Ranges (`range`)](#ranges-range)
  - [Parameters (`param`)](#parameters-param)
  - [Sets (`set`)](#sets-set)
  - [Tuple Types and Sets of Tuples](#tuple-types-and-sets-of-tuples)
  - [Tuple arrays](#tuple-arrays)
  - [Data Input (`.dat` files)](#data-input-dat-files)
- [2. Objective Section](#2-objective-section)
- [3. Constraints Section](#3-constraints-section)
  - [Simple Constraints](#simple-constraints)
  - [Boolean logic in constraints (and/or/not)](#boolean-logic-in-constraints-andornot)
  - [Cardinality over comparisons](#cardinality-over-comparisons)
  - [Reified cardinality](#reified-cardinality-boolean-equality-to-a-cardinality-condition)
  - [Implication Constraints (`=>`)](#implication-constraints-=>)
  - [Conditional Expressions](#conditional-expressions)
  - [Boolean Objectives and Constraints](#boolean-objectives-and-constraints)
  - [Field Access](#field-access)
  - [Multi-indexed and Tuple-indexed Constraints](#multi-indexed-and-tuple-indexed-constraints)
  - [`forall` Constraints](#forall-constraints)
  - [`sum` Expressions](#sum-expressions)
- [Expressions](#expressions)
- [Comments](#comments)
- [Example Models](#example-models)
- [Error Handling](#error-handling)
- [Solving a Model](#solving-a-model)
- [Limitations](#limitations)
- [PyOPL IDE](#pyopl-ide)
  - [Launching the IDE](#launching-the-ide)

This guide describes the syntax and features of the Optimisation Programming Language (OPL) as implemented in PyOPL. PyOPL is a Python library and IDE for defining and solving optimization problems. The PyOPL compiler translates OPL models into code for use with either the Gurobi Optimizer or the open-source SciPy/HiGHS solver. You can choose which solver to use. For SciPy/HiGHS, integrality is passed to `linprog` if present, but full MIP support depends on your SciPy version and solver. PyOPL now provides robust support for tuple types, sets of tuples, tuple field access, multi-indexed variables, advanced sum/forall constructs, and improved semantic error handling.

## Core Concepts

An OPL model consists of three main sections:
1. **Declarations**: Define decision variables, parameters, sets, ranges, and tuple types/sets of tuples.
2. **Objective**: Specify the function to minimize or maximize (supports sum/forall, tuple field access).
3. **Constraints**: Define the conditions that must be satisfied (supports multi-indexed, tuple-indexed, advanced constructs).

---

## 1. Declarations

### Decision Variables (`dvar`)

Decision variables are the unknowns to be determined by the optimizer. They can be scalar, multi-indexed, or tuple-indexed, and support non-negative types and multi-dimensional indexing.

- **Scalar Decision Variable:**
    ```opl
    dvar float x;
    dvar int y;
    dvar boolean z;
    dvar int+ yplus;      // non-negative integer
    dvar float+ xpos;     // non-negative float
    ```
    Declares `x` as a continuous variable, `y` as an integer variable, `z` as a binary variable, and `yplus`, `xpos` as non-negative variables.

- **Indexed Decision Variable:**
    ```opl
    dvar float flow[1..2][1..3];
    dvar boolean assign[1..5];
    dvar int x[i in Items, j in Cities];
    dvar float y[i in 1..N][j in 1..M];
    dvar float x[arcs]; // tuple-indexed
    ```
    Declares `flow` as a 2D array, `assign` as a 1D array, and shows use of named ranges/sets and multi-indexing.

- New: Index expressions in variable/parameter indexing can be integer expressions (e.g., t-1, (i+j), -k) and tuple field access, provided the expression is integer-valued by type inference.

- New: Tuple arrays are supported as first-class data (see “Tuple arrays” below).

### Ranges (`range`)
Ranges define integer sequences for indexing and iteration. Bounds can be integer expressions or parameter names.
```opl
range Items = 1..5;
range MyRange = 10..(N-1);
```
Important:
- Named ranges must be declared in the model with explicit bounds (inline). Model-time “external ranges” declared as `range T;` are parsed but not supported by the code generators at loop sites. Always provide bounds in the model (e.g., `range T = 1..N;`). Data files can still provide scalar parameters N used in range expressions.

### Parameters (`param`)
Parameters are known values provided to the model. They can be scalar or indexed, and can be declared as external (value from `.dat`) either implicitly or explicitly.

- **Scalar Parameter:**
    ```opl
    param float C;           // external, value from .dat
    param int num_items = ...; // explicit external
    float alpha = 5.0;       // inline value
    int+ n = ...;            // non-negative, external
    ```
- **Indexed Parameter:**
    ```opl
    param float weight[1..5];
    param float value[1..5];
    param float costs[1..2][1..3];
    param float d[i in Items, j in Cities] = ...; // explicit external
    ```
    Use `= ...;` for explicit external parameters (value must be provided in `.dat`).

- New: Tuple-indexed, set×range, and set×set parameters are supported. You can use:
  - Inline model lists, or
  - .dat key-value dictionaries with tuple or string keys, or
  - Row-major list-of-rows for 2D parameters when set order is known.

### Sets (`set`)
Sets are collections of elements. You can declare a set (to be assigned in a `.dat` file or later), or define its contents directly in the model:
```opl
set MySet; // Declaration only (data provided in .dat file)
set Cities = {"A", "B", "C"}; // Set assignment with explicit values
```
Sets can be used as indices for variables and parameters. When declared without assignment, the set values must be provided in a `.dat` file. When assigned in the model, the set is immediately available.

- New: Typed scalar sets (e.g., strings) are supported:
```opl
{string} Cities = { "A", "B", "C" };
{string} Warehouses;         // external typed set
{string} Products = ...;     // external typed set via .dat
```
Typed scalar sets can be used as indices for variables/parameters; parameters indexed by such sets can be supplied in .dat as:
- A list (row-major) in set order, or
- A dict keyed by element labels.

Note:
- Code generation emits a helper index map `<SetName>_index` for typed scalar sets to support list-backed parameters internally. This is not part of the OPL syntax, but may appear in generated code.

### Tuple Types and Sets of Tuples

PyOPL provides robust, first-class support for tuple types, sets of tuples, and tuple field access throughout all model constructs. **Nested tuple types and sets of nested tuples are fully supported, including as indices for variables and parameters, in both models and data files.**

```opl
dvar float x[arcs];
dvar float y[nested];
param float w[arcs] = [1.5, 2.5];
```
- Use dot access for fields, e.g. `a.cost` (where `a` is a tuple of type `Arc`) and `o.pair.i` (where `o` is a nested tuple).
- Positional tuple-field access like `a[2]` is not part of the OPL syntax; use dot access.

#### Basic Tuple Example
```opl
tuple Arc { string start; string end; float cost; }
{Arc} arcs = { <"A", "B", 10.0>, <"B", "C", 12.5> };
dvar float x[arcs];
param float w[arcs] = [1.5, 2.5];
```

### Tuple arrays
Tuple arrays (arrays of tuples keyed by a set) are supported as data-only constructs in the model and .dat.
```opl
tuple Arc { string start; string end; float cost; }
{Arc} arcs = { <"A","B",10.0>, <"B","C",12.5> };

Arc Arr[arcs];     // declared in model
Arc Arr[arcs] = ...; // external data in .dat

// Access fields
Arr[a].cost      // dot access
Arr[a]['cost']   // dict-style access (backend emits dicts)
```
Notes:
- In codegen, a tuple array is made available as a dict keyed by the index set element mapping to a record-like dict {field: value}, so `Arr[a].cost` and `Arr[a]['cost']` both work.
- Positional tuple indexing (e.g., `a[2]`) is not supported in OPL; always use dot access.

### Data Input (`.dat` files)

Parameters, sets, and ranges can be assigned values in a separate `.dat` file. Supported types include numbers, strings, booleans, lists, nested lists, sets, and range data.

**Example `mydata.dat`:**
```opl
my_param = 50;
my_set = {1, 2, 3};
my_array = [10, 20];
my_2d_array = [[1, 2], [3, 4]];
Items = 1..5; // range data
```

Additional supported forms:
- Range data (integer) assignment:
```opl
Items = 1..5;            // creates a range_data entry (used for validation/data, not for loops)
```
- Key-value arrays supporting string or tuple keys, mapping to scalar or array values:
```opl
v = [
  <"A","B",10.0>  1.5,
  <"B","C",12.5>  2.5
];

Demand = [
  "StoreA" [10, 12, 8],
  "StoreB" [ 9, 11, 7]
];
```
- 2D parameters indexed by set×range or set×set can be given as:
  - Dict-of-lists keyed by the first index, or
  - Row-major list-of-rows if the set order is known from the model or data.

Backends normalize these shapes internally for robust lookup by labels or tuple keys.

### Nested Tuples, Sets, and Parameter Indexing

PyOPL supports tuple types, including nested, singleton, and empty tuples, as first-class objects for sets, data, and parameter indexing. This section uses a single example to illustrate all concepts step by step.

#### 1. Declaring Nested Tuple Types and Sets

```opl
tuple Inner { int i; int j; }
tuple Outer { Inner pair; float value; }
{Outer} items = { <<1,2>, 3.5>, <<2,3>, 4.0> };
float v[items] = ...; // From .dat file
```
- Here, `Inner` and `Outer` are tuple types; `items` is a set of `Outer` tuples, each containing an `Inner` tuple and a float.
- Empty tuple literals `< >` and singleton tuples (e.g., `<1>`) are supported in sets and as tuple fields.

#### 2. Providing Data in a `.dat` File

You can assign values to sets and parameters externally in a `.dat` file using either list or key-value style.

**List assignment (order must match set):**
```opl
items = { <<1,2>, 3.5>, <<2,3>, 4.0> };
v = [10.0, 20.0];
```

**Key-value assignment (order-independent, recommended for tuple keys):**
```opl
v = [
    <<1,2>, 3.5>   10.0,
    <<2,3>, 4.0>   20.0
];
```
- Both forms are supported for all parameter types indexed by sets (including nested tuples).
- Key-value assignment is especially useful for sparse or tuple-indexed data.

#### 3. Declaring and Using Parameters Indexed by Nested Tuple Sets

```opl
float v[items] = ...; // Value provided in .dat file
dvar float x[items];        // Decision variable indexed by set of nested tuples
```
- Parameters and variables can be indexed by any set of tuples, including nested, singleton, or empty tuples.
- Tuple field access is supported everywhere: use dot notation or integer index, including for nested fields (e.g., `o.pair.i`, `o.value`).

**Example usage in constraints/objective:**
```opl
forall(o in items)
    x[o] >= v[o] + o.value;
minimize sum(o in items) o.pair.i * x[o];
```

#### Notes
- All tuple literal forms (nested, singleton, empty) are supported in both model and data files.
- This mechanism works for any level of tuple nesting.

---

## 2. Objective Section

The objective defines the function to optimize. Only one objective is allowed.

- **Maximize:**
    ```opl
    maximize x + y;
    ```

- Boolean objectives are allowed; True maps to 1, False to 0.
- Note: Backends expect linear objectives. To use boolean/comparison logic in an objective, linearize via sums of reified comparisons or explicit binaries. For example:
  - Preferred: `minimize sum(i in I) (x[i] >= 1);` (compiled via auxiliary binaries)
  - Avoid raw comparison as the entire objective (e.g., `minimize (x > 0);`) with Gurobi; instead reify or sum indicators.

---

## 3. Constraints Section

PyOPL supports rich boolean and arithmetic composition beyond basic comparisons.

- Linear equalities/inequalities
- Not-equal (`!=`) for both boolean and numeric
- Boolean logic: and/or/not trees, including mixed with comparisons
- Implication constraints (=>)
- Cardinality constraints (sum of comparisons)
- Reified constraints and reified cardinality
- Multi-indexed, tuple-indexed constraints; tuple field access in expressions
- Conditional (ternary) expressions with ground (non-dvar) condition

### Simple Constraints:
```opl
subject to {
    x <= 10;
    y >= 5;
    x + y == 20;
    x != y;
}
```
Supported operators: `<=`, `>=`, `==`, `!=`, `<`, `>`
- `!=` between two boolean variables is compiled as XOR: `a != b` → `a + b == 1`.
- `!=` between numeric (or mixed numeric/boolean coerced) expressions uses a disjunctive big-M formulation with automatically tightened M based on inferred bounds and declarations; a conservative fallback is used if unknown.

### Boolean logic in constraints (and/or/not)
Note: Use operators `&&`, `||`, and `!` in models. The textual words `and`, `or`, `not` are not recognized by the parser.

```opl
subject to {
  // All must hold:
  (x1 == 1 && x2 == 0 && y - z <= 3) == true;

  // At least one must hold:
  (a + b >= 2 || c == 1) == true;

  // Negation
  !(p == q) == true;
}
```
- Backends introduce auxiliary binaries and linear linking constraints to respect these semantics.

### Cardinality over comparisons
Sum of comparisons can be used directly:
```opl
subject to {
  // At least K of these comparisons hold:
  sum(t in 1..T) (demand[t] >= threshold) >= K;

  // Exactly K hold:
  sum(i in I) (x[i] <= 0) == K;
}
```
Pattern is recognized and compiled using auxiliary binaries for each comparison term and a cardinality inequality/equality across them.

### Reified cardinality (boolean equality to a cardinality condition)
You can bind a boolean variable to a cardinality threshold:
```opl
subject to {
  b == (sum(i in I) (x[i] >= 1) >= 2);
}
```
This compiles to a standard cardinality reification with tight linking.

### Implication Constraints (`=>`)
Use the implication operator to model logical relationships:
```opl
subject to {
  (x > 0) => (y == 1);
  (a + b >= z) => (u - v <= 5);
  ((p == 1 && q == 1) == true) => (r <= 10);  // composite antecedent (Gurobi); SciPy: see limits
}
```
Solver notes:
- Gurobi backend:
  - Uses indicator constraints automatically for patterns like `(bin == 1) => (linear constraint)`, and specialized contrapositive forms like `(linear >= c) => (bin == 1)`.
  - Falls back to a big-M encoding with a binary flag when an indicator is not applicable; big-M is tightened using cheap bound analysis.
- SciPy/HiGHS backend:
  - Supports linear antecedents and consequents with big-M encoding and automatic tightening.
  - Composite boolean antecedents and reified forms are supported via auxiliary binaries; prefer a single linear comparison or `bin == 1` for robustness and performance.

### Conditional Expressions: 
Use conditional (ternary) expressions in objectives and constraints:
```opl
minimize (x > 0) ? x : 0;
subject to {
    y == (z > 5) ? 1 : 0;
}
```
Only ground (non-dvar) conditions are allowed in conditional expressions.

### Boolean Objectives and Constraints
You can use boolean-valued expressions in objectives and constraints. For example:
```opl
maximize (x > 0);
subject to {
    (x == 1);
}
```
Boolean objectives are interpreted as integer (1 for true, 0 for false).

### Boolean Expression Trees in Constraints
- Gurobi: complex boolean formulas (and/or/not) are compiled to auxiliary binaries with tight linking and can be combined with implications.
- SciPy: supports boolean comparisons and compositions; compiles to linear constraints with auxiliary binaries. Some complex antecedent forms under implications may be restricted (see above).

### Field Access
Use dot notation to access tuple fields in constraints (including nested fields):
```opl
subject to {
    forall(a in arcs)
        x[a] >= a.cost;
    forall(o in nested)
        y[o] >= o.value;
    forall(o in nested)
        y[o] >= o.pair.i;
}
```

### Multi-indexed and Tuple-indexed Constraints
Constraints can be indexed over multiple ranges, sets, or sets of tuples, including nested tuples:
```opl
subject to {
    forall(i in Items, j in Cities: i != j)
        x[i] + d[i, j] >= 0;
    forall(a in arcs)
        x[a] >= w[a];
    forall(o in nested)
        y[o] >= o.value;
}
```

### `forall` Constraints
Apply a constraint over a range, set, or set of tuples, with support for multiple iterators and index constraints.
```opl
subject to {
    forall (i in Items)
        x[i] <= 100;
    forall (i in Items, j in Items: i != j)
        x[i] + x[j] <= 1;
    forall (a in arcs)
        x[a] >= w[a];
}
```

### `sum` Expressions
Summation over a range, set, or set of tuples (including nested tuples), with support for multiple iterators and index constraints.
```opl
minimize sum (i in Items) (cost[i] * x[i]);
minimize sum (i in Items, j in Items: i != j) (cost[i][j] * x[i][j]);
minimize sum (a in arcs) (a.cost * x[a]);
minimize sum (o in nested) (o.value * y[o]);
```
- Tuple field access is supported in the sum/forall body, objectives, and constraints.

---

## Expressions

Expressions can include:
- **Numbers:** `10`, `3.14`, `1e-3`
- **Variable Names:** `x`, `my_param`
- **Indexed Variables:** `flow[i][j]`, `weight[k]`, `x[i in Items, j in Cities]`, `x[a]` where `a` is a tuple, `y[o]` where `o` is a nested tuple
- **Operators:** `+`, `-`, `*`, `/`, `==`, `!=`, `<=`, `>=`, `<`, `>`
- **Unary Minus:** `-x`
- **Parentheses:** `(x + y)`
- **Boolean values:** `true`, `false` (converted to `1` or `0` in arithmetic)
- **Tuple Field Access:** Use dot notation to access tuple fields in expressions.
    - `a.cost` (where `a` is a tuple of type `Arc`)
    - `o.pair.i` (nested tuple field access)
    - Supported in sum/forall, objectives, and constraints.

Additional notes:
- Index expressions can be arithmetic or field accesses as long as they are integer-typed:
  - `x[t-1]`, `s[(i+j)]`, `y[-k]`, `cost[a.cost]` (if field is int).
- Tuple arrays and tuple fields are accessible everywhere:
  - `Arr[a].value`, `Arr[a]['value']`, `o.pair.i`.

---

## Comments


OPL supports several comment styles:
- Single-line: `// comment` or `# comment`
- Multi-line:
    ```opl
    /* multi-line
       comment */
    ```

---

## Example Models

### 1) Typed set indexing and tuple-indexed variable

A compact transportation-style example that shows typed scalar sets and a small tuple type. It demonstrates declaring a set of arcs (tuple-indexed), supplying tuple-indexed parameters, creating tuple-indexed decision variables, using tuple field access in the objective and constraints, and minimizing a linear cost over the tuple index.

```opl
{string} Cities = {"A","B","C"};

tuple Arc { string s; string t; float cost; }
{Arc} arcs = { <"A","B",10.0>, <"B","C",12.5> };

dvar float+ x[arcs];      // tuple-indexed decision variables
param float w[arcs] = [1.5, 2.5];

minimize sum(a in arcs) (a.cost * x[a]);
subject to {
  forall(a in arcs) x[a] >= w[a];
}
```

### 2) Tuple arrays and field access

This example shows how to declare a tuple array (records indexed by a set) and how to reference tuple fields in constraints. The tuple array acts like a small database of records (Order), indexed by a set of Orders; the model uses O[o].demand to build constraints that reference record fields.

```opl
tuple Order { int id; float demand; }
{Order} Orders = { <1, 10.0>, <2, 8.0> };

Order O[Orders] = ...;   // from .dat: records map
dvar float+ ship[Orders];

minimize sum(o in Orders) ship[o];
subject to {
  forall(o in Orders) ship[o] >= O[o].demand;
}
```

### 3) Cardinality and reification

A small combinatorial example showing boolean decision variables over a typed set, direct summation of boolean terms to express cardinality constraints (e.g., at least K selected), and reification of a boolean variable to a cardinality condition (b is true iff at least 3 items selected).

```opl
{string} Items = {"A","B","C","D"};
dvar boolean y[Items];

// At least two items selected
subject to {
  sum(i in Items) (y[i] == 1) >= 2;
}

// Reified: b is true iff at least 3 items selected
dvar boolean b;
subject to {
  b == (sum(i in Items) (y[i] == 1) >= 3);
}
```

### 4) Implications

A tiny example that illustrates implication constraints (antecedent => consequent). It demonstrates common indicator-like patterns where a continuous variable's activation implies a boolean variable (or vice versa). Backends encode these using indicator constraints when supported or via tightened big‑M otherwise.

```opl
dvar float+ x;
dvar boolean z;

subject to {
  (x >= 5) => (z == 1);    // if x active then z on
  (z == 1) => (x <= 20);   // indicator pattern
}
```

---

## Error Handling

PyOPL provides robust semantic error messages for undeclared symbols, type mismatches, illegal operations, and more. Errors include line numbers when available, and diagnostics are shown in both the API and IDE output.

---

## Solving a Model


To solve a model, use the `solve` function:
```python
from pyopl import solve
results = solve('model.mod', 'data.dat')
print(results)
```


By default, PyOPL uses Gurobi as the solver. You can also use the open-source SciPy/HiGHS solver for linear programs and, if supported by your SciPy version, mixed-integer programming (MIP, i.e., integer and boolean variables) by specifying the `solver` argument. Both solvers are selectable in the API and the IDE.

```python
results = solve('model.mod', 'data.dat', solver='gurobi')  # Use Gurobi (default)
results = solve('model.mod', 'data.dat', solver='scipy')   # Use SciPy/HiGHS (LP or MIP, if supported)
```

- `solver='gurobi'` (default): Uses Gurobi for linear and mixed-integer models (requires Gurobi license).
- `solver='scipy'`: Uses SciPy's HiGHS solver for linear programming (LP) and, if supported by your SciPy version, mixed-integer programming (MIP, i.e., integer and boolean variables) models. Integrality is passed to `linprog` if present, but full MIP support depends on your SciPy installation.

The `solve` function returns a dictionary with the following keys:
- `status`: Optimization status (e.g., 'OPTIMAL', 'INFEASIBLE')
- `solution`: Variable values (if optimal)
- `objective_value`: Objective value (if optimal)
- `stats`: Solver statistics (MIPGap, Runtime, etc.)
- `message`: Error or status messages

Gurobi or SciPy/HiGHS output will be printed, including variable values and objective value if optimal.

Solver specifics:
- Gurobi (default): linear and mixed-integer models; uses indicator constraints for many logical patterns and big-M encodings with automatic tightening.
- SciPy/HiGHS: linear programs and (if supported by your SciPy version) MIP. Integrality is passed to `linprog`; boolean/logic and implications are compiled via big-M with automatic tightening; some composite boolean antecedents for implications may be limited.

---

## Limitations
- Named ranges must be declared inline in the model with explicit bounds (e.g., `range T = 1..N;`). A bare `range T;` is parsed but not supported by code generation at loop sites.
- SciPy/HiGHS backend:
  - Composite boolean implication antecedents are supported via big‑M and auxiliaries; prefer simple forms for robustness.
  - Capabilities for larger MIP models depend on your SciPy/HiGHS version.
- Non-linear arithmetic (e.g., variable*variable), piecewise linear, SOS, `<=>` bi-implication, global constraints, and general user-defined functions are not supported.
- Big-M tightening uses declared types, simple expression spans, and collected bounds; when information is insufficient, conservative fallback M is used.

---

## PyOPL IDE

PyOPL includes a graphical IDE for editing, running, and debugging OPL models and data files. The IDE features:

- Syntax highlighting for OPL models and data files
- Side-by-side model and data editors
- Output panel for solver results, errors, and messages
- File tree for easy switching between model and data
- Solver selection (Gurobi or SciPy/HiGHS) — choose your preferred solver from the menu
- Font size adjustment and modern UI

### Launching the IDE

You can launch the IDE from the command line or if installed as a package:

```sh
python -m pyopl
```

This will open the PyOPL IDE window. You can open `.mod` (model) and `.dat` (data) files, edit them, and run your model directly from the interface. You can select either Gurobi or SciPy/HiGHS as the solver from the IDE's menu bar. The IDE provides syntax highlighting, error diagnostics, and a modern UI for rapid prototyping and learning.

---


