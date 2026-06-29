import unittest

from pyopl.pyopl_core import OPLCompiler

MODEL_CODE = r"""
int numVars = ...;
int numConstraints = ...;
int numIndicators = ...;

range Vars = 1..numVars;
range Constraints = 1..numConstraints;
range Indicators = 1..numIndicators;

float objOffset = ...;
float objCoef[Vars] = ...;
float lb[Vars] = ...;
float ub[Vars] = ...;
int isBinary[Vars] = ...;
int isInteger[Vars] = ...;

tuple MatrixElement {
  int v;
  float val;
};

int numElements = ...;
{MatrixElement} matrix[Constraints] = ...;

float rhs[Constraints] = ...;
string sense[Constraints] = ...;

int numIndicatorElements = ...;
int indicatorBinaryVar[Indicators] = ...;
int indicatorActiveValue[Indicators] = ...;
{MatrixElement} indicatorMatrix[Indicators] = ...;
float indicatorRhs[Indicators] = ...;
string indicatorSense[Indicators] = ...;

{int} BinaryVars = {v | v in Vars: isBinary[v] == 1};
{int} IntegerVars = {v | v in Vars: isInteger[v] == 1};
{int} ContinuousVars = {v | v in Vars: isBinary[v] == 0 && isInteger[v] == 0};

dvar boolean xBinary[BinaryVars];
dvar int xInteger[IntegerVars];
dvar float xContinuous[ContinuousVars];

dexpr float x[v in Vars] =
  sum(b in BinaryVars: b == v) xBinary[b] +
  sum(i in IntegerVars: i == v) xInteger[i] +
  sum(c in ContinuousVars: c == v) xContinuous[c];

minimize objOffset + sum(v in Vars) objCoef[v] * x[v];

subject to {
  forall(v in Vars) {
    lower_bound:
      x[v] >= lb[v];
    upper_bound:
      x[v] <= ub[v];
  }

  forall(c in Constraints) {
    ct_row:
      if (sense[c] == "E")
        sum(e in matrix[c]) e.val * x[e.v] == rhs[c];
      else if (sense[c] == "L")
        sum(e in matrix[c]) e.val * x[e.v] <= rhs[c];
      else
        sum(e in matrix[c]) e.val * x[e.v] >= rhs[c];
  }

  forall(i in Indicators) {
    indicator_row:
      if (indicatorSense[i] == "E")
        (x[indicatorBinaryVar[i]] == indicatorActiveValue[i]) =>
          sum(e in indicatorMatrix[i]) e.val * x[e.v] == indicatorRhs[i];
      else if (indicatorSense[i] == "L")
        (x[indicatorBinaryVar[i]] == indicatorActiveValue[i]) =>
          sum(e in indicatorMatrix[i]) e.val * x[e.v] <= indicatorRhs[i];
      else
        (x[indicatorBinaryVar[i]] == indicatorActiveValue[i]) =>
          sum(e in indicatorMatrix[i]) e.val * x[e.v] >= indicatorRhs[i];
  }
}
"""


DATA_CODE = r"""
numVars = 3;
numConstraints = 2;
numIndicators = 2;

objOffset = 0.0;
objCoef = [3.0, 1.0, 5.0];
lb = [0, 0, 0];
ub = [10.0, 10.0, 1.0];
isBinary = [0, 0, 1];
isInteger = [0, 0, 0];

numElements = 4;
matrix = [
  { <1, 1.0>, <2, 1.0> },
  { <1, 1.0>, <2, -1.0> }
];

rhs = [4.0, 2.0];
sense = ["G", "L"];

numIndicatorElements = 3;
indicatorBinaryVar = [3, 3];
indicatorActiveValue = [1, 0];
indicatorMatrix = [
  { <1, 1.0>, <2, 2.0> },
  { <1, 1.0> }
];
indicatorRhs = [7.0, 2.0];
indicatorSense = ["L", "G"];
"""


class TestGenericIndicatorCompile(unittest.TestCase):
    def test_nested_indexed_indicator_model_compiles_for_both_backends(self):
        for solver in ("scipy", "gurobi"):
            with self.subTest(solver=solver):
                compiler = OPLCompiler(syntax_error_reporting="full")
                _ast, code, _data = compiler.compile_model(MODEL_CODE, DATA_CODE, solver=solver)
                self.assertTrue(code)


if __name__ == "__main__":
    unittest.main()
