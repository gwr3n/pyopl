range Items = 1..5;
float weight[1..5] = ...;
float value[1..5] = ...;
float C = 10;

dvar boolean x[1..5];

maximize sum (i in Items) (value[i] * x[i]);

subject to {
    sum (i in Items) (weight[i] * x[i]) <= C;
}