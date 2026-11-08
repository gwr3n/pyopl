int StopCount = ...;
range Stops = 1..StopCount;
float fare[Stops][Stops] = ...;

dvar boolean link[Stops][Stops];
dvar float+ cargo[Stops][Stops];

minimize serviceCost:
    sum(i in Stops, j in Stops: i != j) fare[i][j] * link[i][j];

subject to {
    forall(i in Stops)
        departOnce: sum(j in Stops: j != i) link[i][j] == 1;
    forall(j in Stops)
        arriveOnce: sum(i in Stops: i != j) link[i][j] == 1;
    rootDispatch:
        sum(j in Stops: j != 1) cargo[1][j]
          - sum(i in Stops: i != 1) cargo[i][1] == StopCount - 1;
    forall(i in 2..StopCount)
        delivery:
            sum(j in Stops: j != i) cargo[j][i]
              - sum(j in Stops: j != i) cargo[i][j] == 1;
    forall(j in Stops)
        depotCapacity: cargo[1][j] <= (StopCount - 1) * link[1][j];
    forall(i in 2..StopCount, j in Stops: i != j)
        routeCapacity: cargo[i][j] <= (StopCount - 2) * link[i][j];
}