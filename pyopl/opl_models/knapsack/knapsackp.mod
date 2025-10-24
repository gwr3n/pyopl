range Items = 1..12;
range Resources = 1..7;
int TotalCapacity = ...;
float Value[Items];
float Use[Resources][Items];

dvar boolean Take[Items];
dvar int+ Capacity[Resources];

maximize sum(i in Items) Value[i] * Take[i];

subject to {
    forall( r in Resources )
        sum( i in Items ) 
            Use[r][i] * Take[i] <= Capacity[r];
    
    sum( r in Resources ) Capacity[r] <= TotalCapacity;
}