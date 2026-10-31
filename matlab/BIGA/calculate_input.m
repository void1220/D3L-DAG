% Legacy MATLAB scheduling routine.
function [inputFeatures, fit] = calculate_input(pop,info,data,fit,fitold)
if info.mode == 1
    min_fit = [181.51 195.26 220.51 269.26 300 325.92 372.45 402.79 767.32 746.49 2000 2000 2000 2000 2200 2200];
elseif info.mode == 2
    min_fit = [198.48 142.96 343.69 266.47 382.24 274.59 791.44 686.03 919.15 743.23 2000 1700 2400 3000 3000 3000];
end
standardDeviation = std(fit) / 1000;
mean_fit = min_fit(info.num) / mean(fit);
min_fit = min_fit(info.num) / min(fit);
sum_ev = 0;
for i = 1:info.np
    sum_ev = sum_ev + ( fitold(i) - fit(i) );
end
E_V = sqrt( info.np * ( min(fitold) - min(fit) ) ) / ( sum_ev + 1 )  ;
inputFeatures = [standardDeviation, mean_fit,min_fit,E_V];
end
