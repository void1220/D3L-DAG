% Legacy MATLAB scheduling routine.
function [info,pop,fit]=operator_5(pop,data,info,fit)
for x = 1:info.tl3
    Mmax = 10;
    [pop,fit] = SA(pop,data,info, Mmax);
end
