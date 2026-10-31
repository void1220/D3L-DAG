% Legacy MATLAB scheduling routine.
function [info,pop,fit]=operator_7(pop,data,info,fit)
for x = 1:info.tl3
    Max_iter = 1 * 1000;
    [pop, fit] = RIME(pop,data,info, Max_iter);
end
