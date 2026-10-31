% Legacy MATLAB scheduling routine.
function [info,pop,fit]=operator_18(pop,data,info,fit)
for x = 1:info.tl3
    [pop, fit,  ~, ~] = SSA(pop, info, data, fit);
end
