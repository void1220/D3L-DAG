% Legacy MATLAB scheduling routine.
function [info,pop,fit]=operator_13(pop,data,info,fit)
[pop_size,~] = size(pop);
for x = 1 : info.tl1
    [info,pop1] = RL_GA_cross3(pop,info,fit);
    [info,popout] = RL_GA_mutation2_11(pop1, info, data);
    fitnew = decode_01(popout,info,data);
    for i=1:pop_size
       if fitnew(i) < fit(i)
          pop(i,:)=popout(i,:);
          fit(i)=fitnew(i);
       end
    end
end
