% Legacy MATLAB scheduling routine.
function [info,pop,fit]=operator_16(pop,data,info,fit)
[pop_size,~] = size(pop);
for x = 1 : info.tl2
    [info,pop1] = RL_GA_cross2(pop,info,fit);
    [info,pop2] = RL_GA_mutation3_aco(pop1, info);
    [info,pop3] = RL_GA_swap_seq(pop2,info);
    [info,pop4] = RL_GA_cross3_seq(pop3,info);
    [info,popout] = RL_GA_mutation3_eng(pop4, info);
    fitnew = decode_01(popout,info,data);
    for i=1:pop_size
       if fitnew(i) < fit(i)
          pop(i,:)=popout(i,:);
          fit(i)=fitnew(i);
       end
    end
end
