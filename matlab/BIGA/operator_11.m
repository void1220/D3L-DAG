% Legacy MATLAB scheduling routine.
function [info,pop,fit]=operator_11(pop,data,info,fit)
dim = data.n * 3;
Lb=0.*ones(1,dim);
Ub=1.*ones(1,dim);
[pop_size,~] = size(pop);
for x = 1:info.tl3
    [info,pop1] = DE_main(pop,data,info,fit);
    [info,popout] = RL_GA_mutation2(pop1, info, data);
    for i = 1:pop_size
        I = popout(i,:) <= Lb;
        popout(i,I) = rand();
        J = popout(i,:) >= Ub;
        popout(i,J) = rand();
    end
    fitnew = decode_01(popout,info,data);
    for i=1:pop_size
       if fitnew(i) < fit(i)
          pop(i,:)=popout(i,:);
          fit(i)=fitnew(i);
       end
    end
end
