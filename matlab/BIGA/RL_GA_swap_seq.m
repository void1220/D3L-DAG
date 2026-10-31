% Legacy MATLAB scheduling routine.
function [info,popout] = RL_GA_swap_seq(pop,info)
[ps, len]=size(pop);
ncr=round(ps*info.pc);
popout=pop;
for iter = 1 : 2 : ncr
    i = randi([1 ps]);
    j=randi([len/3+1 len/3*2-1]);
    k=j+1;
    temp = popout(i,j);
    popout(i,j) = popout(i,k);
    popout(i,k) = temp;
end
