% Legacy MATLAB scheduling routine.
function [info,popout] = RL_GA_mutation3_aco(pop, info)
[ps, len]=size(pop);
nmu = round(ps*info.pm);
popout=pop;
for i = 1 : 2 : nmu
    j=randi([1 ps]);
    j0=randi([1 len/3-2]);
    j1=randi([j0+1 len/3-1]);
    j2=randi([j1+1 len/3]);
    popout(j,j1)=rand();
    popout(j,j2)=rand();
end
end
