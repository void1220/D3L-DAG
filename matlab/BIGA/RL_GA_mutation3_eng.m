% Legacy MATLAB scheduling routine.
function [info,popout] = RL_GA_mutation3_eng(pop, info)
[ps, len]=size(pop);
nmu = round(ps*info.pm);
popout=pop;
for i = 1 : 2 : nmu
    j=randi([1 ps]);
    j0=randi([len/3 * 2+1 len-2]);
    j1=randi([j0+1 len-1]);
    j2=randi([j1+1 len]);
    popout(j,j1)=rand(1,1);
    popout(j,j2)=rand(1,1);
end
end
