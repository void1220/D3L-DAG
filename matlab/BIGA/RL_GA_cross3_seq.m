% Legacy MATLAB scheduling routine.
function [info,popout] = RL_GA_cross3_seq(pop,info)
[ps, len]=size(pop);
ncr=round(ps*info.pc);
popout=pop;
for i=1 : 2 : ncr
    i1=randi([1 ps]);
    i2=randi([1 ps]);
    if i1 == 12
        i2 = i1 -1;
    end
    j=randi([len/3+1 len/3*2-2]);
    k=randi([j+1 len/3*2-1]);
    m=randi([k+1 len/3*2]);
    popout(i1,:)=[pop(i1,1:j) pop(i2,j+1:k) pop(i1,k+1:m) pop(i2,m+1:end)];
    popout(i2,:)=[pop(i2,1:j) pop(i1,j+1:k) pop(i2,k+1:m) pop(i1,m+1:end)];
end
