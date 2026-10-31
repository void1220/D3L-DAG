% Legacy MATLAB scheduling routine.
function [info,popout] = RL_GA_cross3_eng(pop,info,fit)
[ps, len]=size(pop);
ncr=round(ps*info.pc);
popout=pop;
f=1./fit;
f=f./sum(f);
f=cumsum(f);
for i=1 : 2 : ncr
    i1=find(rand<=f,1,'first');
    i2=find(rand<=f,1,'first');
    j=randi([len/3*2+1 len-2]);
    k=randi([j+1 len-1]);
    m=randi([k+1 len]);
    popout(i1,:)=[pop(i1,1:j) pop(i2,j+1:k) pop(i1,k+1:m) pop(i2,m+1:end)];
    popout(i2,:)=[pop(i2,1:j) pop(i1,j+1:k) pop(i2,k+1:m) pop(i1,m+1:end)];
end
