% Legacy MATLAB scheduling routine.
function [info,pop,fit]=operator_9(pop,data,info,fit)
[pop_size,~] = size(pop);
if info.num<=4
    load Qres_small.mat
elseif (info.num>4)&&(info.num<8)
    load Qres_medium.mat
else
    load Qres_large.mat
end
fithis=zeros(info.ng,pop_size);
fitb=zeros(1,info.ng);
for x = 1:info.tl1
    [~,index]=sort(fit);
    fithis(x,:) = fit;
    fitb(x)=min(fit);
    popb=pop(index(1),:);
    s=getState(pop,popb,fit,info,fithis,x);
    [~,info.l]=max(Q(s,:));
    popout = GWO_gen(pop,info,fit);
    fitnew = decode_01(popout,info,data);
    for i=1:pop_size
       if fitnew(i) < fit(i)
          pop(i,:)=popout(i,:);
          fit(i)=fitnew(i);
       end
    end
end
