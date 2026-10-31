% Legacy MATLAB scheduling routine.
function [energy,time]=LGWO(info,data)
info.h=0.1;
info.np=40;
info.pc=0.8;
info.pm=0.15;
info.lamda=1;
info.cmax=data.cmax;
info.m=data.m;
info.n=data.n;
if info.num<=4
    load Qres_small.mat
elseif (info.num>4)&&(info.num<8)
    load Qres_medium.mat
else
    load Qres_large.mat
end
tic
pop=rand(info.np,info.n*3);
fit=zeros(1,info.np);
able=zeros(1,info.np);
for i=1:info.np
   [fit(i),able(i)]=decode_LGWO(pop(i,:),info,data);
end
fitnew=zeros(size(fit));
info.ng=6000;
fitb=zeros(1,info.ng);
adv=zeros(info.np,info.n);
fithis=zeros(info.ng,info.np);
for I=1:info.ng
    time_cost=toc;
    if time_cost>info.maxrt
        break
    end
    [~,index]=sort(fit);
    fithis(I,:)=fit;
    fitb(I)=min(fit);
    popb=pop(index(1),:);
    s=getState(pop,popb,fit,info,fithis,I);
    [~,info.l]=max(Q(s,:));
    poptemp=GWO_gen(pop,info,fit);
    fit_his(I)=min(fit);
    L_his(I)=info.l;
    for i=1:info.np
        [fitnew(i),able(i),adv(i,:)]=decode_LGWO(poptemp(i,:),info,data);
    end
    for i=1:info.np
       if fitnew(i)<fit(i)
          pop(i,:)=poptemp(i,:);
          fit(i)=fitnew(i);
       end
    end
    temp=['当前迭代次数为：',num2str(I),'，最佳适应度函数值为：',num2str(min(fit))];
    disp(temp)
end
energy=min(fit);
time=time_cost;
plot(fit_his)
plot(L_his)
