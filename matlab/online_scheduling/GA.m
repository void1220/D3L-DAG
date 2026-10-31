% Legacy MATLAB scheduling routine.
function [energy,time]=GA(info,data)
pop=rand(info.np,data.n*3);
info.n=data.n;
info.m=data.m;
info.h=1;
info.cmax=data.cmax;
fit=zeros(1,info.np);
able=zeros(1,info.np);
for i=1:info.np
   [fit(i),able(i)]=decode_GA(pop(i,:),info,data,fit);
end
fitnew=zeros(size(fit));
fitb=zeros(1,info.ng);
adv=zeros(info.np,info.n);
fithis=zeros(info.ng,info.np);
tic
for I=1:info.ng
    time_cost=toc;
    if time_cost>info.maxrt
        break
    end
    fithis(I,:)=fit;
    fitb(I)=min(fit);
    popnext=GA_gen(pop,info,fit);
    for i=1:info.np
        [fitnew(i),able(i),adv(i,:)]=decode_GA(popnext(i,:),info,data,fit);
    end
    for i=1:info.np
       if fitnew(i)<fit(i)
          pop(i,:)=popnext(i,:);
          fit(i)=fitnew(i);
       end
    end
    temp=['当前迭代次数为：',num2str(I),'，最佳适应度函数值为：',num2str(min(fit))];
    disp(temp)
end
energy=min(fit);
time=time_cost;
