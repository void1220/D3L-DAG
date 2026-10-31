% Legacy MATLAB scheduling routine.
function solnew=GA_fre(sol,info,data)
pop(1,:)=sol;
for i=2:info.np
    pop(i,1:info.n*2)=sol(1:info.n*2);
    pop(i,info.n*2+1:end)=rand(1,info.n);
end
fit=zeros(1,info.np);
fitnext=zeros(1,info.np);
for i=1:info.np
    fit(i)=decode_fre(pop(i,:),info,data);
end
for t=1:info.nt
    popnext_fre=GA_gen(pop(:,info.n*2+1:end),info,fit);
    popnext=[pop(:,1:info.n*2) popnext_fre];
    for i=1:info.np
        fitnext(i)=decode_fre(popnext(i,:),info,data);
    end
    pop_all=[pop;popnext];
    fit_all=[fit fitnext];
    [~,index]=sort(fit_all);
    pop=pop_all(index(1:info.np),:);
    fit=fit_all(index(1:info.np));
end
solnew=pop(1,:);
