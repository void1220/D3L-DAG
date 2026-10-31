% Legacy MATLAB scheduling routine.
function [info,solnew]=insert_eng(sol,data,info)
soltemp=sol;
fit=decode(sol,info,data);
temp2=randperm(data.n);
j=temp2(1);
i=temp2(2);
soltemp(data.n+j)=sol(data.n+i);
soltemp(data.n+i:data.n+j-1)=sol(data.n+i+1:data.n+j);
fittemp=decode(soltemp,info,data);
if fittemp<fit
    sol=soltemp;
    if fittemp<info.fitbest
        info.solbest=sol;
        info.fitbest=fittemp;
    end
else
    temp=rand(1,1);
    if exp((fit-fittemp)/fit)>temp
        sol=soltemp;
    end
end
solnew=sol;
