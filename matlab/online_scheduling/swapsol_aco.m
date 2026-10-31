% Legacy MATLAB scheduling routine.
function [info,solnew]=swapsol_aco(sol,data,info)
soltemp=sol;
fit=decode(sol,info,data);
for i=1:data.n-1
    for j=i+1:data.n
        soltemp(i)=sol(j);
        soltemp(j)=sol(i);
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
    end
end
solnew=sol;
