% Legacy MATLAB scheduling routine.
function [info,solnew]=swapsol_eng(sol,data,info)
soltemp=sol;
fit=decode(sol,info,data);
for i=1:data.n-1
    for j=i+1:data.n
        soltemp(data.n*2+i)=sol(data.n*2+j);
        soltemp(data.n*2+j)=sol(data.n*2+i);
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
solnew=info.solbest;
