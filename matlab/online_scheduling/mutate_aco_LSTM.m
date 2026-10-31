% Legacy MATLAB scheduling routine.
function [info,solnew]=mutate_aco_LSTM(sol,data,info)
soltemp=sol;
fit=decode_AILS(sol,info,data);
for i=1:data.n-1
    soltemp(i)=randi(info.m);
    fittemp=decode_AILS(soltemp,info,data);
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
solnew=info.solbest;
