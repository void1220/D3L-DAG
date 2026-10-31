% Legacy MATLAB scheduling routine.
function [info,solnew,fit]=swapsol_aco(sol,data,info)
fit=decode(sol,info,data);
x = rand();
if info.num<4
    Matching_Coefficient = (1 - x) * 0.1 + 0.35;
elseif info.num<8
    Matching_Coefficient = (1 - x) * 0.1 + 0.25;
elseif info.num<10
    Matching_Coefficient = (1 - x) * 0.1 + 0.15;
elseif info.num<=16
    Matching_Coefficient = (1 - x) * 0.05;
end
for i=1:data.n-1
    if rand()<Matching_Coefficient
        for j=i+1:data.n
            soltemp=sol;
            soltemp(i) = sol(j);
            soltemp(j) = sol(i);
            fittemp = decode(soltemp,info,data);
            if fittemp<fit
                sol = soltemp;
                fit = fittemp;
            end
        end
    end
end
solnew=sol;
end
