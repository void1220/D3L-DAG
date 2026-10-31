% Legacy MATLAB scheduling routine.
function [energy,time,fit_curve]=ILS(info,data)
dataheft = data;
info.m = dataheft.m;
info.n = size(dataheft.ct,1);
info.cmax = data.cmax;
dataheft.aft = dataheft.hx;
dataheft.w = dataheft.ct;
dataheft.ckef = dataheft.c;
dataheft.c = dataheft.st;
dataheft.pre = dataheft.xh;
dataheft.pkind = dataheft.p;
dataheft.ps = dataheft.pkind;
info.rank=rankqueen1(dataheft,info);
sch=heft(dataheft,info);
sol=initial_sol(sch,data);
info.fitbest=decode(sol,info,data);
info.solbest=sol;
info.tl = 1000;
fit_curve = zeros(1, info.tl*100);
index = 1;
tic
for t=1 : info.tl
    [info,solnew]=swapsol_aco(sol,data,info);
    fit_curve(index) = decode(solnew,info,data);
    index = index + 1;
    time_cost=toc;
    if time_cost>=info.maxrt
        break
    end
    [info,solnew]=insert_aco(solnew,data,info);
    fit_curve(index) = decode(solnew,info,data);
    index = index + 1;
    time_cost=toc;
    if time_cost>=info.maxrt
        break
    end
    [info,solnew]=mutate_aco(solnew,data,info);
    fit_curve(index) = decode(solnew,info,data);
    index = index + 1;
    time_cost=toc;
    if time_cost>=info.maxrt
        break
    end
    [info,solnew]=swapsol_seq(solnew,data,info);
    fit_curve(index) = decode(solnew,info,data);
    index = index + 1;
    time_cost=toc;
    if time_cost>=info.maxrt
        break
    end
    [info,solnew]=insert_seq(solnew,data,info);
    fit_curve(index) = decode(solnew,info,data);
    index = index + 1;
    time_cost=toc;
    if time_cost>=info.maxrt
        break
    end
    [info,solnew]=swapsol_eng(solnew,data,info);
    fit_curve(index) = decode(solnew,info,data);
    index = index + 1;
    time_cost=toc;
    if time_cost>=info.maxrt
        break
    end
    [info,solnew]=insert_eng(solnew,data,info);
    fit_curve(index) = decode(solnew,info,data);
    index = index + 1;
    time_cost=toc;
    if time_cost>=info.maxrt
        break
    end
    [info,solnew]=mutate_eng(solnew,data,info);
    fit_curve(index) = decode(solnew,info,data);
    index = index + 1;
    if t>=3
        solnew=info.solbest;
    end
    time_cost=toc;
    if time_cost>=info.maxrt
        break
    end
end
if info.num <= 2
    sol=opt_energy_IILS(solnew,data,info,sch);
else
    sol = opt_energy_ILS(solnew,data,info);
end
energy=decode(sol,info,data);
time=time_cost;
