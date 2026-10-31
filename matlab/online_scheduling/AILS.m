% Legacy MATLAB scheduling routine.
function [energy,time]=AILS(info,data)
dataheft = data;
info.m = dataheft.m;
info.n = size(dataheft.ct,1);
info.cmax = data.cmax;
info=setuppara(info);
dataheft.aft = dataheft.hx;
dataheft.w = dataheft.ct;
dataheft.ckef = dataheft.c;
dataheft.c = dataheft.st;
dataheft.pre = dataheft.xh;
dataheft.pkind = dataheft.p;
dataheft.ps = dataheft.pkind;
info.rank=rankqueen1(dataheft,info);
sch=heft(dataheft,info);
if info.mode==1
    temp1=['load seq_fft',num2str(info.num),'_net.mat'];
    temp2=['load aco_fft',num2str(info.num),'_net.mat'];
    eval(temp1)
    eval(temp2)
else
    temp1=['load seq_gs',num2str(info.num),'_net.mat'];
    temp2=['load aco_gs',num2str(info.num),'_net.mat'];
    eval(temp1)
    eval(temp2)
end
if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    dlX = gpuArray(dlX);
end
info.nt=100;
info.pc=0.9;
info.pm=0.2;
sol=initial_sol(sch,data);
sol = loc_energy(sol,data,info);
fit = decode(sol,info,data);
info.fitbest=fit;
info.solbest=sol;
tic
for i=1:info.tl
    time_cost=toc;
    if time_cost>info.maxrt
        break
    end
    [info,solnew]=swapsol_aco_LSTM(sol,data,info,aco_net);
    [info,solnew]=mutate_aco_LSTM(solnew,data,info);
    [info,solnew]=mutate_eng(solnew,data,info);
    solnew=GA_fre(solnew,info,data);
end
sol=opt_energy_AILS(solnew,data,info);
energy=decode_AILS(sol,info,data);
time=time_cost;
