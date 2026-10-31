% Legacy MATLAB scheduling routine.
function sol=initial_sol(sch,data)
sol(1:data.n)=sch.xij;
[~,sol(data.n+1:data.n*2)]=sort(sch.st/(max(sch.st)+1));
sol(data.n*2+1:data.n*3)=ones(1,data.n);
